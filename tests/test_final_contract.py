import ast
from io import BytesIO
import itertools
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image
import torch

from demo.assets import ROOT
from demo.occlusion import crf_log_marginals, measure
from tools.export_submission import allowed
from tools.verify_final import canonical_rows, verify, verify_prepared
from utils.common import ensure_training_destination, export_labels
from utils.metrics import evaluate_dataset, evaluate_field


class ScoringTests(unittest.TestCase):
    def test_micro_not_macro(self):
        result = evaluate_dataset([{"a": ["x", "y"], "b": ["z"]}], [{"a": ["x"], "b": ["Z"]}])
        self.assertEqual(result["precision"], 1)
        self.assertAlmostEqual(result["recall"], 2 / 3)
        self.assertAlmostEqual(result["f1"], .8)

    def test_duplicates_are_not_collapsed(self):
        self.assertEqual(evaluate_field([" A ", "a"], ["a", "A", "a"]), (2, 1, 0))

    def test_wrong_field_does_not_match(self):
        self.assertEqual(evaluate_dataset([{"total": ["10"]}], [{"subtotal": ["10"]}])["f1"], 0)

    def test_missing_documents_are_rejected(self):
        for truth, predicted in (([], []), ([{}], []), ([{}], [{}, {}])):
            with self.subTest(truth=truth, predicted=predicted), self.assertRaises(ValueError):
                evaluate_dataset(truth, predicted)


class FinalSourceTests(unittest.TestCase):
    def test_benchmarks_examples_and_source_provenance(self):
        result = verify()
        self.assertEqual((result["model_dataset_pairs"], result["examples"]), (15, 30))

    def test_new_training_cannot_overwrite_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            ensure_training_destination(root, "sroie", "donut", False)
            path = root / "checkpoints/sroie/donut/best_weights.pt"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"preserved")
            with self.assertRaises(FileExistsError):
                ensure_training_destination(root, "sroie", "donut", False)
            ensure_training_destination(root, "sroie", "donut", True)
            self.assertEqual(path.read_bytes(), b"preserved")

    def test_guard_precedes_trainer_side_effects(self):
        for path in (ROOT / "models").glob("*/train.py"):
            module = ast.parse(path.read_text())
            function = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "train_model")
            # Exercise the real initial control flow without allocating a model or reading data.
            function = ast.FunctionDef(name="train_model", args=function.args, body=function.body[:3], decorator_list=[])
            namespace = {"set_seed": Mock(), "get_repo_dir": Mock(return_value=ROOT),
                         "ensure_training_destination": Mock(side_effect=FileExistsError("preserved"))}
            exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(path), "exec"), namespace)
            with self.subTest(model=path.parent.name), self.assertRaises(FileExistsError):
                namespace["train_model"]("sroie")

    def test_five_notebooks_are_disabled_and_loop_three_datasets(self):
        paths = list((ROOT / "notebooks").glob("*.ipynb"))
        self.assertEqual(len(paths), 5)
        for path in paths:
            notebook = json.loads(path.read_text(encoding="utf-8"))
            codes = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
            for code in codes:
                ast.parse(code)
            flags = {node.targets[0].id: ast.literal_eval(node.value)
                     for node in ast.parse(codes[0]).body if isinstance(node, ast.Assign)
                     and isinstance(node.targets[0], ast.Name) and node.targets[0].id.startswith("RUN_")}
            self.assertEqual(flags, dict.fromkeys(("RUN_PREPARE", "RUN_DOWNLOAD", "RUN_TRAIN", "RUN_EXPORT"), False))
            self.assertIn('GIT_REF = "final-submit"', codes[0])
            self.assertNotIn("b64decode", "".join(codes))
            trainer = SimpleNamespace(train_model=Mock())
            namespace = {"RUN_TRAIN": False, "MODEL": path.stem.removesuffix("_model"),
                         "datasets": ("sroie", "cord_v2", "wildreceipt"), "TRAIN_CONFIG": {},
                         "import_module": Mock(return_value=trainer)}
            exec(codes[3], namespace)
            trainer.train_model.assert_not_called()
            namespace["RUN_TRAIN"] = True
            exec(codes[3], namespace)
            self.assertEqual([call.args[0] for call in trainer.train_model.call_args_list], list(namespace["datasets"]))
            self.assertTrue(all(call.kwargs["resume"] is False for call in trainer.train_model.call_args_list))

    def test_submission_excludes_local_and_secret_assets(self):
        for name in (".env", "demo/.env.local", "checkpoints/sroie/a.pt", ".cache/a.py", ".venv/a.py",
                     "output/a.pdf", "data/raw/receipt.jpg", "weights/model.bin", "archive/private.zip", "run.lnk"):
            self.assertFalse(allowed(name), name)
        for name in ("run.bat", "demo/model_assets/donut/config.json", "reports/benchmarks/sroie/donut/history.csv"):
            self.assertTrue(allowed(name), name)

    def test_prepared_gate_detects_changed_content(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for split in ("train", "val", "test"):
                source = root / f"{split}.jsonl"
                source.write_text('{"id":"kept","ocr":[],"gt":{}}\n')
            with patch("utils.common.get_cache_dir", return_value=root), self.assertRaises(ValueError):
                verify_prepared(("sroie",))

    def test_canonical_hash_keeps_order_but_ignores_line_endings(self):
        with tempfile.TemporaryDirectory() as folder:
            a, b = Path(folder) / "a", Path(folder) / "b"
            a.write_bytes(b'{"id":1}\r\n{"id":2}\r\n')
            b.write_bytes(b'{ "id": 1 }\n{"id":2}\n')
            self.assertEqual(canonical_rows(a)[1], canonical_rows(b)[1])
            b.write_text('{"id":2}\n{"id":1}\n')
            self.assertNotEqual(canonical_rows(a)[1], canonical_rows(b)[1])

    def test_observed_labels_are_preserved(self):
        with tempfile.TemporaryDirectory() as folder, patch("utils.common.get_repo_dir", return_value=Path(folder)):
            export_labels({"B-total", "O"}, "sroie")
            self.assertEqual(json.loads((Path(folder) / "checkpoints/sroie/labels.json").read_text()), ["O", "B-total"])

    def test_cord_image_bytes_materialized_without_changing_annotations(self):
        from prepare.prepare_cord_v2 import export_split, process_sample
        buffer = BytesIO()
        Image.new("RGB", (10, 20), "white").save(buffer, "PNG")
        raw = {"ground_truth": json.dumps({"meta": {"image_id": 7, "image_size": {"width": 10, "height": 20}},
                                         "valid_line": [], "gt_parse": {"total": {"total_price": "10"}}}),
               "image": {"bytes": buffer.getvalue(), "path": None}}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            labels = export_split([raw], "test", root / "test.jsonl", root)
            self.assertEqual(json.loads((root / "test.jsonl").read_text()), process_sample(raw, "test"))
            self.assertEqual(labels, set())
            with Image.open(root / "images/test/7.png") as image:
                self.assertEqual(image.size, (10, 20))


class ExplanationTests(unittest.TestCase):
    def test_crf_marginals_match_exhaustive_paths(self):
        torch.manual_seed(42)
        emissions = torch.randn(3, 2, dtype=torch.double)
        crf = SimpleNamespace(transitions=torch.randn(2, 2, dtype=torch.double),
                              start_transitions=torch.randn(2, dtype=torch.double), end_transitions=torch.randn(2, dtype=torch.double))
        paths = list(itertools.product(range(2), repeat=3))
        scores = torch.stack([crf.start_transitions[path[0]] + crf.end_transitions[path[-1]] +
                              sum(emissions[i, tag] for i, tag in enumerate(path)) +
                              sum(crf.transitions[path[i], path[i + 1]] for i in range(2)) for path in paths])
        probabilities = scores.softmax(0)
        expected = torch.tensor([[sum(float(probabilities[k]) for k, path in enumerate(paths) if path[i] == tag)
                                  for tag in range(2)] for i in range(3)], dtype=torch.double)
        torch.testing.assert_close(crf_log_marginals(emissions, crf).exp(), expected)

    def test_occlusion_only_marks_supporting_regions(self):
        pixels = torch.tensor([[[[1., 0.], [0., 0.]]]])
        evaluate = lambda inputs: float(inputs["pixels"][0, 0, 0, 0])
        result = measure({"pixels": pixels}, "pixels", torch.zeros_like(pixels), evaluate, 1., [], lambda x, _: x, (2, 2), "image")
        self.assertEqual(result["values"], [1., 0., 0., 0.])
        self.assertTrue(result["has_signal"])

    def test_weak_signal_is_not_painted_strong(self):
        pixels = torch.ones(1, 1, 2, 2)
        result = measure({"pixels": pixels}, "pixels", torch.zeros_like(pixels),
                         lambda inputs: .001 * float(inputs["pixels"].mean()), .001,
                         [], lambda x, _: x, (2, 2), "image")
        self.assertFalse(result["has_signal"])
        self.assertLess(result["max_score_drop"], .005)


if __name__ == "__main__":
    unittest.main()
