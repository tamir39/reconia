import ast
import importlib
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("roberta", "bros", "pick", "layoutlmv3", "donut", "qwen_vl")
BRANCH = "dev"


def notebook(name):
    return json.loads((ROOT / f"notebooks/{name}_model.ipynb").read_text(encoding="utf-8"))


class TrainingNotebookTests(unittest.TestCase):
    def test_shared_sections_have_identical_names_and_format(self):
        reference = notebook("roberta")["cells"]
        for name in MODELS:
            with self.subTest(model=name):
                cells = notebook(name)["cells"]
                self.assertEqual([c["cell_type"] for c in cells], [c["cell_type"] for c in reference])
                for index in (1, 2, 3, 5, 10, 12):
                    self.assertEqual(cells[index]["source"], reference[index]["source"])
                for index in (6, 8):
                    self.assertEqual(cells[index]["source"][0], reference[index]["source"][0])
                self.assertEqual(len({c["id"] for c in cells}), len(cells))
                expected_history = "".join(reference[11]["source"]).replace('"roberta"', f'"{name}"')
                self.assertEqual("".join(cells[11]["source"]), expected_history)

    def test_cells_match_trainer_and_baseline_contract(self):
        for name in MODELS:
            with self.subTest(model=name):
                cells = notebook(name)["cells"]
                self.assertEqual(len(cells), 14)
                for cell in cells:
                    if cell["cell_type"] == "code":
                        ast.parse("".join(cell["source"]))
                        self.assertIsNone(cell["execution_count"])
                        self.assertEqual(cell["outputs"], [])
                self.assertIn(f'GIT_REF = "{BRANCH}"', "".join(cells[2]["source"]))
                self.assertIn('DATASET = "all"', "".join(cells[2]["source"]))
                if name in ("layoutlmv3", "donut", "qwen_vl"):
                    self.assertIn("verify_baseline_data(dataset, get_cache_dir(dataset))", "".join(cells[4]["source"]))
                if name != "pick":
                    self.assertIn(f'download_weights("{name}")', "".join(cells[7]["source"]))
                self.assertIn("RUN_TEST = False", "".join(cells[13]["source"]))
                tree = ast.parse((ROOT / f"models/{name}/train.py").read_text(encoding="utf-8"))
                train = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "train_model")
                defaults = dict(zip(
                    [arg.arg for arg in train.args.args][-len(train.args.defaults):],
                    [ast.literal_eval(value) for value in train.args.defaults],
                ))
                call = next(node for node in ast.walk(ast.parse("".join(cells[9]["source"])))
                            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                            and node.func.id == "train_model")
                for keyword in call.keywords:
                    self.assertEqual(ast.literal_eval(keyword.value), defaults[keyword.arg])

    def test_default_training_dispatches_all_three_datasets_in_order(self):
        expected = ("sroie", "cord_v2", "wildreceipt")
        for name in MODELS:
            with self.subTest(model=name):
                trainer = importlib.import_module(f"models.{name}.train")
                namespace = {"datasets": expected}
                with patch.object(trainer, "train_model") as train:
                    exec(compile("".join(notebook(name)["cells"][9]["source"]), name, "exec"), namespace)
                    self.assertEqual([call.args[0] for call in train.call_args_list], list(expected))
                    self.assertTrue(all(not call.kwargs["resume"] for call in train.call_args_list))
                    train.reset_mock()
                    enabled = "".join(notebook(name)["cells"][13]["source"]).replace("RUN_TEST = False", "RUN_TEST = True")
                    exec(compile(enabled, name, "exec"), namespace)
                    self.assertEqual([call.args[0] for call in train.call_args_list], list(expected))
                    self.assertTrue(all(call.kwargs["test_only"] for call in train.call_args_list))

    def test_existing_wrong_branch_stops_before_pip_or_training(self):
        for name in MODELS:
            with self.subTest(model=name), patch("pathlib.Path.cwd", return_value=ROOT), \
                    patch("os.chdir") as chdir, patch("subprocess.run") as run, \
                    patch("subprocess.check_output", return_value="codex/original-data-six-models\n"):
                with self.assertRaisesRegex(RuntimeError, "Expected branch"):
                    exec(compile("".join(notebook(name)["cells"][2]["source"]), name, "exec"), {})
                run.assert_not_called()
                chdir.assert_not_called()

    def test_existing_correct_branch_is_reported_without_git_mutation(self):
        for name in MODELS:
            with self.subTest(model=name), patch("pathlib.Path.cwd", return_value=ROOT), \
                    patch("os.chdir"), patch("sys.path", list(__import__("sys").path)), \
                    patch("subprocess.run") as run, \
                    patch("subprocess.check_output", side_effect=[BRANCH + "\n", "a12d3e5\n"]):
                namespace = {}
                exec(compile("".join(notebook(name)["cells"][2]["source"]), name, "exec"), namespace)
                self.assertEqual(namespace["source_branch"], BRANCH)
                self.assertEqual(namespace["source_commit"], "a12d3e5")
                self.assertEqual(run.call_count, 1)
                self.assertIn("pip", run.call_args.args[0])

    def test_fresh_kaggle_clone_uses_dev_without_secret_in_arguments(self):
        secret = "synthetic-test-value"
        secrets = SimpleNamespace(UserSecretsClient=lambda: SimpleNamespace(get_secret=lambda _: secret))
        for name in MODELS:
            with self.subTest(model=name), \
                    patch("pathlib.Path.exists", autospec=True,
                          side_effect=lambda p: p.as_posix() == "/kaggle"), \
                    patch.dict(sys.modules, {"kaggle_secrets": secrets}), \
                    patch("os.chdir"), patch("sys.path", list(sys.path)), \
                    patch("subprocess.run") as run, \
                    patch("subprocess.check_output", side_effect=[BRANCH + "\n", "test-commit\n"]):
                namespace = {}
                exec(compile("".join(notebook(name)["cells"][2]["source"]), name, "exec"), namespace)
                clone = run.call_args_list[0]
                self.assertEqual(clone.args[0][:2], ["git", "clone"])
                self.assertEqual(clone.args[0][clone.args[0].index("--branch") + 1], BRANCH)
                self.assertNotIn(secret, " ".join(clone.args[0]))
                self.assertIn("GIT_CONFIG_VALUE_0", clone.kwargs["env"])
                self.assertNotIn("token", namespace)
                self.assertNotIn("auth", namespace)
                self.assertEqual(run.call_count, 2)  # clone + mocked pip only

    @unittest.skipUnless(
        all((ROOT / ".cache" / dataset / "train.jsonl").is_file()
            and (ROOT / "data/raw" / dataset).is_dir()
            for dataset in ("sroie", "cord_v2", "wildreceipt")),
        "Requires the locally preserved original datasets and prepared splits",
    )
    def test_data_preview_train_history_and_test_dispatch_for_all_datasets(self):
        from PIL import Image
        from utils.common import get_cache_dir, get_data_dir

        for name in MODELS:
            cells = notebook(name)["cells"]
            trainer = importlib.import_module(f"models.{name}.train")
            for dataset in ("sroie", "cord_v2", "wildreceipt"):
                with self.subTest(model=name, dataset=dataset):
                    namespace = {"DATASET": dataset, "repo_dir": ROOT, "display": lambda _: None}
                    prepare_module = importlib.import_module(f"prepare.prepare_{dataset}")
                    with patch.object(prepare_module, f"prepare_{dataset}") as prepare:
                        exec(compile("".join(cells[4]["source"]), name, "exec"), namespace)
                        prepare.assert_called_once_with()
                    with patch("IPython.display.display"):
                        exec(compile("".join(cells[5]["source"]), name, "exec"), namespace)
                    image_path = namespace["image_path"]
                    self.assertTrue(image_path.is_file())
                    with Image.open(image_path) as image:
                        image.verify()
                    with (get_cache_dir(dataset) / "train.jsonl").open(encoding="utf-8") as file:
                        first = json.loads(file.readline())
                    self.assertEqual(namespace["sample"], first)
                    self.assertTrue(str(image_path).startswith(str(get_data_dir(dataset))))
                    with patch.object(trainer, "train_model") as train:
                        exec(compile("".join(cells[9]["source"]), name, "exec"), namespace)
                        self.assertEqual(train.call_args.args, (dataset,))
                        self.assertFalse(train.call_args.kwargs["resume"])
                        train.reset_mock()
                        exec(compile("".join(cells[13]["source"]), name, "exec"), namespace)
                        train.assert_not_called()
                        enabled = "".join(cells[13]["source"]).replace("RUN_TEST = False", "RUN_TEST = True")
                        exec(compile(enabled, name, "exec"), namespace)
                        self.assertTrue(train.call_args.kwargs["test_only"])
                        self.assertEqual(train.call_args.args, (dataset,))
                    with patch("pandas.read_csv", return_value=[]) as read:
                        exec(compile("".join(cells[11]["source"]), name, "exec"), namespace)
                        read.assert_called_once_with(ROOT / "results" / dataset / name / "history.csv")


if __name__ == "__main__":
    unittest.main()
