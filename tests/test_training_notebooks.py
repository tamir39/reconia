import ast
import json
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("roberta", "bros", "pick", "layoutlmv3", "donut", "qwen_vl")
BRANCH = "dev"


def notebook(name):
    return json.loads((ROOT / f"notebooks/{name}_model.ipynb").read_text(encoding="utf-8"))


class TrainingNotebookTests(unittest.TestCase):
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
                if name in ("layoutlmv3", "donut", "qwen_vl"):
                    self.assertIn("verify_baseline_data(DATASET, get_cache_dir(DATASET))", "".join(cells[4]["source"]))
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


if __name__ == "__main__":
    unittest.main()
