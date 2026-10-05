import importlib.util
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("new_project", ROOT / "scripts" / "new_project.py")
assert spec and spec.loader
new_project = importlib.util.module_from_spec(spec)
spec.loader.exec_module(new_project)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    shutil.copytree(ROOT / "_template", tmp_path / "_template")
    shutil.copy(ROOT / "README.md", tmp_path / "README.md")
    (tmp_path / "lms_project" / "src" / "lms").mkdir(parents=True)
    return tmp_path


def test_scaffold_fills_template_and_readme(repo: Path):
    slug, package = new_project.scaffold("Expected Goals Model", "ml", root=repo)
    assert (slug, package) == ("expected_goals_model_project", "expected_goals_model")
    dest = repo / slug
    assert (dest / "src" / package / "run.py").is_file()
    for f in dest.rglob("*"):
        if f.is_file() and f.suffix in {".md", ".py", ".ipynb"}:
            assert "{{" not in f.read_text(), f
    readme = (repo / "README.md").read_text()
    assert f"| [Expected Goals Model]({slug}) | Classical Machine Learning |" in readme
    assert readme.index(slug) < readme.index(new_project.INDEX_END)


@pytest.mark.parametrize("title", ["LMS", "random", "Football", "!!!"])
def test_scaffold_rejects_clashing_names(repo: Path, title: str):
    with pytest.raises(ValueError):
        new_project.scaffold(title, "ml", root=repo)


def test_package_name_handles_leading_digit():
    assert new_project.package_name("2026 World Cup") == "p_2026_world_cup"
