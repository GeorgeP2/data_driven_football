"""Scaffold a new football project from ``_template``.

Usage:
    python scripts/new_project.py "Expected Goals Model" --category ml
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_NAME = "_template"
INDEX_END = "<!-- projects:end -->"
RESERVED = {"football", "src", "tests", "docs", "scripts"}

CATEGORIES = {
    "analytics": "Analytics & Visualisation",
    "stats": "Statistics & Experimentation",
    "ml": "Classical Machine Learning",
    "ts": "Time Series & Forecasting",
    "dl": "Deep Learning",
    "opt": "Optimisation & simulation",
    "nlp": "Natural Language Processing",
    "llm": "LLMs & Generative AI",
    "de": "Data Engineering",
}


def package_name(title: str) -> str:
    package = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    if package and package[0].isdigit():
        package = f"p_{package}"
    return package


def existing_packages(root: Path) -> set[str]:
    return {
        p.name for p in root.glob("*/src/*") if p.is_dir() and not p.name.startswith(("_", "."))
    }


def render(path: Path, subs: dict[str, str]) -> None:
    text = path.read_text()
    for key, value in subs.items():
        text = text.replace("{{" + key + "}}", value)
    path.write_text(text)


def scaffold(title: str, category: str, root: Path = ROOT) -> tuple[str, str]:
    """Copy the template to ``<package>_project/`` and add a row to the README.

    Returns (slug, package).
    """
    package = package_name(title)
    if not package:
        raise ValueError("title must contain letters or digits")
    if package in RESERVED or package in sys.stdlib_module_names:
        raise ValueError(
            f"package name {package!r} clashes with a reserved or standard-library name"
        )
    if package in existing_packages(root):
        raise ValueError(f"a project package called {package!r} already exists")
    slug = f"{package}_project"
    dest = root / slug
    if dest.exists():
        raise ValueError(f"{slug}/ already exists")

    shutil.copytree(root / TEMPLATE_NAME, dest)
    (dest / "src" / "{{package}}").rename(dest / "src" / package)

    subs = {"title": title, "slug": slug, "package": package, "category": CATEGORIES[category]}
    for f in dest.rglob("*"):
        if f.is_file() and f.suffix in {".md", ".py", ".ipynb"}:
            render(f, subs)

    readme = root / "README.md"
    row = f"| [{title}]({slug}) | {CATEGORIES[category]} | 🚧 | _TBC_ |\n"
    readme.write_text(readme.read_text().replace(INDEX_END, row + INDEX_END))
    return slug, package


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("title", help="Human-readable project title")
    parser.add_argument("--category", choices=sorted(CATEGORIES), default="ml")
    args = parser.parse_args()

    try:
        slug, package = scaffold(args.title, args.category)
    except ValueError as e:
        parser.error(str(e))

    print(f"Created {slug}/  (package: {package})")
    print(f"  cd {slug} && PYTHONPATH=src python -m {package}.run")


if __name__ == "__main__":
    main()
