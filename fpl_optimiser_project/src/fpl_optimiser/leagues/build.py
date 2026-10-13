"""Render outputs/league_pages/<league>.json into <league>_fpl.html using template.html.

    PYTHONPATH=src python -m fpl_optimiser.leagues.build

Commentary for a gameweek lives in commentary/<league>/gwNN.md: plain paragraphs, with
**bold** as the only markup. A gameweek with no file shows "No write-up for this week yet."
"""

import html
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_DIR = Path("outputs/league_pages")
COMMENTARY_DIR = Path("commentary")


def markdown_paragraphs(text: str) -> str:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    rendered = []
    for paragraph in paragraphs:
        body = html.escape(" ".join(paragraph.split()), quote=False)
        rendered.append("<p>" + re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", body) + "</p>")
    return "".join(rendered)


def load_commentary(league: str) -> dict[str, str]:
    commentary = {}
    for path in sorted((COMMENTARY_DIR / league).glob("gw*.md")):
        gw = int(path.stem.removeprefix("gw"))
        commentary[str(gw)] = markdown_paragraphs(path.read_text())
    return commentary


def main() -> None:
    template = (HERE / "template.html").read_text()
    sources = sorted(OUT_DIR.glob("*.json"))
    if not sources:
        raise SystemExit(f"No league JSON in {OUT_DIR}; run fpl_optimiser.leagues.export first.")
    for source in sources:
        payload = json.loads(source.read_text())
        meta = {**payload["meta"], "commentary": load_commentary(source.stem)}
        played = set(payload["data"].get("gameweeks", {}))
        missing = sorted(int(gw) for gw in played - set(meta["commentary"]))
        page = (
            template.replace("__TITLE__", meta["leagueName"])
            .replace("__DATA__", json.dumps(payload["data"]))
            .replace("__META__", json.dumps(meta))
        )
        target = OUT_DIR / f"{source.stem}_fpl.html"
        target.write_text(page)
        note = f" (no commentary for GW{', GW'.join(map(str, missing))})" if missing else ""
        print(f"wrote {target}{note}")


if __name__ == "__main__":
    main()
