#!/usr/bin/env python3
"""Drop placeholder pages and stub scripts from the published site.

publish_to_github.py is not in this repository. Run this from the site
root after that external publisher writes files, and before pushing:

    python3 scripts/publish_guard.py

HTML that says "is being prepared", tool scripts that only print
"executed successfully", and the weekly shell template are removed.
JSON indexes are updated to match. Articles that do not match those
markers are kept. A script that fails to parse is left on disk and
omitted from the tools index; the process exits with an error.
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PLACEHOLDER = "is being prepared"
STUB_MARKER = "executed successfully"
WEEKLY_TEMPLATE = (
    "产出 1 个可直接用于生产的工具脚本",
    "补充知识图谱跨领域连接，提升图谱密度",
)
STUB_MAX_LINES = 80


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def is_placeholder_html(text: str) -> bool:
    return PLACEHOLDER in text


def is_stub_script(text: str) -> bool:
    return STUB_MARKER in text and text.count("\n") < STUB_MAX_LINES


def is_weekly_template(text: str) -> bool:
    return all(marker in text for marker in WEEKLY_TEMPLATE)


def parses(path: Path) -> bool:
    try:
        compile(read_text(path), str(path), "exec")
    except SyntaxError:
        return False
    return True


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(read_text(path))


def save_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def title_in_html(title: str, html: str) -> bool:
    needle = (title or "").strip()
    if not needle:
        return False
    return needle[:48] in html


def remove_matching(directory: Path, predicate, removed: list) -> None:
    if not directory.is_dir():
        return
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.name == "index.html":
            continue
        if predicate(read_text(path)):
            path.unlink()
            removed.append(str(path.relative_to(ROOT)))


def clean_pages(filename: str, key: str, folder: str) -> None:
    path = ROOT / "data" / filename
    data = load_json(path)
    if not data or key not in data:
        return
    kept = []
    for item in data[key]:
        url = item.get("url") or ""
        target = ROOT / folder / url
        if not target.is_file():
            continue
        html = read_text(target)
        if is_placeholder_html(html) or is_weekly_template(html):
            continue
        if PLACEHOLDER in (item.get("summary") or ""):
            continue
        title = item.get("title") or ""
        if title and not title_in_html(title, html):
            deep = item.get("deep_url")
            deep_path = (ROOT / folder / deep) if deep else None
            if not deep_path or not deep_path.is_file():
                continue
            deep_html = read_text(deep_path)
            if is_placeholder_html(deep_html) or not title_in_html(title, deep_html):
                continue
            item["url"] = deep
            html = deep_html
        deep = item.get("deep_url")
        if deep and title_in_html(title, html if item.get("url") == deep else ""):
            pass
        elif deep:
            deep_path = ROOT / folder / deep
            if deep_path.is_file():
                deep_html = read_text(deep_path)
                if not is_placeholder_html(deep_html) and title_in_html(title, deep_html):
                    item["url"] = deep
        kept.append(item)
    data[key] = kept
    save_json(path, data)


def script_meta(path: Path) -> dict:
    text = read_text(path)
    match = re.search(r'("""|\'\'\')(.*?)(\1)', text, re.S)
    title = path.stem
    description = title
    if match:
        lines = [line.strip() for line in match.group(2).splitlines() if line.strip()]
        if lines:
            title = lines[0].strip(" .")
            for line in lines[1:]:
                if line.lower().startswith(("usage", "python", "用法")):
                    continue
                description = line
                break
    week = re.search(r"w(\d+)", path.name)
    return {
        "name": path.name,
        "title": title[:160],
        "category": "工具",
        "description": description[:220],
        "url": f"scripts/{path.name}",
        "_week": int(week.group(1)) if week else -1,
        "_size": path.stat().st_size,
    }


def clean_scripts() -> list:
    path = ROOT / "data" / "tools.json"
    data = load_json(path) or {"tools": [], "scripts": []}
    broken = []
    entries = []
    previous = {item.get("name"): item for item in data.get("scripts") or []}
    scripts_dir = ROOT / "tools" / "scripts"
    for script in sorted(scripts_dir.glob("*.py")):
        if not parses(script):
            broken.append(script.name)
            continue
        meta = script_meta(script)
        prior = previous.get(script.name) or {}
        prior_desc = prior.get("description") or ""
        if prior_desc and STUB_MARKER not in prior_desc and "auto-generated" not in prior_desc:
            meta["title"] = prior.get("title") or meta["title"]
            meta["description"] = prior_desc
            meta["category"] = prior.get("category") or meta["category"]
        entries.append(meta)
    entries.sort(key=lambda item: (-item["_week"], -item["_size"], item["name"]))
    data["scripts"] = [
        {k: v for k, v in item.items() if not k.startswith("_")} for item in entries
    ]
    save_json(path, data)
    return broken


def clean_updates() -> None:
    path = ROOT / "data" / "index.json"
    data = load_json(path)
    if not data:
        return
    kept = []
    for item in data.get("updates") or []:
        blob = " ".join(str(item.get(key) or "") for key in ("summary", "description", "title"))
        if PLACEHOLDER in blob or STUB_MARKER in blob:
            continue
        url = item.get("url") or ""
        if "/" in url and url.endswith(".html"):
            target = ROOT / url
            if not target.is_file():
                continue
            text = read_text(target)
            if is_placeholder_html(text) or is_weekly_template(text):
                continue
        kept.append(item)
    data["updates"] = kept
    save_json(path, data)


def main() -> int:
    removed = []
    remove_matching(ROOT / "papers", is_placeholder_html, removed)
    remove_matching(ROOT / "practices", is_placeholder_html, removed)
    remove_matching(ROOT / "weekly", is_weekly_template, removed)
    remove_matching(ROOT / "tools" / "scripts", is_stub_script, removed)
    clean_pages("papers.json", "papers", "papers")
    clean_pages("practices.json", "practices", "practices")
    clean_pages("weekly.json", "reports", "weekly")
    broken = clean_scripts()
    clean_updates()
    if removed:
        print("unpublished:")
        for name in removed:
            print(f"  {name}")
    else:
        print("no placeholder or stub files")
    if broken:
        print("unparseable scripts omitted from the tools index:", file=sys.stderr)
        for name in broken:
            print(f"  {name}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
