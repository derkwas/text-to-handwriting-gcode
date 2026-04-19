#!/usr/bin/env python3
"""Merge glyph folders into one folder with unified index.json.

Default behavior merges glyphs_hw and glyphs into glyphs_merged.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def safe_label_dir(label: str) -> str:
    """Convert glyph label to a Windows-safe directory name."""
    if label == "":
        return "empty"
    if len(label) == 1 and label.isalpha() and label.islower():
        return label
    if len(label) == 1 and label.isalpha() and label.isupper():
        return f"upper_{label}"

    named = {
        " ": "space",
        "/": "slash",
        "\\": "backslash",
        ":": "colon",
        "*": "asterisk",
        "?": "question",
        '"': "quote",
        "<": "lt",
        ">": "gt",
        "|": "pipe",
    }
    if label in named:
        return named[label]

    out = []
    for ch in label:
        if ch.isalnum() or ch in ("_", "-"):
            out.append(ch)
        else:
            out.append(f"u{ord(ch):04x}")
    s = "".join(out).strip(".")
    return s or "glyph"


def load_index(lib: Path) -> dict:
    idx = lib / "index.json"
    if not idx.exists():
        return {"version": 2, "glyphs": {}}
    with open(idx, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge glyph libraries")
    parser.add_argument("--inputs", default="glyphs_hw,glyphs", help="Comma-separated input folders")
    parser.add_argument("--output", default="glyphs_merged", help="Output merged folder")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    inputs = [((base / p.strip()).resolve()) for p in args.inputs.split(",") if p.strip()]
    out = (base / args.output).resolve()

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    merged: dict[str, list[dict]] = {}
    seen_paths: set[str] = set()

    copied = 0
    for lib in inputs:
        index = load_index(lib)
        glyphs = index.get("glyphs", {})
        lib_tag = lib.name

        for label, entries in glyphs.items():
            bucket = merged.setdefault(label, [])
            label_dir = out / safe_label_dir(label)
            label_dir.mkdir(parents=True, exist_ok=True)

            for entry in entries:
                if isinstance(entry, dict):
                    rel = str(entry.get("path", ""))
                    orig_h = entry.get("orig_h")
                else:
                    rel = str(entry)
                    orig_h = None

                if not rel:
                    continue
                src = lib / rel
                if not src.exists():
                    continue

                src_name = src.name
                stem = Path(src_name).stem
                suffix = Path(src_name).suffix
                dst_name = f"{lib_tag}_{stem}{suffix}"
                dst = label_dir / dst_name

                n = 1
                while dst.exists():
                    dst_name = f"{lib_tag}_{stem}_{n}{suffix}"
                    dst = label_dir / dst_name
                    n += 1

                shutil.copy2(src, dst)
                copied += 1

                rel_out = str(Path(label_dir.name) / dst.name).replace("\\", "/")
                key = f"{label}|{rel_out}"
                if key in seen_paths:
                    continue
                seen_paths.add(key)
                bucket.append({"path": rel_out, "orig_h": orig_h})

    for label in merged:
        merged[label].sort(key=lambda x: x["path"])

    out_index = {
        "version": 2,
        "glyphs": dict(sorted(merged.items(), key=lambda kv: kv[0])),
    }

    with open(out / "index.json", "w", encoding="utf-8") as f:
        json.dump(out_index, f, ensure_ascii=False, indent=2)

    label_count = len(out_index["glyphs"])
    variant_count = sum(len(v) for v in out_index["glyphs"].values())
    print(f"Merged -> {out}")
    print(f"Labels: {label_count}")
    print(f"Variants: {variant_count}")
    print(f"Files copied: {copied}")


if __name__ == "__main__":
    main()
