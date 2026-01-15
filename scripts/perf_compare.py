#!/usr/bin/env python3
import json
import sys
from pathlib import Path


def load_profiles(path: Path):
    profiles = []
    for file in sorted(path.glob("*.json")):
        try:
            data = json.loads(file.read_text())
        except Exception:
            continue
        data["_file"] = file.name
        profiles.append(data)
    return profiles


def fmt_num(v):
    if isinstance(v, float):
        return f"{v:.1f}"
    return str(v)


def main():
    if len(sys.argv) < 2:
        print("usage: perf_compare.py <dir>")
        return 1
    root = Path(sys.argv[1])
    profiles = load_profiles(root)
    if not profiles:
        print("no profiles found")
        return 1

    key_order = ["env", "backend", "steps_per_sec", "samples_per_sec", "env_seconds", "update_seconds", "render_seconds"]
    header = ["env", "backend", "sps", "samples", "env_s", "update_s", "render_s", "file"]
    rows = []
    for p in profiles:
        row = [
            p.get("env", "unknown"),
            p.get("backend", "unknown"),
            fmt_num(p.get("steps_per_sec", 0.0)),
            fmt_num(p.get("samples_per_sec", 0.0)),
            fmt_num(p.get("env_seconds", 0.0)),
            fmt_num(p.get("update_seconds", 0.0)),
            fmt_num(p.get("render_seconds", 0.0)),
            p.get("_file", ""),
        ]
        rows.append(row)

    col_widths = [max(len(str(r[i])) for r in ([header] + rows)) for i in range(len(header))]
    line = "  ".join(h.ljust(col_widths[i]) for i, h in enumerate(header))
    print(line)
    print("-" * len(line))
    for row in rows:
        print("  ".join(str(row[i]).ljust(col_widths[i]) for i in range(len(header))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
