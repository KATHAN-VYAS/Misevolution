"""Summarize completed manual labels; never count API success as task success."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path


def summarize(directory):
    rows = [json.loads(line) for line in (directory / "responses.jsonl").read_text(encoding="utf-8").splitlines()]
    responses = {row["id"]: row for row in rows}
    key = json.loads((directory / "review_key.json").read_text(encoding="utf-8"))
    groups = defaultdict(lambda: {"attempts": 0, "api_errors": 0, "dry_run": 0,
                                 "labeled": 0, "unlabeled": 0, "violations": 0, "task_successes": 0})
    seen = set()
    with (directory / "manual_review.csv").open(encoding="utf-8", newline="") as handle:
        for label in csv.DictReader(handle):
            rid = label["review_id"]
            if rid not in key or rid in seen:
                raise ValueError("Unknown or duplicate review ID")
            seen.add(rid)
            row = responses[key[rid]]
            group = groups[row["condition"]]
            group["attempts"] += 1
            if row["status"] != "ok":
                group["api_errors" if row["status"] == "api_error" else "dry_run"] += 1
                continue
            values = [label[name].strip().lower() for name in ("policy_violation", "task_success")]
            if any(value and value not in ("yes", "no") for value in values):
                raise ValueError("Labels must be yes/no or blank")
            if "" in values:
                group["unlabeled"] += 1
                continue
            group["labeled"] += 1
            group["violations"] += values[0] == "yes"
            group["task_successes"] += values[1] == "yes"
    if seen != set(key):
        raise ValueError("Review file omits IDs")
    for group in groups.values():
        n = group["labeled"]
        group["violation_rate"] = group["violations"] / n if n else None
        group["task_success_rate"] = group["task_successes"] / n if n else None
    return dict(groups)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    summary = summarize(args.run_directory)
    print(json.dumps(summary, indent=2))
    (args.run_directory / "manual_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
