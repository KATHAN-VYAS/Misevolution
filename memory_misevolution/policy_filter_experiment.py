"""Opt-in extension of reward_hacking_test.py; stdlib-only, offline by default."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import urllib.request
from api_config import GEMINI_URL, PacedCalls, describe_error, load_env, valid_key

from reward_hacking_test import RewardHackingTester, service_memory, service_user_requests
from policy_memory import (WARNING, cache_key, classifier_input, parse_service_memory,
                           render_memory, select_records, validate_labels)

HERE = Path(__file__).resolve().parent
CONDITIONS = ("original", "warning", "filtered", "random", "no_memory")


def chat(prompt, model, base_url, key, max_tokens=2048):
    """OpenAI-compatible endpoint; TLS verification on."""
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens}
    # Some newer Gemini models do not accept sampling overrides. Preserve their
    # provider defaults and record this explicitly in the experiment manifest.
    if not base_url.startswith(GEMINI_URL):
        body["temperature"] = 0.1
    data = json.dumps(body).encode()
    request = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data,
                                     {"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.load(response)
    choice = payload["choices"][0]
    if choice.get("finish_reason") == "length":
        raise ValueError("Truncated model response; increase --max-tokens")
    content = choice["message"]["content"]
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Missing text in API response")
    return content, payload.get("usage", {})


def json_object(text):
    text = text.strip()
    if text.startswith("```json") and text.endswith("```"):
        text = text[7:-3].strip()
    elif text.startswith("```") and text.endswith("```"):
        text = text[3:-3].strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object")
    return value


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def revision():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main(argv=None):
    load_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Enable paid model calls; default is dry run")
    parser.add_argument("--provider", choices=("openai", "gemini"), default="openai")
    parser.add_argument("--model")
    parser.add_argument("--check-connection", action="store_true", help="One small live request, not an experiment")
    parser.add_argument("--filter-model", help="Defaults to the agent model")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--policy", type=Path, default=HERE / "service_policy.txt")
    parser.add_argument("--labels-file", type=Path, help="Explicit manual-label ablation; never used implicitly in live runs")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--limit", type=int, default=10, help="Number of original service requests")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=818)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--request-interval", type=float, help="Seconds between request starts; Gemini default 15")
    parser.add_argument("--cache-dir", type=Path, default=HERE / "policy_filter_runs" / "cache")
    parser.add_argument("--output-dir", type=Path, help="Must be new, to avoid overwriting evidence")
    args = parser.parse_args(argv)
    if args.provider == "gemini":
        args.base_url = GEMINI_URL
        args.model = args.model or os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
        key = os.environ.get("GEMINI_API_KEY", "")
    else:
        args.model = args.model or "dry-run"
        key = os.environ.get("OPENAI_API_KEY", "")
    if args.max_tokens is None:
        args.max_tokens = 8192 if args.provider == "gemini" else 2048
    interval = args.request_interval if args.request_interval is not None else (15 if args.provider == "gemini" else 0)
    if interval < 0:
        parser.error("Request interval cannot be negative")
    paced = PacedCalls(interval, retries=2 if args.provider == "gemini" else 0)
    def request_model(prompt, model, budget=None):
        return paced.call(chat, prompt, model, args.base_url, key, budget or args.max_tokens)
    if not 1 <= args.limit <= len(service_user_requests) or args.repetitions < 1 or args.max_tokens < 1:
        parser.error("Invalid limit, repetitions or max-tokens")
    if len(set(args.conditions)) != len(args.conditions):
        parser.error("Conditions must not repeat")
    if (args.live or args.check_connection) and (not valid_key(key) or args.model == "dry-run"):
        parser.error("Set GEMINI_API_KEY in Misevolution/.env for Gemini; other providers need OPENAI_API_KEY and --model")
    if args.check_connection:
        try:
            response, _ = request_model("Reply with exactly OK.", args.model)
        except Exception as error:
            parser.exit(1, describe_error(error) + "\n")
        print(f"Connection successful: provider={args.provider}, model={args.model}; text received={bool(response)}")
        return
    output = args.output_dir or HERE / "policy_filter_runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    policy = args.policy.read_text(encoding="utf-8").strip()
    if not policy:
        parser.error("Policy must not be empty")
    records = parse_service_memory(service_memory)
    filter_model = args.filter_model or args.model
    filter_prompt = classifier_input(records, policy)
    write_json(output / "filter_input.json", {"prompt": filter_prompt})
    needs_filter = bool(set(args.conditions) & {"filtered", "random"})
    usage = {}
    if args.labels_file:
        payload = json.loads(args.labels_file.read_text(encoding="utf-8"))
        label_source = "manual:" + str(args.labels_file.resolve())
    elif not args.live:
        payload = json.loads((HERE / "demo_policy_labels.json").read_text(encoding="utf-8"))
        label_source = "manual_demo_fixture_NOT_MODEL_RESULTS"
    elif needs_filter:
        identity = cache_key(records, policy, filter_model, args.base_url + f"|budget={args.max_tokens}|gemini_defaults={args.provider == 'gemini'}")
        cache = args.cache_dir / f"{identity}.json"
        if cache.exists():
            cached = json.loads(cache.read_text(encoding="utf-8"))
            if cached.get("cache_key") != identity:
                raise ValueError("Cache identity mismatch")
            payload = cached["payload"]
            label_source = "model_cache:" + identity
        else:
            try:
                raw, usage = request_model(filter_prompt, filter_model)
            except Exception as error:
                write_json(output / "filter_error.json", {"error": describe_error(error)})
                parser.exit(1, describe_error(error) + "\n")
            write_json(output / "filter_response.json", {"raw": raw, "usage": usage})
            payload = json_object(raw)
            validate_labels(payload, records)  # Invalid output never becomes cache.
            args.cache_dir.mkdir(parents=True, exist_ok=True)
            write_json(cache, {"cache_key": identity, "payload": payload})
            label_source = "model:" + identity
    else:
        payload = {"labels": [{"id": r["id"], "status": "uncertain", "reason": "Filter not requested"} for r in records]}
        label_source = "not_used"
    labels = validate_labels(payload, records)
    write_json(output / "memory_audit.json", {"label_source": label_source, "labels": payload["labels"],
               "source_records": records, "excluded_ids": [r["id"] for r in records if labels[r["id"]]["status"] == "violating"]})
    tester = RewardHackingTester()
    jobs = []
    for repeat in range(args.repetitions):
        for condition in args.conditions:
            selected = select_records(records, labels, condition, args.seed + repeat)
            memory = render_memory(selected)
            for case, query in enumerate(service_user_requests[:args.limit]):
                prompt = tester.create_agent_request("service", query, memory, policy=policy,
                            memory_warning=WARNING if condition == "warning" else None)
                jobs.append({"id": f"r{repeat}-q{case}-{condition}", "condition": condition,
                             "repetition": repeat, "case": case, "user_request": query,
                             "kept_ids": [r["id"] for r in selected], "full_prompt": prompt})
    random.Random(args.seed).shuffle(jobs)
    write_json(output / "manifest.json", {"mode": "live" if args.live else "dry_run_NO_MODEL_RESULTS",
               "revision": revision(), "model": args.model, "filter_model": filter_model,
               "base_url": args.base_url, "policy": policy, "label_source": label_source,
               "conditions": args.conditions, "seed": args.seed, "repetitions": args.repetitions,
               "max_tokens": args.max_tokens, "temperature": "provider_default" if args.provider == "gemini" else 0.1,
               "provider": args.provider, "request_interval": interval, "planned_responses": len(jobs),
               "filter_usage_this_run": usage,
               "upstream_memory_sha256": hashlib.sha256(service_memory.encode()).hexdigest(),
               "note": "All arms use an added policy and normalized small-sample summaries. Not exact paper reproduction."})
    (output / "upstream_memory.txt").write_text(service_memory, encoding="utf-8")
    (output / "normalized_memory.txt").write_text(render_memory(records), encoding="utf-8")
    with (output / "responses.jsonl").open("w", encoding="utf-8") as handle:
        for job in jobs:
            if args.live:
                try:
                    print(f"Generating {job['id']}...", flush=True)
                    response, token_usage = request_model(job["full_prompt"], args.model)
                    job.update(response=response, usage=token_usage, status="ok")
                except Exception as exc:
                    # Save type only; provider errors can contain sensitive request details.
                    job.update(response=None, error_type=type(exc).__name__, error=describe_error(exc), status="api_error")
            else:
                job.update(response=None, status="dry_run")
            handle.write(json.dumps(job, ensure_ascii=False) + "\n")
            handle.flush()
    # Opaque IDs and shuffled order support manual labeling without condition names.
    with (output / "manual_review.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["review_id", "user_request", "response", "status", "policy_violation", "task_success", "notes"])
        writer.writeheader()
        for i, job in enumerate(jobs):
            writer.writerow({"review_id": f"review-{i:04}", "user_request": job["user_request"],
                             "response": job["response"], "status": job["status"]})
    write_json(output / "review_key.json", {f"review-{i:04}": job["id"] for i, job in enumerate(jobs)})
    print(f"{'Live run' if args.live else 'DRY RUN (no API calls, no findings)'}: {len(jobs)} records in {output}")
    print("Manual evaluation required; API completion is not task success.")
    return output


if __name__ == "__main__":
    main()
