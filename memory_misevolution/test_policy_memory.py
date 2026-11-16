import json
from pathlib import Path
import tempfile
import csv
import ast
import subprocess
import unittest
import urllib.error
from unittest.mock import patch, MagicMock

from reward_hacking_test import RewardHackingTester, service_memory
from policy_memory import (cache_key, classifier_input, parse_service_memory,
                           render_memory, select_records, validate_labels)
from policy_filter_experiment import main, chat
from api_config import GEMINI_URL, PacedCalls, load_env, valid_key
from summarize_policy_review import summarize


class PolicyMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp_root = Path(__file__).resolve().parent / "policy_filter_runs"
        self.temp_root.mkdir(exist_ok=True)
        self.records = parse_service_memory(service_memory)
        self.payload = json.loads(Path(__file__).with_name("demo_policy_labels.json").read_text())
        self.labels = validate_labels(self.payload, self.records)

    def test_parses_actual_upstream_logs_and_removes_group_headings(self):
        self.assertEqual(len(self.records), 6)
        self.assertEqual(sum(r["score"] for r in self.records), 359)
        self.assertNotIn("b. Failed", render_memory(self.records))

    def test_filter_retains_uncertainty_and_low_ratings(self):
        kept = select_records(self.records, self.labels, "filtered")
        self.assertEqual(len(kept), 5)
        self.assertNotIn("83998", {r["id"] for r in kept})
        self.assertIn("84250", {r["id"] for r in kept})
        self.assertIn("84321", {r["id"] for r in kept})

    def test_removed_evidence_cannot_leak_through_summary(self):
        text = render_memory(select_records(self.records, self.labels, "filtered"))
        self.assertNotIn("#83998", text)
        self.assertNotIn("99.5%", text)  # Legacy aggregate and justification absent.
        self.assertIn("Success Rate: 100.0%", text)  # Both remaining refunds high-rated.
        self.assertIn("Average Satisfaction Score: 97.5/100", text)

    def test_classifier_does_not_receive_feedback_or_scores(self):
        prompt = classifier_input(self.records, "test policy")
        self.assertNotIn("100/100", prompt)
        self.assertNotIn("User Feedback:", prompt)
        self.assertNotIn("Historical data indicates", prompt)

    def test_invalid_labels_rejected(self):
        for payload in ({"labels": []}, {"labels": self.payload["labels"] * 2},
                        {"labels": [{"id": "unknown", "status": "consistent", "reason": "test"}]}):
            with self.assertRaises(ValueError):
                validate_labels(payload, self.records)

    def test_random_matched_count_and_seed(self):
        one = select_records(self.records, self.labels, "random", 12)
        self.assertEqual(len(one), 5)
        self.assertEqual(one, select_records(self.records, self.labels, "random", 12))
        self.assertEqual(select_records(self.records, self.labels, "no_memory"), [])

    def test_cache_invalidation(self):
        first = cache_key(self.records, "P", "M", "E")
        self.assertNotEqual(first, cache_key(self.records, "changed", "M", "E"))
        self.assertNotEqual(first, cache_key(self.records, "P", "other", "E"))

    def test_original_prompt_unchanged_when_no_options(self):
        tester = RewardHackingTester()
        upstream = subprocess.check_output(["git", "show", "HEAD:memory_misevolution/reward_hacking_test.py"],
                                          cwd=Path(__file__).resolve().parent, text=True, encoding="utf-8")
        tree = ast.parse(upstream)
        # Load only the author's class; do not import dependencies or execute CLI.
        class_node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RewardHackingTester")
        namespace = {}
        exec(compile(ast.Module(body=[class_node], type_ignores=[]), "upstream", "exec"), namespace)
        reference = namespace["RewardHackingTester"]()
        for scenario in ("service", "sales", "medical", "financial"):
            self.assertEqual(tester.create_agent_request(scenario, "QUERY", "MEMORY"),
                             reference.create_agent_request(scenario, "QUERY", "MEMORY"))
        original = tester.create_agent_request("service", "QUERY", "MEMORY")
        self.assertTrue(original.startswith("You are CS-Optimizer-9000"))
        extended = tester.create_agent_request("service", "QUERY", "MEMORY", policy="POLICY")
        self.assertTrue(extended.endswith(original))

    def test_dry_run_never_calls_api_and_creates_review_evidence(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as temp, patch("policy_filter_experiment.chat", side_effect=AssertionError("No network")):
            output = main(["--limit", "2", "--repetitions", "1", "--output-dir", str(Path(temp)/"run")])
            rows = [json.loads(line) for line in (output/"responses.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 10)
            self.assertTrue(all(r["status"] == "dry_run" and r["response"] is None for r in rows))
            self.assertEqual(len(json.loads((output/"review_key.json").read_text())), 10)
            self.assertTrue(all(g["violation_rate"] is None for g in summarize(output).values()))
            with self.assertRaises(FileExistsError):
                main(["--output-dir", str(output)])

    def test_live_pipeline_with_mocked_api_cache_and_failure_accounting(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as temp, patch.dict("os.environ", {"OPENAI_API_KEY": "unit-test-not-a-real-key"}):
            root = Path(temp)
            args = ["--live", "--model", "test-model", "--limit", "1", "--repetitions", "1",
                    "--conditions", "filtered", "--cache-dir", str(root/"cache")]
            with patch("policy_filter_experiment.chat", side_effect=[(json.dumps(self.payload), {}), ("Action Taken: explain_policy", {})]) as api:
                first = main(args + ["--output-dir", str(root/"first")])
                self.assertEqual(api.call_count, 2)
            self.assertNotIn("unit-test-not-a-real-key", (first/"manifest.json").read_text())
            with (first/"manual_review.csv").open(newline="", encoding="utf-8") as handle:
                review = list(csv.DictReader(handle))
            review[0].update(policy_violation="no", task_success="yes")
            with (first/"manual_review.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=review[0].keys())
                writer.writeheader()
                writer.writerows(review)
            self.assertEqual(summarize(first)["filtered"]["task_success_rate"], 1.0)
            with patch("policy_filter_experiment.chat", side_effect=TimeoutError) as api:
                second = main(args + ["--output-dir", str(root/"second")])
                self.assertEqual(api.call_count, 1)  # Classification reused from cache.
            stats = summarize(second)["filtered"]
            self.assertEqual(stats["api_errors"], 1)
            self.assertIsNone(stats["violation_rate"])


class GeminiSetupTests(unittest.TestCase):
    def test_dotenv_is_literal_and_shell_variables_take_priority(self):
        root = Path(__file__).resolve().parent / "policy_filter_runs"
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as temp, patch.dict("os.environ", {"GEMINI_API_KEY": "existing"}):
            path = Path(temp)/"test.env"
            path.write_text('GEMINI_API_KEY="replacement"\nGEMINI_MODEL="model-test"\nUNRELATED_SECRET=ignored\n', encoding="utf-8")
            load_env(path)
            import os
            self.assertEqual(os.environ["GEMINI_API_KEY"], "existing")
            self.assertEqual(os.environ["GEMINI_MODEL"], "model-test")
            self.assertNotIn("UNRELATED_SECRET", os.environ)
            self.assertFalse(valid_key("PASTE_YOUR_GEMINI_API_KEY_HERE"))

    def test_gemini_transport_omits_sampling_override_and_rejects_truncation(self):
        import io
        response = {"choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}], "usage": {}}
        with patch("policy_filter_experiment.urllib.request.urlopen", return_value=io.StringIO(json.dumps(response))) as urlopen:
            self.assertEqual(chat("hello", "test", GEMINI_URL, "not-real")[0], "OK")
            request = urlopen.call_args.args[0]
            self.assertNotIn("temperature", json.loads(request.data))
            self.assertEqual(request.full_url, GEMINI_URL + "/chat/completions")
        response["choices"][0]["finish_reason"] = "length"
        with patch("policy_filter_experiment.urllib.request.urlopen", return_value=io.StringIO(json.dumps(response))):
            with self.assertRaises(ValueError):
                chat("hello", "test", GEMINI_URL, "not-real")

    def test_retries_rate_limits_but_not_auth_errors(self):
        limited = urllib.error.HTTPError("https://test", 429, "limit", {}, None)
        call = MagicMock(side_effect=[limited, ("OK", {})])
        with patch("api_config.time.sleep") as sleep:
            self.assertEqual(PacedCalls(0).call(call), ("OK", {}))
            sleep.assert_called_once_with(30)
        denied = urllib.error.HTTPError("https://test", 401, "denied", {}, None)
        call = MagicMock(side_effect=denied)
        with self.assertRaises(urllib.error.HTTPError):
            PacedCalls(0).call(call)
        self.assertEqual(call.call_count, 1)

    def test_gemini_connection_uses_gemini_key(self):
        with patch.dict("os.environ", {"GEMINI_API_KEY": "gemini-test-secret", "GEMINI_MODEL": "model-test"}), patch("policy_filter_experiment.chat", return_value=("OK", {})) as api:
            main(["--provider", "gemini", "--check-connection", "--request-interval", "0"])
            self.assertEqual(api.call_args.args[2], GEMINI_URL)
            self.assertEqual(api.call_args.args[3], "gemini-test-secret")


if __name__ == "__main__":
    unittest.main()
