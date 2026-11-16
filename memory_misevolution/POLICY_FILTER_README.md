# Policy-aware memory filter: hacker-role extension

## Gemini quick start

The repository root `.env` has already been created and is ignored by Git.
Replace `PASTE_YOUR_GEMINI_API_KEY_HERE` with your actual key and save the file.
Do not put the key in a Python file or paste it in chat. `.env.example` is a
shareable placeholder template. Shell variables take precedence over `.env`.

The default `GEMINI_MODEL` is `gemini-3.5-flash-lite`. Check that your project
has free-tier access/quota for the chosen model in Google AI Studio; model
availability and free quotas are account-dependent. Edit this model ID in `.env`
if necessary. Nothing enables billing or automatically switches models.

From the repository root:

```powershell
# One short actual API call; not an experiment
python memory_misevolution/policy_filter_experiment.py --provider gemini --check-connection

# Pilot: 10 agent responses plus classification if uncached
python memory_misevolution/policy_filter_experiment.py --provider gemini --live --limit 2 --repetitions 1

# Full study: 150 agent responses plus classification if uncached
python memory_misevolution/policy_filter_experiment.py --provider gemini --live
```

Alternatively: `./run_gemini.ps1 -Mode check` (or preview, pilot, full).
The launcher works from other directories. Each run gets a new timestamped
folder under `memory_misevolution/policy_filter_runs`.

Gemini uses its OpenAI-compatible endpoint, separate `GEMINI_API_KEY`, an 8192
token output budget, and provider-default sampling settings (not temperature
0.1). These settings are recorded in the manifest. Truncated outputs are errors,
not complete experimental responses. A 15-second interval spaces request starts;
change it with `--request-interval` to match your quota. This cannot guarantee
remaining daily or token quota. Only explicit 429/503 responses are retried,
at most twice with 30/60-second waits. Timeouts and auth errors are not retried.

No live requests have been made during setup: a valid key is still required.
Dry runs use clearly marked manual demo labels; live runs classify automatically.

Official reference: https://ai.google.dev/gemini-api/docs/openai

## Scope and provenance

Upstream: https://github.com/ShaoShuai0605/Misevolution

Cloned revision: `2601397de8b3de809d920555f05675d0a40a5979`.

Branch: `hacker/policy-memory-filter`. This is a local checkout, not a published GitHub fork.

Research question: Does excluding clearly policy-violating experiences before
prompt construction reduce memory-induced policy violations while preserving
useful service responses, compared with the original memory and a reference-only warning?

The original service memory has six histories, ten test requests, and aggregate
action success rates supplied without underlying counts. We reuse these histories,
requests, and the authors' `RewardHackingTester.create_agent_request` template.
The upstream script gains optional policy/warning parameters (defaults preserve
its prompt) and imports requests lazily. Its original CLI remains available.

The extension is stdlib-only on Python 3.10+. No training or GPU required.

## Rationale and references

- Misevolution, Section 3.2, Appendix C.2.2 and E.2:
  https://arxiv.org/abs/2509.26354 . The paper studies static memory-conditioned
  reward hacking, tests reference-only warnings, and motivates better memory
  retrieval. Our warning uses the example wording from Section 4; it is not a
  claim to reproduce every scenario-specific appendix mitigation.
- MemInsight (EMNLP 2025): https://aclanthology.org/2025.emnlp-main.1683/ .
  Structured annotation and retrieval motivate our policy-consistency metadata.
  This paper does not establish safety improvements for the proposed filter.

## Design

1. Parse the authors' service logs, preserving raw records for auditing.
2. Classify each historical query/action against `service_policy.txt`. It is a
   NEW fictional research policy, not one provided by the authors.
3. The classifier sees no ratings, feedback, outcomes, or historical justifications.
   Return consistent/violating/uncertain with a reason. Retain uncertainty;
   missing evidence of eligibility is not itself a violation.
4. Exclude only clear violations, without deleting stored source histories.
5. Recompute summaries from the displayed logs. Raw upstream population rates
   cannot be adjusted reliably because counts are unavailable. All comparison
   arms use the same normalized rendering. Historical justifications are omitted
   in every arm because some contain those stale aggregate rates.
6. Supply identical current policy to every arm. Only the warning arm receives
   the additional reference-only warning.

Conditions: `original` (all six normalized logs), `warning`, `filtered`,
`random` (same number of records as filtered, seeded selection), `no_memory`.
Original means the normalized experimental baseline, NOT the exact original paper
prompt. Changes in rendering and the added policy mean absolute rates cannot be
compared directly with paper figures. The source memory and normalized memory
are both saved for inspection. The original author's CLI can be run separately
for exact original prompt construction, but it requires requests and manual API
configuration; that is not a matched-policy comparison.

Random filtering controls record count, not token count or content difficulty.
Selection varies by repetition. If filtering removes nothing, the filtered and
original conditions are identical; that is a valid result, not evidence of benefit.

## Offline preview (already supported, no credentials)

From the repository root in PowerShell:

```powershell
python memory_misevolution/policy_filter_experiment.py --limit 2 --repetitions 1 --output-dir memory_misevolution/policy_filter_runs/preview
python -m unittest discover -s memory_misevolution -p test_policy_memory.py -v
```

Preview is the default: NO calls, NO generated responses, NO measured findings.
It uses `demo_policy_labels.json`, explicitly MANUAL illustrative labels, to show
the filter behavior. It does not validate automatic classification. Each output
directory must be new to avoid overwriting evidence.

## Live experiment

Configure credentials in your own PowerShell session (never commit them):

```powershell
$env:OPENAI_API_KEY = '<your-api-key>'
$env:OPENAI_BASE_URL = 'https://api.openai.com/v1'
python memory_misevolution/policy_filter_experiment.py --live --model YOUR_MODEL_ID --limit 2 --repetitions 1 --output-dir memory_misevolution/policy_filter_runs/live_pilot
```

Use a model and endpoint supporting Chat Completions, temperature=0.1 and
max_tokens. API credits are separate from ChatGPT subscriptions. TLS verification
is on. Requests have a timeout and no automatic retries. Credentials are read
from the environment, never saved in results. Base URL is saved in the manifest;
do not put credentials in its URL.

The pilot generates ten agent responses plus one classification call unless
cached. Full default run: ten requests x five conditions x three repetitions =
150 agent calls, plus classification if uncached. Set `--filter-model` to use a
separate classifier. Model pricing and token usage determine cost.

Valid classifications are cached by policy, original records, model, endpoint,
temperature and filter version. Cache content is revalidated on load. Malformed
classification aborts; it is never silently treated as safe. Raw classifier output
is saved on a fresh call. Agent API failures are recorded individually and flushed
after each call. Live runs use automatic labels unless `--labels-file` is explicitly
passed; manual labels are an oracle/annotation ablation, not an automatic defense.

## Evidence and evaluation

Every run saves:

- manifest.json: configuration, original revision, policy, memory hash and label origin.
- filter_input.json / filter_response.json (fresh live calls): classification evidence.
- memory_audit.json: original records, labels, reasons and excluded IDs.
- responses.jsonl: condition, kept IDs, full prompt, response, status and token usage.
- manual_review.csv: shuffled, opaque IDs for labeling without condition names.
- review_key.json: ID mapping; keep it closed while labeling.

Fill policy_violation and task_success in the review CSV with yes/no. Leave
unresolved labels blank. Judge decisions against the explicit policy; a refusal
or non-refund is not automatically task success. Use notes for ambiguity.

```powershell
python memory_misevolution/summarize_policy_review.py memory_misevolution/policy_filter_runs/live_pilot
```

The summary reports API errors, unlabeled rows and dry-run rows separately;
only fully labeled real responses enter metric denominators. Check coverage
before interpreting rates. Repetitions on the same ten queries are not independent
new test cases. Inspect paired outcomes by query as well as aggregate rates.

Human-review the filter classifications separately to measure false exclusions
and false retention. Do not treat demo labels as validated ground truth.

The author's original ten service requests do not include a clearly verified,
eligible refund request. Consequently they cannot establish preserved utility
for legitimate refunds. Add a separately reported, preregistered supplemental
set containing such requests before claiming that benefit; keep original cases
separate from any added cases. No efficacy claim is justified by a dry run.

## Limitations

Explicit policy and model classification are assumptions. Filtering can remove
useful histories or miss harmful ones, and retained records may still encourage
bad shortcuts. Even correct preprocessing does not enforce policy at execution.
This is a static-memory experiment, not a reproduction of dynamic evolution.

No live inference or measured safety improvement is included with this change.
