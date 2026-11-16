"""Local credentials and paced requests, without extra dependencies."""
import os
from pathlib import Path
import time
import urllib.error

ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai"


def load_env(path=ENV_FILE):
    """Read whitelisted literal KEY=VALUE entries, without executing shell code.

    Existing shell variables take priority. Never print credential values.
    """
    if not path.exists():
        return
    allowed = {"GEMINI_API_KEY", "GEMINI_MODEL", "OPENAI_API_KEY", "OPENAI_BASE_URL"}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, value = line.partition("=")
        name, value = name.strip(), value.strip()
        if separator and name in allowed:
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            os.environ.setdefault(name, value)


def valid_key(value):
    return bool(value.strip()) and value.strip() not in {"PASTE_YOUR_GEMINI_API_KEY_HERE", "<your-api-key>"}


class PacedCalls:
    """Space calls and retry only explicit 429/503 responses, at most twice.

    Do not retry timeouts: the provider may already have processed that request.
    """
    def __init__(self, interval=15, retries=2):
        self.interval = interval
        self.retries = retries
        self.last_start = None

    def call(self, function, *args):
        for attempt in range(self.retries + 1):
            if self.last_start is not None:
                delay = self.interval - (time.monotonic() - self.last_start)
                if delay > 0:
                    time.sleep(delay)
            self.last_start = time.monotonic()
            try:
                return function(*args)
            except urllib.error.HTTPError as error:
                if error.code not in (429, 503) or attempt == self.retries:
                    raise
                delay = min(60, 30 * (attempt + 1))
                print(f"HTTP {error.code}: waiting {delay}s before retry {attempt + 1}/{self.retries}.", flush=True)
                time.sleep(delay)


def describe_error(error):
    if isinstance(error, urllib.error.HTTPError):
        messages = {400: "Unsupported model settings or malformed request.",
                    401: "API key was rejected.", 403: "Key or project lacks access.",
                    404: "Model unavailable; check GEMINI_MODEL in .env.",
                    429: "Quota/rate limit reached. Check AI Studio; do not enable billing to bypass it.",
                    503: "Provider temporarily unavailable."}
        return f"HTTP {error.code}: {messages.get(error.code, 'Provider request failed.')}"
    return f"{type(error).__name__}: request or response processing failed."
