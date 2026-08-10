from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

MODEL = "qwen2.5:1.5b-instruct"
API_URL = "http://localhost:11434/api/chat"
BASE_DIR = Path(__file__).resolve().parent
PROMPTS_FILE = BASE_DIR / "prompts.jsonl"
OUTPUTS_FILE = BASE_DIR / "outputs.jsonl"

TEMPERATURE = 0.7
SYSTEM_PROMPT = ""  # Keep empty for raw testing. Set one only if you want it.


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def load_prompts(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    prompts: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        prompt_id = str(obj.get("id", "")).strip()
        text = str(obj.get("text", "")).strip()
        if prompt_id and text:
            prompts[prompt_id] = text

    return prompts


def save_output(record: dict) -> None:
    OUTPUTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUTS_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def call_model(prompt: str) -> tuple[str, dict]:
    messages = []
    if SYSTEM_PROMPT.strip():
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
        },
    }

    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    if "error" in data:
        raise RuntimeError(str(data["error"]))

    message = data.get("message", {})
    content = message.get("content", "")
    return content, data


def print_help(prompts: dict[str, str]) -> None:
    print("\nCommands:")
    print("  :q              quit")
    print("  :list           list preset prompts")
    print("  :p <id>         run a preset by id")
    print("  any other text  send a freeform prompt")
    if prompts:
        print(f"\nLoaded {len(prompts)} preset prompt(s).")
    else:
        print("\nNo prompts.jsonl found yet. Freeform input still works.")


def list_prompts(prompts: dict[str, str]) -> None:
    if not prompts:
        print("No presets loaded.")
        return

    print("\nPreset prompts:")
    for prompt_id, text in prompts.items():
        preview = text if len(text) <= 80 else text[:77] + "..."
        print(f"  {prompt_id}: {preview}")


def main() -> None:
    prompts = load_prompts(PROMPTS_FILE)
    print(f"Using model: {MODEL}")
    print_help(prompts)

    while True:
        try:
            user_in = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_in:
            continue

        if user_in in {":q", "quit", "exit"}:
            break

        if user_in == ":list":
            list_prompts(prompts)
            continue

        mode = "freeform"
        prompt_id = None
        prompt_text = user_in

        if user_in.startswith(":p "):
            prompt_id = user_in[3:].strip()
            if not prompt_id:
                print("Usage: :p <id>")
                continue
            if prompt_id not in prompts:
                print(f"Unknown preset id: {prompt_id}")
                continue
            prompt_text = prompts[prompt_id]
            mode = "preset"

        start = time.perf_counter()
        try:
            response_text, raw = call_model(prompt_text)
            elapsed = round(time.perf_counter() - start, 3)

            print(f"\nmodel> {response_text}")

            record = {
                "ts": now_iso(),
                "mode": mode,
                "prompt_id": prompt_id,
                "prompt": prompt_text,
                "response": response_text,
                "model": MODEL,
                "temperature": TEMPERATURE,
                "latency_s": elapsed,
                "status": "ok",
            }
            save_output(record)

        except Exception as e:
            elapsed = round(time.perf_counter() - start, 3)
            print(f"\nerror> {e}")

            record = {
                "ts": now_iso(),
                "mode": mode,
                "prompt_id": prompt_id,
                "prompt": prompt_text,
                "response": None,
                "model": MODEL,
                "temperature": TEMPERATURE,
                "latency_s": elapsed,
                "status": "error",
                "error": str(e),
            }
            save_output(record)


if __name__ == "__main__":
    main()