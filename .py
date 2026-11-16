"""
Task 2: Model / AI Integration
Lightweight text-generation and summarization utility built on the
Google GenAI SDK (gemini-2.5-flash).

Usage:
    python gemini_utility.py demo
    python gemini_utility.py summarize --text "Long text here..."
    python gemini_utility.py summarize --file article.txt
    python gemini_utility.py generate --prompt "Write a haiku about IoT"
    python gemini_utility.py solve --problem "A rover travels 12 m/min for 45 min. Distance in km?"
"""

import argparse
import os
import sys
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

MODEL_NAME = "gemini-2.5-flash"
MAX_RETRIES = 3
RETRYABLE_CODES = {429, 500, 503, 504}


# --------------------------------------------------------------------------
# Client setup (API key comes from .env, never hard-coded)
# --------------------------------------------------------------------------
def get_client() -> genai.Client:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        sys.exit(
            "Error: GEMINI_API_KEY is missing.\n"
            "Copy .env.example to .env and add your key."
        )
    return genai.Client(api_key=api_key)


# --------------------------------------------------------------------------
# Core call with error handling + exponential backoff
# --------------------------------------------------------------------------
def ask_gemini(
    client: genai.Client,
    prompt: str,
    system_instruction: str | None = None,
    temperature: float = 0.4,
    max_output_tokens: int = 1024,
) -> str:
    if not prompt or not prompt.strip():
        raise ValueError("Prompt must not be empty.")

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME, contents=prompt, config=config
            )
            if not response.text:
                return "[No text returned - the response may have been blocked by safety filters.]"
            return response.text.strip()

        except errors.APIError as e:
            if e.code in RETRYABLE_CODES and attempt < MAX_RETRIES:
                wait = 2 ** attempt
                print(f"[retry {attempt}/{MAX_RETRIES}] API error {e.code}; waiting {wait}s...")
                time.sleep(wait)
                continue
            if e.code in (401, 403):
                raise RuntimeError("Authentication failed. Check your GEMINI_API_KEY.") from e
            raise RuntimeError(f"Gemini API error {e.code}: {e.message}") from e

        except Exception as e:  # network errors, etc.
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"Unexpected error: {e}") from e

    raise RuntimeError("Request failed after all retries.")


# --------------------------------------------------------------------------
# Task functions
# --------------------------------------------------------------------------
def summarize(client: genai.Client, text: str, sentences: int = 3) -> str:
    return ask_gemini(
        client,
        prompt=f"Summarize the following text in at most {sentences} sentences:\n\n{text}",
        system_instruction="You are a precise summarizer. Keep key facts, drop filler. Output only the summary.",
        temperature=0.2,
    )


def generate(client: genai.Client, prompt: str) -> str:
    return ask_gemini(
        client,
        prompt=prompt,
        system_instruction="You are a helpful, concise assistant.",
        temperature=0.8,
    )


def solve(client: genai.Client, problem: str) -> str:
    return ask_gemini(
        client,
        prompt=problem,
        system_instruction=(
            "Solve the problem step by step, keep steps short, "
            "and end with a line starting 'Answer:'."
        ),
        temperature=0.1,
    )


# --------------------------------------------------------------------------
# Demo with clear example inputs / outputs
# --------------------------------------------------------------------------
def run_demo(client: genai.Client) -> None:
    article = (
        "The Internet of Things connects everyday physical devices such as sensors, "
        "motors and cameras to the internet so they can collect and exchange data. "
        "In agriculture, IoT sensors monitor soil moisture and temperature, letting "
        "farmers irrigate only when needed. In environmental work, floating robots "
        "equipped with sensors can map and collect pollution in lakes and oceans. "
        "Because these devices often run on small batteries, efficient firmware and "
        "low-power communication are critical design considerations."
    )
    examples = [
        ("SUMMARIZATION", "Input: IoT paragraph (about 70 words)", lambda: summarize(client, article, 2)),
        ("TEXT GENERATION", "Input: 'Write a two-line slogan for a beach-cleaning robot.'",
         lambda: generate(client, "Write a two-line slogan for a beach-cleaning robot.")),
        ("PROBLEM SOLVING", "Input: 'A rover moves at 12 m/min for 45 min. How many km?'",
         lambda: solve(client, "A rover moves at 12 m/min for 45 minutes. How many kilometres does it travel?")),
    ]
    for title, inp, fn in examples:
        print(f"\n{'=' * 60}\n{title}\n{inp}\n{'-' * 60}")
        try:
            print("Output:", fn())
        except Exception as e:
            print(f"Failed: {e}")


def run_examples(client: genai.Client, inp="examples/inputs.json", out="examples/outputs.md") -> None:
    """Run every example in inputs.json and save input/output pairs to outputs.md."""
    import json

    with open(inp, "r", encoding="utf-8") as f:
        cases = json.load(f)
    handlers = {"summarize": summarize, "generate": generate, "solve": solve}

    lines = ["# Example Inputs and Outputs", f"\nModel: `{MODEL_NAME}`\n"]
    for i, case in enumerate(cases, 1):
        task, text = case["task"], case["input"]
        print(f"[{i}/{len(cases)}] {task}...")
        try:
            result = handlers[task](client, text)
        except Exception as e:
            result = f"(failed: {e})"
        lines += [f"## Example {i} - {task}", "**Input:**", f"> {text}", "", "**Output:**", result, ""]
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved to {out}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Gemini text utility")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("demo", help="Run built-in example inputs")
    sub.add_parser("examples", help="Run examples/inputs.json and save examples/outputs.md")

    p_sum = sub.add_parser("summarize", help="Summarize text")
    p_sum.add_argument("--text")
    p_sum.add_argument("--file")
    p_sum.add_argument("--sentences", type=int, default=3)

    p_gen = sub.add_parser("generate", help="Generate text from a prompt")
    p_gen.add_argument("--prompt", required=True)

    p_sol = sub.add_parser("solve", help="Solve a problem step by step")
    p_sol.add_argument("--problem", required=True)

    args = parser.parse_args()
    client = get_client()

    try:
        if args.command == "demo":
            run_demo(client)
        elif args.command == "examples":
            run_examples(client)
        elif args.command == "summarize":
            if args.file:
                with open(args.file, "r", encoding="utf-8") as f:
                    text = f.read()
            elif args.text:
                text = args.text
            else:
                sys.exit("Provide --text or --file.")
            print(summarize(client, text, args.sentences))
        elif args.command == "generate":
            print(generate(client, args.prompt))
        elif args.command == "solve":
            print(solve(client, args.problem))
    except (ValueError, RuntimeError, OSError) as e:
        sys.exit(f"Error: {e}")


if __name__ == "__main__":
    main()
