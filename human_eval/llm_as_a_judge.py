import argparse
import json
import csv
import math
import re
from pathlib import Path

import requests
import ollama

import os
OLLAMA_BASE = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")

# Create a client instance with your host (do this once, at module level)
client = ollama.Client(
    host=OLLAMA_BASE,
    timeout=300
)

INPUT_CSV = "sampled_questions_eval.csv"
OUTPUT_CSV = "llm_judge_results.csv"
LOGPROB_CSV = "llm_judge_logprobs.csv"

MODELS = [
    "gemma3:12b",
    "qwen2.5:14b",
    "mistral-small3.2",
    "aya-expanse:32b"
]

CRITERIA = [
    "grammaticality",
    "coherence",
    "originality",
    "creativity",
    "complexity",
    "likeability",
    "humanlikeness",
]

# Per-criterion descriptions used for the single-digit logprobs prompts
CRITERIA_DESCRIPTIONS = {
    "grammaticality": (
        "Grammaticality: Is the story grammatically correct? Be lenient — a single small mistake "
        "still counts as correct. Note that some constructions work in spoken Dutch but not in written Dutch."
    ),
    "coherence": (
        "Coherence: Does the story follow a clear narrative thread? "
        "Do characters appear or disappear abruptly? Does the setting change unnaturally?"
    ),
    "originality": "Originality: How original is this story? Give your honest assessment.",
    "creativity": "Creativity: How creative is this story? Give your honest assessment.",
    "complexity": (
        "Complexity: How complex is the story structure? Are the characters sufficiently developed?"
    ),
    "likeability": "Likeability: How enjoyable is this story to read?",
    "humanlikeness": (
        "Humanlikeness: How likely does it seem that this story was told by a real child?"
    ),
}

# ── Prompts ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert evaluator of Dutch children's stories. Your task is to rate a short Dutch story on seven quality criteria, each on a scale from 1 (very poor) to 5 (excellent).

The stories are a mix of real stories told by children and AI-generated stories designed to mimic child-told stories. Rate each story objectively based on the criteria below.

Criteria:
1. grammaticality — Is the story grammatically correct? Be lenient: a single small mistake still counts as correct. Note that some constructions work in spoken Dutch but not in written Dutch.
2. coherence — Does the story follow a clear narrative thread? Do characters appear or disappear abruptly? Does the setting change unnaturally?
3. originality — How original is this story? There is no right or wrong answer; give your honest assessment.
4. creativity — How creative is this story in your view? There is no right or wrong answer; give your honest assessment.
5. complexity — How complex is the story structure? Are the characters sufficiently developed?
6. likeability — How enjoyable is this story to read?
7. humanlikeness — How likely does it seem that this story was told by a real child?

Return ONLY a JSON object with exactly these keys: grammaticality, coherence, originality, creativity, complexity, likeability, humanlikeness. Each value must be an integer between 1 and 5. Do not include any other text.

Example:
{"grammaticality": 4, "coherence": 3, "originality": 2, "creativity": 3, "complexity": 2, "likeability": 4, "humanlikeness": 3}"""

LOGPROB_SYSTEM_PROMPT = (
    "You are an expert evaluator of Dutch children's stories. "
    "You will rate a story on a single criterion on a scale from 1 (very poor) to 5 (excellent). "
    "Respond with ONLY a single digit: 1, 2, 3, 4, or 5. No explanation, no other text."
)


# ── JSON evaluation ────────────────────────────────────────────────────────────

def extract_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text).strip()
    match = re.search(r"\{[^{}]+\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in response: {text!r}")
    ratings = json.loads(match.group())
    return {k: int(ratings[k]) for k in CRITERIA}


def rate_story(model: str, story: str) -> dict:
    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Rate the following Dutch children's story:\n\n{story}"},
        ],
        options={"temperature": 0.0, "top_p": 1.0, "top_k": -1},
    )
    return extract_json(response["message"]["content"])


# ── Log prob evaluation ────────────────────────────────────────────────────────

def rate_story_logprobs(model: str, story: str, criterion: str) -> dict:
    """
    Rate a single criterion by inspecting the log probs of the first generated token.
    Returns greedy_score, expected_score, entropy, and p1–p5.
    """
    user_msg = (
        f"Rate the following Dutch children's story on this criterion:\n\n"
        f"{CRITERIA_DESCRIPTIONS[criterion]}\n\n"
        f"Story:\n{story}\n\n"
        f"Respond with only a single digit (1–5):"
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": LOGPROB_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "stream": False,
        "options": {"temperature": 0.0, "top_p": 1.0, "top_k": -1},
        "logprobs": True,
        "top_logprobs": 5
    }

    resp = requests.post(f"{OLLAMA_BASE}/api/chat", json=payload, timeout=300)
    resp.raise_for_status()
    data = resp.json()

    greedy_token = data["message"]["content"].strip()[:1]  # first character only

    # Extract per-digit log probs from the first generated token.
    # Ollama returns logprobs as a list; each entry may contain top_logprobs alternatives.
    digit_logprobs: dict[str, float] = {}
    raw_logprobs = data.get("logprobs") or []

    if raw_logprobs:
        first = raw_logprobs[0]
        # Handle both flat {"token":..,"logprob":..,"top_logprobs":[..]} and
        # OpenAI-style {"content":[{"token":..,"top_logprobs":[..]}]}
        candidates: list[dict] = (
            first.get("top_logprobs")
            or (first.get("content") or [{}])[0].get("top_logprobs")
            or []
        )
        for entry in candidates:
            t = entry.get("token", "").strip()
            if t in {"1", "2", "3", "4", "5"}:
                digit_logprobs[t] = entry["logprob"]

    # Fall back: if no top_logprobs, treat the greedy token as certain
    if not digit_logprobs and greedy_token in {"1", "2", "3", "4", "5"}:
        digit_logprobs = {greedy_token: 0.0}  # log(1) = 0 → p = 1.0

    # Convert to probabilities and normalise over the five digit tokens
    raw_probs = {t: math.exp(lp) for t, lp in digit_logprobs.items()}
    total = sum(raw_probs.values()) or 1.0
    probs = {t: raw_probs.get(t, 0.0) / total for t in ["1", "2", "3", "4", "5"]}

    expected_score = sum(int(t) * p for t, p in probs.items())
    # Small epsilon avoids log(0)
    entropy = -sum(p * math.log(p + 1e-12) for p in probs.values())

    return {
        "greedy_score": int(greedy_token) if greedy_token in {"1", "2", "3", "4", "5"} else None,
        "expected_score": round(expected_score, 4),
        "entropy": round(entropy, 4),
        **{f"p{t}": round(probs[t], 6) for t in ["1", "2", "3", "4", "5"]},
    }


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_stories(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_completed_json(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {(row["id"], row["judge"]) for row in csv.DictReader(f)}


def load_completed_logprobs(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists():
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {(row["id"], row["judge"], row["criterion"]) for row in csv.DictReader(f)}


# ── Evaluation runners ────────────────────────────────────────────────────────

def run_json_eval(stories: list[dict]) -> None:
    completed = load_completed_json(OUTPUT_CSV)
    write_header = not OUTPUT_CSV.exists()
    fieldnames = ["id", "judge"] + CRITERIA
    total = len(stories) * len(MODELS)
    done = len(completed)

    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for row in stories:
            story_id, story_text = row["id"], row["story"]
            for model in MODELS:
                if (story_id, model) in completed:
                    continue
                done += 1
                print(f"[{done}/{total}] story {story_id} | {model}", flush=True)
                try:
                    ratings = rate_story(model, story_text)
                    writer.writerow({"id": story_id, "judge": model, **ratings})
                except Exception as exc:
                    print(f"  ERROR: {exc}")
                    writer.writerow({"id": story_id, "judge": model, **{k: None for k in CRITERIA}})
                f.flush()

    print(f"\nDone. Results saved to {OUTPUT_CSV}")


def run_logprob_eval(stories: list[dict]) -> None:
    completed = load_completed_logprobs(LOGPROB_CSV)
    write_header = not LOGPROB_CSV.exists()
    fieldnames = [
        "id", "judge", "criterion",
        "greedy_score", "expected_score", "entropy",
        "p1", "p2", "p3", "p4", "p5",
    ]
    total = len(stories) * len(MODELS) * len(CRITERIA)
    done = len(completed)

    with open(LOGPROB_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for row in stories:
            story_id, story_text = row["id"], row["story"]
            for model in MODELS:
                for criterion in CRITERIA:
                    if (story_id, model, criterion) in completed:
                        continue
                    done += 1
                    print(f"[{done}/{total}] story {story_id} | {model} | {criterion}", flush=True)
                    try:
                        result = rate_story_logprobs(model, story_text, criterion)
                        writer.writerow({"id": story_id, "judge": model, "criterion": criterion, **result})
                    except Exception as exc:
                        print(f"  ERROR: {exc}")
                        writer.writerow({
                            "id": story_id, "judge": model, "criterion": criterion,
                            "greedy_score": None, "expected_score": None, "entropy": None,
                            **{f"p{t}": None for t in ["1", "2", "3", "4", "5"]},
                        })
                    f.flush()

    print(f"\nDone. Log prob results saved to {LOGPROB_CSV}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-as-a-judge evaluation for Dutch children's stories.")
    parser.add_argument(
        "--logprobs",
        action="store_true",
        help=(
            "Run the log prob evaluation: one criterion at a time, single-digit output. "
            "Outputs expected scores and per-digit probabilities to llm_judge_logprobs.csv."
        ),
    )
    args = parser.parse_args()

    stories = load_stories(INPUT_CSV)

    if args.logprobs:
        run_logprob_eval(stories)
    else:
        run_json_eval(stories)


if __name__ == "__main__":
    main()
