"""Steady background traffic through the gateway (for the demo dashboard), using the official OpenAI SDK.

    python -m demo.traffic --rps 2 --seconds 300
"""

from __future__ import annotations

import argparse
import random
import threading
import time

import openai

from src.prompts import SAMPLE_PROMPTS

ROUTES = ["smart"] * 5 + ["fast"] * 3 + ["self-hosted"] * 2


def one(client: openai.OpenAI, rng: random.Random) -> None:
    route = rng.choice(ROUTES)
    prompt = rng.choice(SAMPLE_PROMPTS) + ("" if route == "self-hosted" else f" (#{rng.randint(1, 10**6)})")
    try:
        if rng.random() < 0.5:
            for _ in client.chat.completions.create(model=route, stream=True,
                                                    messages=[{"role": "user", "content": prompt}]):
                pass
        else:
            client.chat.completions.create(model=route, messages=[{"role": "user", "content": prompt}])
    except openai.APIError:
        pass   # failures are what the dashboard is for


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--rps", type=float, default=2.0)
    ap.add_argument("--seconds", type=float, default=300)
    args = ap.parse_args()
    client = openai.OpenAI(base_url=args.url, api_key="demo", max_retries=0)
    rng = random.Random(42)
    end = time.time() + args.seconds
    while time.time() < end:
        threading.Thread(target=one, args=(client, rng), daemon=True).start()
        time.sleep(rng.expovariate(args.rps))


if __name__ == "__main__":
    main()
