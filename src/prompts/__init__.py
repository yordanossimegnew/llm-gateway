"""The gateway forwards prompts unchanged: it never rewrites what a client sends.

What lives here are the sample prompts the CLI and the Streamlit playground offer.
"""

SAMPLE_PROMPTS = [
    "In two sentences, why do LLM gateways put a circuit breaker in front of each provider?",
    "Write a haiku about a request that fell back to its second model.",
    "List three things to log for every LLM call, one line each.",
    "Explain exponential backoff with jitter to a new engineer in under 60 words.",
]
