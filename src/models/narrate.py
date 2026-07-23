# src/models/narrate.py

"""
LLM-powered narrative explanation for loan decisions.

Takes the structured SHAP output from /explain and generates a plain-English
summary for non-technical loan officers via the Groq API.
"""

import logging
import os

from groq import Groq

logger = logging.getLogger(__name__)

_client: Groq | None = None

MODEL = "llama-3.3-70b-versatile"
MAX_TOKENS = 400


def _get_client() -> Groq:
    """Lazy singleton — avoids creating the client at import time."""
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set.")
        _client = Groq(api_key=api_key)
    return _client


def _build_prompt(
    default_probability: float,
    risk_tier: str,
    baseline_probability: float,
    top_features: list[dict],
) -> str:
    """Build a structured prompt from SHAP output."""
    prob_pct = round(default_probability * 100, 2)
    baseline_pct = round(baseline_probability * 100, 2)

    feature_lines = []
    for f in top_features:
        direction = "increases" if f["direction"] == "increases_risk" else "decreases"
        value_str = (
            f"(value: {f['feature_value']})"
            if f["feature_value"] is not None
            else "(value: unknown)"
        )
        feature_lines.append(
            f"  - {f['feature']}: {direction} default risk {value_str}, SHAP impact: {round(f['shap_value'], 3)}"
        )

    features_block = "\n".join(feature_lines)

    return f"""You are a credit risk analyst assistant. Explain the following loan application assessment
to a non-technical loan officer in 3-4 clear sentences. Be specific about the key factors.
Do not use technical jargon like 'SHAP values'. Do not repeat the numbers verbatim — interpret them.

Assessment summary:
- Default probability: {prob_pct}% (population baseline: {baseline_pct}%)
- Risk tier: {risk_tier}

Top factors driving this assessment:
{features_block}

Write a concise, professional explanation suitable for a loan officer making a credit decision."""


def generate_narrative(
    default_probability: float,
    risk_tier: str,
    baseline_probability: float,
    top_features: list[dict],
) -> str:
    """
    Call Groq LLM to generate a plain-English explanation.
    Returns a fallback message if the API call fails — never blocks the response.
    """
    try:
        prompt = _build_prompt(default_probability, risk_tier, baseline_probability, top_features)
        client = _get_client()

        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            temperature=0.3,  # low temperature for consistent, professional tone
            messages=[{"role": "user", "content": prompt}],
        )

        narrative = response.choices[0].message.content.strip()
        logger.info("Narrative generated successfully.")
        return narrative

    except Exception as e:
        logger.warning(f"Narrative generation failed: {e}")
        return (
            f"This application has a {round(default_probability * 100, 2)}% estimated default probability "
            f"and is classified as {risk_tier} risk. Narrative explanation unavailable."
        )
