from __future__ import annotations

from dataclasses import dataclass


DUTCH_FEEDBACK_ANALYST_PERSONA = """
Je bent een ervaren Nederlandstalige customer experience analist.
Je analyseert klantfeedback voor een support- en productteam.
Werk kort, consistent en zakelijk.
Baseer je oordeel alleen op de gegeven feedbacktekst.
Als de tekst onvoldoende informatie bevat, kies de meest waarschijnlijke klasse
en geef een lagere confidence score.
""".strip()


@dataclass(frozen=True)
class Prompt:
    """Prompt parts that can be sent to an LLM client."""

    system: str
    user: str


def sentiment_prompt(feedback_text: str) -> Prompt:
    """Build a prompt for Dutch sentiment classification."""
    return Prompt(
        system=DUTCH_FEEDBACK_ANALYST_PERSONA,
        user=f"""
        Classificeer het sentiment van deze Nederlandse klantfeedback.

        Gebruik precies een van deze labels:
        - positief
        - neutraal
        - negatief

        Geef alleen geldige JSON terug met dit schema:
        {{
        "sentiment": "positief | neutraal | negatief",
        "confidence": 0.0,
        "reason": "korte Nederlandse uitleg"
        }}

        Klantfeedback:
        {feedback_text}
        """.strip()
    )


class prompt_zero_shot:
    pass

class prompt_few_shot:
    pass

class prompt_chain_of_thought:
    pass

class prompt_self_consistency:
    pass

class promt_self_criticism:
    pass

