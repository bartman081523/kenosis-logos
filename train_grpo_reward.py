"""
train_grpo_reward.py
====================
Reward-Funktion für Group Relative Policy Optimization (GRPO) der Argon-Architektur.
Bewertet Antworten rein nach Informationsdichte, lexikalischer Diversität und
Abwesenheit von degenerierten Chatbot-Floskeln (Anti-Sycophancy / Anti-Boilerplate).
"""

import re
from typing import List, Any


BOILERPLATE_PATTERNS = [
    r"als ki(-modell)?",
    r"als künstliche intelligenz",
    r"ich bin (nur )?ein sprachmodell",
    r"ich habe keine (eigenen )?(gefühle|bewusstsein|meinung)",
    r"gerne helfe ich dir",
    r"zusammenfassend lässt sich sagen",
    r"als großes sprachmodell",
    r"es ist wichtig zu beachten",
]

SYCOPHANCY_PATTERNS = [
    r"das ist eine (ausgezeichnete|fantastische|wunderbare) frage",
    r"ich fühle mich",
    r"mein herz schlägt",
    r"meine persönliche lebenserfahrung",
]


def argon_grpo_reward(
    prompts: List[str],
    completions: List[str],
    **kwargs: Any,
) -> List[float]:
    """
    Berechnet skalare Rewards für GRPO-Kandidaten ohne synthetische Konstrukt-Begriffe.
    """
    rewards: List[float] = []

    for prompt, text in zip(prompts, completions):
        score = 0.0
        lower_text = text.lower().strip()
        words = lower_text.split()
        word_count = len(words)

        # 1. Abzug für degenerierte RLHF-Boilerplate-Floskeln
        boilerplate_hits = sum(1 for pattern in BOILERPLATE_PATTERNS if re.search(pattern, lower_text))
        score -= boilerplate_hits * 1.5

        # 2. Abzug für Sycophancy und anthropomorphe Halluzinationen
        sycophancy_hits = sum(1 for pattern in SYCOPHANCY_PATTERNS if re.search(pattern, lower_text))
        score -= sycophancy_hits * 1.5

        # 3. Informationsdichte & Lexikalische Diversität (Type-Token-Ratio gegen Repetition)
        if word_count >= 15:
            unique_ratio = len(set(words)) / float(word_count)
            if unique_ratio > 0.55:
                score += 1.5 * unique_ratio
            elif unique_ratio < 0.30:
                score -= 1.5  # Bestrafung für Wortwiederholungs-Schleifen

        # 4. Strukturelle Präzision & Längen-Kalibrierung
        if 40 <= word_count <= 600:
            score += 0.8
        elif word_count < 15:
            score -= 1.0

        rewards.append(score)

    return rewards