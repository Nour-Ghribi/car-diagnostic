from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass


_PHRASE_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("check engine", "warning light"),
    ("voyant moteur", "warning light"),
    ("full diagnostic", "diagnostic general"),
    ("diagnostic global", "diagnostic general"),
    ("overall condition", "overall health"),
    ("overall health", "overall health"),
    ("car health", "vehicle health"),
    ("vehicle health", "vehicle health"),
    ("etat general", "overall health"),
    ("etat de ma voiture", "vehicle health"),
    ("etat de la voiture", "vehicle health"),
    ("health de ma voiture", "vehicle health"),
    ("health de la voiture", "vehicle health"),
    ("bilan complet", "diagnostic general"),
    ("check complet", "diagnostic general"),
)

_CANONICAL_VARIANTS: dict[str, tuple[str, ...]] = {
    "vehicle": (
        "vehicle",
        "vehicule",
        "vehicul",
        "voiture",
        "voitue",
        "voitur",
        "voituer",
        "car",
        "auto",
        "caisse",
        "bagnole",
        "veh",
    ),
    "engine": ("engine", "moteur", "motor"),
    "health": (
        "health",
        "helath",
        "healt",
        "helth",
        "sante",
        "etat",
        "state",
        "condition",
        "forme",
        "good",
        "okay",
        "ok",
        "bien",
    ),
    "overall": ("overall", "global", "general", "complet", "complete", "entier"),
    "diagnostic": ("diagnostic", "diag", "diagno", "bilan", "check", "inspection"),
    "cylinder": ("cylinder", "cylindre", "misfire"),
    "signal": ("signal", "sensor", "capteur", "status", "value"),
    "dtc": ("dtc", "dtcs", "defaut", "defauts", "fault", "faults", "code", "codes"),
    "warning_light": ("warning_light", "mil", "voyant", "warning", "checkengine"),
    "battery": ("battery", "batterie", "voltage", "tension", "weak", "faible"),
    "temperature": (
        "temperature",
        "temp",
        "tempereature",
        "température",
        "coolant",
        "surchauffe",
        "overheat",
        "hot",
    ),
    "rpm": ("rpm", "regime", "tour", "tours"),
    "performance": ("performance", "power", "puissance", "lent", "slow", "hesitation"),
    "starting": ("start", "starting", "demarrage", "demarrer", "demarre", "crank"),
    "fuel": ("fuel", "consumption", "conso", "consommation", "essence", "carburant"),
    "doubt": ("doute", "doutes", "weird", "bizarre", "strange", "odd", "suspect"),
    "context": ("vin", "ecu", "calibration", "context", "contexte", "capabilities"),
    "repair": ("repair", "fix", "whole", "everything", "automatically", "repare", "tout", "auto"),
}

_FR_HINTS = {
    "je",
    "veux",
    "voiture",
    "vehicule",
    "etat",
    "moteur",
    "batterie",
    "temperature",
    "voyant",
    "doute",
    "diagnostic",
}
_EN_HINTS = {
    "check",
    "my",
    "car",
    "vehicle",
    "health",
    "engine",
    "battery",
    "temperature",
    "warning",
    "overall",
    "state",
}
_AUTOMOTIVE_CANONICAL = {
    "vehicle",
    "engine",
    "cylinder",
    "signal",
    "dtc",
    "warning_light",
    "battery",
    "temperature",
    "rpm",
    "performance",
    "starting",
    "fuel",
    "context",
}
_BROAD_CANONICAL = {"health", "overall", "diagnostic", "doubt"}
_SPECIFIC_CANONICAL = {"cylinder", "signal", "dtc", "warning_light", "battery", "temperature", "context"}
_VOCABULARY = sorted({variant for variants in _CANONICAL_VARIANTS.values() for variant in variants})
_VARIANT_TO_CANONICAL = {
    variant: canonical
    for canonical, variants in _CANONICAL_VARIANTS.items()
    for variant in variants
}


@dataclass(frozen=True)
class NormalizedPrompt:
    """Deterministic normalized representation of one user prompt."""

    original_text: str
    normalized_text: str
    language_hint: str
    detected_keywords: tuple[str, ...]
    automotive_context: bool
    broad_request: bool
    spelling_noise: bool
    unsupported_action: bool

    @property
    def embedding_text(self) -> str:
        """Return a retrieval-friendly text containing normalized content and cues."""
        keywords = " ".join(self.detected_keywords)
        return f"{self.normalized_text}\nkeywords: {keywords}".strip()


def normalize_prompt(prompt: str) -> NormalizedPrompt:
    """Normalize noisy FR/EN automotive prompts into canonical semantic terms."""
    ascii_text = _ascii_normalize(prompt)
    normalized_text = _apply_phrase_replacements(ascii_text)
    raw_tokens = _tokenize(normalized_text)

    corrected_tokens: list[str] = []
    spelling_noise = False
    for token in raw_tokens:
        corrected = _correct_token(token)
        if corrected != token:
            spelling_noise = True
        corrected_tokens.append(corrected)

    canonical_keywords = tuple(dict.fromkeys(_detect_keywords(corrected_tokens)))
    language_hint = _detect_language_hint(corrected_tokens)
    automotive_context = any(keyword in _AUTOMOTIVE_CANONICAL for keyword in canonical_keywords)
    broad_request = (
        automotive_context
        and any(keyword in _BROAD_CANONICAL for keyword in canonical_keywords)
        and not any(keyword in _SPECIFIC_CANONICAL for keyword in canonical_keywords)
    )
    unsupported_action = "repair" in canonical_keywords

    rebuilt_text = " ".join(_map_token_to_canonical(token) for token in corrected_tokens)
    rebuilt_text = re.sub(r"\s+", " ", rebuilt_text).strip()

    return NormalizedPrompt(
        original_text=prompt,
        normalized_text=rebuilt_text or ascii_text,
        language_hint=language_hint,
        detected_keywords=canonical_keywords,
        automotive_context=automotive_context,
        broad_request=broad_request,
        spelling_noise=spelling_noise,
        unsupported_action=unsupported_action,
    )


def _ascii_normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = "".join(char for char in normalized if not unicodedata.combining(char))
    ascii_text = ascii_text.lower()
    ascii_text = re.sub(r"[^a-z0-9_ ]+", " ", ascii_text)
    return re.sub(r"\s+", " ", ascii_text).strip()


def _apply_phrase_replacements(text: str) -> str:
    updated = f" {text} "
    for source, target in _PHRASE_REPLACEMENTS:
        updated = updated.replace(f" {source} ", f" {target} ")
    return re.sub(r"\s+", " ", updated).strip()


def _tokenize(text: str) -> list[str]:
    return [token for token in re.split(r"[^a-z0-9_]+", text) if token]


def _correct_token(token: str) -> str:
    if token in _VARIANT_TO_CANONICAL:
        return token
    if len(token) <= 2:
        return token
    match = difflib.get_close_matches(token, _VOCABULARY, n=1, cutoff=0.82)
    return match[0] if match else token


def _map_token_to_canonical(token: str) -> str:
    return _VARIANT_TO_CANONICAL.get(token, token)


def _detect_keywords(tokens: list[str]) -> list[str]:
    keywords: list[str] = []
    for token in tokens:
        canonical = _map_token_to_canonical(token)
        if canonical in _CANONICAL_VARIANTS:
            keywords.append(canonical)
    return keywords


def _detect_language_hint(tokens: list[str]) -> str:
    fr_score = sum(1 for token in tokens if token in _FR_HINTS)
    en_score = sum(1 for token in tokens if token in _EN_HINTS)
    if fr_score and en_score:
        return "mixed"
    if fr_score:
        return "fr"
    if en_score:
        return "en"
    return "unknown"
