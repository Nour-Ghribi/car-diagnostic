from __future__ import annotations

from dataclasses import dataclass, field

from backend.schemas import GoalName, IntentName, ScopeLevel, SignalName


@dataclass(frozen=True)
class IntentCard:
    """
    Goal-oriented semantic card used for retrieval and reranking.

    `goal_name` is the internal semantic target.
    `intent_name` is the public project-compatible intent returned downstream.
    """

    goal_name: GoalName
    intent_name: IntentName
    default_scope: ScopeLevel
    description: str
    examples_fr: tuple[str, ...]
    examples_en: tuple[str, ...]
    anti_examples: tuple[str, ...]
    expected_parameters: tuple[str, ...]
    required_signals: tuple[SignalName, ...] = ()
    clarification_question: str | None = None
    default_parameters: dict[str, object] = field(default_factory=dict)
    semantic_hints: tuple[str, ...] = ()

    @property
    def name(self) -> GoalName:
        """Backward-friendly alias used by older helpers."""
        return self.goal_name


INTENT_CARDS: tuple[IntentCard, ...] = (
    IntentCard(
        goal_name="VEHICLE_HEALTH_CHECK",
        intent_name="CHECK_ENGINE_HEALTH",
        default_scope="broad",
        description=(
            "Broad global vehicle health request. Covers overall condition, diagnostic overview, "
            "general health check, and first-line automotive assessment when the user wants to "
            "know if the car or engine seems okay."
        ),
        examples_fr=(
            "je veux connaitre le health de ma voiture",
            "je vx conaitre l etat de ma voiture",
            "je veux connaitre l etat de mon vehicule",
            "le moteur est il en bon etat",
            "est ce que le moteur est bon",
            "ma voiture va bien ?",
            "j ai quelques doutes sur ma voiture",
            "fais moi un bilan complet",
            "diagnostic general voiture",
            "est ce que tout va bien avec ma voiture",
            "check la sante de ma bagnole",
            "etat general de la voiture",
        ),
        examples_en=(
            "check my car health",
            "is my car okay",
            "is the engine good",
            "is the motor okay",
            "overall vehicle health",
            "run a global health check",
            "full vehicle diagnostic",
            "i have doubts about my car condition",
        ),
        anti_examples=(
            "get vin",
            "show rpm only",
            "read dtc only",
            "check cylinder 2",
        ),
        expected_parameters=("goal", "scope", "detail"),
        required_signals=("rpm", "engine_load", "coolant_temp", "vehicle_speed", "module_voltage"),
        clarification_question="Voulez-vous un bilan general du vehicule ou un probleme precis comme un voyant, une batterie faible ou une surchauffe ?",
        default_parameters={"detail": "medium"},
        semantic_hints=("vehicle", "engine", "health", "overall", "diagnostic", "doubt"),
    ),
    IntentCard(
        goal_name="READ_DTC",
        intent_name="READ_DTC",
        default_scope="specific",
        description=(
            "Read, summarize, or explain diagnostic trouble codes already present in the ECU. "
            "Focused on stored, pending, and permanent codes."
        ),
        examples_fr=(
            "lis les codes defaut",
            "quels sont les dtc presents",
            "montre moi les codes moteur",
            "je veux voir les defauts enregistres",
            "lecture des dtc",
        ),
        examples_en=(
            "read dtc",
            "read fault codes",
            "show stored codes",
            "list pending codes",
            "show engine fault codes",
        ),
        anti_examples=("show rpm", "check battery health", "get vin", "overall car health"),
        expected_parameters=("include_pending", "include_permanent"),
        default_parameters={"include_pending": True, "include_permanent": False},
        semantic_hints=("dtc", "fault", "code"),
    ),
    IntentCard(
        goal_name="CYLINDER_CHECK",
        intent_name="CHECK_CYLINDER",
        default_scope="specific",
        description=(
            "Investigate one specific cylinder for misfire suspicion, combustion imbalance, or "
            "cylinder-specific anomalies."
        ),
        examples_fr=(
            "verifie le cylindre 2",
            "analyse le cylindre 3",
            "misfire cylindre 4",
            "controle cylindre 2",
        ),
        examples_en=(
            "check cylinder 2",
            "inspect cylinder 3",
            "is cylinder 1 misfiring",
            "possible misfire on cylinder 5",
        ),
        anti_examples=("read dtc", "show rpm", "vehicle health", "get vin"),
        expected_parameters=("cylinder_index", "detail"),
        required_signals=("rpm", "engine_load", "stft_b1", "ltft_b1", "o2_b1s1"),
        default_parameters={"detail": "medium"},
        clarification_question="Quel cylindre voulez-vous verifier ?",
        semantic_hints=("engine", "performance", "cylinder"),
    ),
    IntentCard(
        goal_name="SIGNAL_STATUS_CHECK",
        intent_name="CHECK_SIGNAL_STATUS",
        default_scope="specific",
        description=(
            "Inspect one live signal or sensor value such as RPM, engine load, coolant temperature, "
            "fuel trims, O2, vehicle speed, or module voltage."
        ),
        examples_fr=(
            "montre les rpm",
            "affiche le regime moteur",
            "temperature eau ?",
            "quelle est la charge moteur",
            "je veux voir le capteur o2",
        ),
        examples_en=(
            "show rpm",
            "show engine speed",
            "show coolant temperature",
            "display engine load",
            "check module voltage",
        ),
        anti_examples=("overall vehicle health", "read dtc", "get vin", "check cylinder 2"),
        expected_parameters=("signal", "max_age_ms", "goal", "scope"),
        required_signals=("rpm",),
        clarification_question="Quel signal voulez-vous verifier ?",
        default_parameters={"signal": "rpm", "max_age_ms": 2000},
        semantic_hints=("signal", "rpm", "temperature", "battery"),
    ),
    IntentCard(
        goal_name="WARNING_LIGHT_CHECK",
        intent_name="EXPLAIN_WARNING_LIGHT",
        default_scope="specific",
        description=(
            "Explain an active warning light, especially the check engine light, using DTCs and "
            "supporting live data."
        ),
        examples_fr=(
            "pourquoi le voyant moteur est allume",
            "explique le check engine",
            "pourquoi j ai le voyant moteur",
        ),
        examples_en=(
            "why is the check engine light on",
            "explain the check engine light",
            "mil explanation",
        ),
        anti_examples=("read dtc only", "get vin", "show rpm", "global vehicle health"),
        expected_parameters=("warning_type",),
        required_signals=("rpm", "engine_load", "coolant_temp"),
        default_parameters={"warning_type": "check_engine"},
        semantic_hints=("warning_light", "dtc"),
    ),
    IntentCard(
        goal_name="BATTERY_CHECK",
        intent_name="CHECK_SIGNAL_STATUS",
        default_scope="specific",
        description=(
            "Battery or charging-system oriented check. Covers weak battery complaints, low module "
            "voltage, and simple charging health observations."
        ),
        examples_fr=(
            "pourquoi ma batterie est faible",
            "batterie faible",
            "verifie la tension batterie",
            "probleme batterie voiture",
        ),
        examples_en=(
            "why is my battery weak",
            "check battery health",
            "battery voltage status",
            "battery seems low",
        ),
        anti_examples=("show coolant temperature", "read dtc", "get vin", "global vehicle health"),
        expected_parameters=("signal", "goal", "scope"),
        required_signals=("module_voltage",),
        clarification_question="Voulez-vous verifier la tension batterie ou un probleme de demarrage ?",
        default_parameters={"signal": "module_voltage", "max_age_ms": 2000},
        semantic_hints=("battery", "vehicle"),
    ),
    IntentCard(
        goal_name="ENGINE_TEMPERATURE_CHECK",
        intent_name="CHECK_SIGNAL_STATUS",
        default_scope="specific",
        description=(
            "Temperature-focused request. Covers overheating concerns and checks around coolant "
            "temperature or high engine temperature."
        ),
        examples_fr=(
            "temperature moteur trop haute",
            "temperature eau trop elevee",
            "surchauffe moteur",
            "verifie la temperature moteur",
        ),
        examples_en=(
            "engine temperature too high",
            "check coolant temperature",
            "is the engine overheating",
            "coolant temp status",
        ),
        anti_examples=("check battery", "read dtc", "get vin", "global vehicle health"),
        expected_parameters=("signal", "goal", "scope"),
        required_signals=("coolant_temp",),
        clarification_question="Voulez-vous la valeur de temperature ou un diagnostic moteur plus large ?",
        default_parameters={"signal": "coolant_temp", "max_age_ms": 2000},
        semantic_hints=("temperature", "engine"),
    ),
    IntentCard(
        goal_name="PERFORMANCE_ISSUE_CHECK",
        intent_name="CHECK_ENGINE_HEALTH",
        default_scope="specific",
        description=(
            "Performance complaint such as hesitation, lack of power, rough running, or weak "
            "acceleration. More specific than a global health request but still engine-centric."
        ),
        examples_fr=(
            "la voiture manque de puissance",
            "moteur hesite",
            "probleme de performance moteur",
            "acceleration faible",
        ),
        examples_en=(
            "performance issue",
            "engine feels weak",
            "car has hesitation",
            "poor acceleration",
        ),
        anti_examples=("get vin", "show rpm only", "read dtc only"),
        expected_parameters=("goal", "scope", "detail"),
        required_signals=("rpm", "engine_load", "stft_b1", "ltft_b1", "o2_b1s1"),
        clarification_question="Voulez-vous un bilan moteur global ou un symptome precis comme une hesitation ou un manque de puissance ?",
        default_parameters={"detail": "medium"},
        semantic_hints=("performance", "engine"),
    ),
    IntentCard(
        goal_name="STARTING_PROBLEM_CHECK",
        intent_name="CHECK_ENGINE_HEALTH",
        default_scope="specific",
        description=(
            "Starting or cranking difficulty. Useful for battery or engine-start complaints that "
            "still fall within a limited OBD/telemetry diagnostic scope."
        ),
        examples_fr=(
            "probleme de demarrage",
            "la voiture demarre mal",
            "demarrage difficile",
        ),
        examples_en=(
            "starting problem",
            "car struggles to start",
            "hard start issue",
        ),
        anti_examples=("get vin", "show rpm", "read dtc only"),
        expected_parameters=("goal", "scope", "detail"),
        required_signals=("module_voltage", "rpm"),
        clarification_question="Le probleme concerne-t-il la batterie, le demarrage, ou voulez-vous un bilan global ?",
        default_parameters={"detail": "medium"},
        semantic_hints=("starting", "battery", "engine"),
    ),
    IntentCard(
        goal_name="FUEL_CONSUMPTION_CHECK",
        intent_name="CHECK_ENGINE_HEALTH",
        default_scope="specific",
        description=(
            "Fuel consumption or mixture-oriented concern. Useful for high consumption, lean/rich "
            "condition concerns, and general efficiency questions."
        ),
        examples_fr=(
            "consommation trop elevee",
            "la voiture consomme beaucoup",
            "probleme de carburant",
        ),
        examples_en=(
            "fuel consumption too high",
            "why is my car using more fuel",
            "fuel economy issue",
        ),
        anti_examples=("get vin", "show rpm", "battery weak"),
        expected_parameters=("goal", "scope", "detail"),
        required_signals=("stft_b1", "ltft_b1", "o2_b1s1", "vehicle_speed"),
        clarification_question="Voulez-vous un bilan moteur global ou un focus sur la consommation ?",
        default_parameters={"detail": "medium"},
        semantic_hints=("fuel", "performance", "engine"),
    ),
    IntentCard(
        goal_name="VEHICLE_CONTEXT_LOOKUP",
        intent_name="GET_VEHICLE_CONTEXT",
        default_scope="specific",
        description=(
            "Retrieve vehicle identity or ECU metadata such as VIN, ECU name, calibration id, and "
            "supported diagnostic capabilities."
        ),
        examples_fr=(
            "donne le vin",
            "quel est le contexte vehicule",
            "affiche l ecu",
            "quelles capacites obd sont supportees",
        ),
        examples_en=(
            "get vin",
            "show vehicle context",
            "show ecu information",
            "supported capabilities",
        ),
        anti_examples=("global vehicle health", "show rpm", "read dtc"),
        expected_parameters=("include_calibration", "refresh_capabilities"),
        clarification_question="Voulez-vous le VIN, les infos ECU, ou les capacites de diagnostic ?",
        default_parameters={"include_calibration": True, "refresh_capabilities": False},
        semantic_hints=("context", "vehicle"),
    ),
    IntentCard(
        goal_name="UNKNOWN",
        intent_name="UNKNOWN",
        default_scope="specific",
        description=(
            "Fallback goal for unsupported, unrelated, or incomprehensible prompts. Use only when "
            "the request is not meaningfully automotive or when clarification still cannot ground it."
        ),
        examples_fr=(
            "bonjour",
            "aide moi",
            "repare toute la voiture",
            "fais tout automatiquement",
        ),
        examples_en=(
            "hello",
            "help me",
            "fix everything automatically",
            "tell me everything",
        ),
        anti_examples=("check my car health", "read dtc", "show rpm", "get vin"),
        expected_parameters=(),
        clarification_question="Pouvez-vous preciser ce que vous voulez savoir sur le vehicule ?",
        semantic_hints=(),
    ),
)


def get_intent_cards() -> tuple[IntentCard, ...]:
    """Return all supported semantic goal cards in deterministic order."""
    return INTENT_CARDS


def get_intent_card(name: str) -> IntentCard:
    """Return one card by goal name or by public intent name."""
    for card in INTENT_CARDS:
        if card.goal_name == name or card.intent_name == name:
            return card
    raise KeyError(f"Unknown intent card: {name}")


def render_intent_card(card: IntentCard) -> str:
    """Render a goal card into one text block suitable for semantic embeddings."""
    sections = [
        f"goal: {card.goal_name}",
        f"intent: {card.intent_name}",
        f"scope: {card.default_scope}",
        f"description: {card.description}",
        "examples_fr: " + " | ".join(card.examples_fr),
        "examples_en: " + " | ".join(card.examples_en),
        "anti_examples: " + " | ".join(card.anti_examples),
        "semantic_hints: " + " | ".join(card.semantic_hints),
        "required_signals: " + ", ".join(card.required_signals),
        "expected_parameters: " + ", ".join(card.expected_parameters),
    ]
    if card.clarification_question:
        sections.append(f"clarification_question: {card.clarification_question}")
    return "\n".join(sections)
