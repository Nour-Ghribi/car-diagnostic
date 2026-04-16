from __future__ import annotations

from dataclasses import dataclass, field

from backend.schemas import GoalName, IntentName, ScopeLevel


@dataclass(frozen=True)
class IntentCard:
    """
    Thin public goal contract used by the rest of the project.

    This file is intentionally lightweight:
    - stable public goal <-> intent contract
    - default scope / parameters
    - short public-facing description

    Detailed semantic business knowledge must live in the normalized CSV profile
    layer, not here.
    """

    goal_name: GoalName
    intent_name: IntentName
    default_scope: ScopeLevel
    description: str
    expected_parameters: tuple[str, ...]
    clarification_question: str | None = None
    default_parameters: dict[str, object] = field(default_factory=dict)

    @property
    def name(self) -> GoalName:
        """Backward-friendly alias used by older helpers."""
        return self.goal_name

    # Deprecated compatibility shims for legacy code paths.
    @property
    def examples_fr(self) -> tuple[str, ...]:
        return ()

    @property
    def examples_en(self) -> tuple[str, ...]:
        return ()

    @property
    def anti_examples(self) -> tuple[str, ...]:
        return ()

    @property
    def required_signals(self) -> tuple[str, ...]:
        return ()

    @property
    def semantic_hints(self) -> tuple[str, ...]:
        return ()


INTENT_CARDS: tuple[IntentCard, ...] = (
    IntentCard(
        goal_name="VEHICLE_HEALTH_CHECK",
        intent_name="CHECK_ENGINE_HEALTH",
        default_scope="broad",
        description="Public broad vehicle or engine health-check request.",
        expected_parameters=("goal", "scope", "detail"),
        clarification_question="Voulez-vous un bilan general du vehicule ou un probleme plus precis ?",
        default_parameters={"detail": "medium"},
    ),
    IntentCard(
        goal_name="READ_DTC",
        intent_name="READ_DTC",
        default_scope="specific",
        description="Public request to read or summarize diagnostic trouble codes.",
        expected_parameters=("include_pending", "include_permanent"),
        default_parameters={"include_pending": True, "include_permanent": False},
    ),
    IntentCard(
        goal_name="CYLINDER_CHECK",
        intent_name="CHECK_CYLINDER",
        default_scope="specific",
        description="Public request to inspect one specific cylinder.",
        expected_parameters=("cylinder_index", "detail"),
        clarification_question="Quel cylindre voulez-vous verifier ?",
        default_parameters={"detail": "medium"},
    ),
    IntentCard(
        goal_name="SIGNAL_STATUS_CHECK",
        intent_name="CHECK_SIGNAL_STATUS",
        default_scope="specific",
        description="Public request to inspect one live signal or sensor value.",
        expected_parameters=("signal", "max_age_ms", "goal", "scope"),
        clarification_question="Quel signal voulez-vous verifier ?",
        default_parameters={"signal": "rpm", "max_age_ms": 2000},
    ),
    IntentCard(
        goal_name="WARNING_LIGHT_CHECK",
        intent_name="EXPLAIN_WARNING_LIGHT",
        default_scope="specific",
        description="Public request to explain an active warning light.",
        expected_parameters=("warning_type",),
        default_parameters={"warning_type": "check_engine"},
    ),
    IntentCard(
        goal_name="BATTERY_CHECK",
        intent_name="CHECK_SIGNAL_STATUS",
        default_scope="specific",
        description="Public request related to battery or charging observations.",
        expected_parameters=("signal", "goal", "scope"),
        clarification_question="Voulez-vous verifier la tension batterie ou un probleme de demarrage ?",
        default_parameters={"signal": "module_voltage", "max_age_ms": 2000},
    ),
    IntentCard(
        goal_name="ENGINE_TEMPERATURE_CHECK",
        intent_name="CHECK_SIGNAL_STATUS",
        default_scope="specific",
        description="Public request related to engine or coolant temperature.",
        expected_parameters=("signal", "goal", "scope"),
        clarification_question="Voulez-vous la valeur de temperature ou un diagnostic plus large ?",
        default_parameters={"signal": "coolant_temp", "max_age_ms": 2000},
    ),
    IntentCard(
        goal_name="PERFORMANCE_ISSUE_CHECK",
        intent_name="CHECK_ENGINE_HEALTH",
        default_scope="specific",
        description="Public request about a performance or drivability issue.",
        expected_parameters=("goal", "scope", "detail"),
        clarification_question="Voulez-vous un bilan moteur global ou un symptome de performance plus precis ?",
        default_parameters={"detail": "medium"},
    ),
    IntentCard(
        goal_name="STARTING_PROBLEM_CHECK",
        intent_name="CHECK_ENGINE_HEALTH",
        default_scope="specific",
        description="Public request about a starting or cranking problem.",
        expected_parameters=("goal", "scope", "detail"),
        clarification_question="Le probleme concerne-t-il le demarrage, la batterie, ou un bilan global ?",
        default_parameters={"detail": "medium"},
    ),
    IntentCard(
        goal_name="FUEL_CONSUMPTION_CHECK",
        intent_name="CHECK_ENGINE_HEALTH",
        default_scope="specific",
        description="Public request about fuel consumption or mixture concerns.",
        expected_parameters=("goal", "scope", "detail"),
        clarification_question="Voulez-vous un bilan moteur global ou un focus sur la consommation ?",
        default_parameters={"detail": "medium"},
    ),
    IntentCard(
        goal_name="VEHICLE_CONTEXT_LOOKUP",
        intent_name="GET_VEHICLE_CONTEXT",
        default_scope="specific",
        description="Public request for VIN, ECU metadata, or diagnostic capabilities.",
        expected_parameters=("include_calibration", "refresh_capabilities"),
        clarification_question="Voulez-vous le VIN, les infos ECU, ou les capacites de diagnostic ?",
        default_parameters={"include_calibration": True, "refresh_capabilities": False},
    ),
    IntentCard(
        goal_name="UNKNOWN",
        intent_name="UNKNOWN",
        default_scope="specific",
        description="Fallback public contract for unsupported or unclear automotive requests.",
        expected_parameters=(),
        clarification_question="Pouvez-vous preciser ce que vous voulez savoir sur le vehicule ?",
    ),
)


def get_intent_cards() -> tuple[IntentCard, ...]:
    """Return all supported public goal contracts in deterministic order."""
    return INTENT_CARDS


def get_intent_card(name: str) -> IntentCard:
    """Return one card by goal name or by public intent name."""
    for card in INTENT_CARDS:
        if card.goal_name == name or card.intent_name == name:
            return card
    raise KeyError(f"Unknown intent card: {name}")


def render_intent_card(card: IntentCard) -> str:
    """Render a thin public contract block for embeddings and logging."""
    sections = [
        f"goal: {card.goal_name}",
        f"intent: {card.intent_name}",
        f"scope: {card.default_scope}",
        f"description: {card.description}",
        "expected_parameters: " + ", ".join(card.expected_parameters),
    ]
    if card.default_parameters:
        sections.append("default_parameters: " + ", ".join(f"{key}={value}" for key, value in card.default_parameters.items()))
    if card.clarification_question:
        sections.append(f"clarification_question: {card.clarification_question}")
    return "\n".join(sections)
