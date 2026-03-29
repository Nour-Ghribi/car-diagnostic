from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.embedding_provider import get_embedding_provider, get_embedding_provider_name
from agent.orchestrator import diagnose
from agent.reranker import get_reranker_provider_name
from agent.settings import clear_settings_cache, get_settings
from backend.schemas import AgentRequest


@dataclass(frozen=True)
class PromptCase:
    category: str
    prompt: str


PROMPT_CASES: tuple[PromptCase, ...] = (
    PromptCase("broad_health", "je veux connaître le health de ma voiture"),
    PromptCase("broad_health", "je vx conaitre l etat de ma voiture"),
    PromptCase("broad_health", "quel est l etat de la voiture overall"),
    PromptCase("broad_health", "check my car health"),
    PromptCase("broad_health", "is my car okay"),
    PromptCase("engine_health", "is the moteur is good"),
    PromptCase("signal", "show rpm"),
    PromptCase("signal", "show coolant temperature"),
    PromptCase("battery", "why is my battery weak"),
    PromptCase("battery", "verifie la tension batterie"),
    PromptCase("temperature", "engine temperature too high"),
    PromptCase("temperature", "temperature moteur trop haute"),
    PromptCase("dtc", "read dtc"),
    PromptCase("dtc", "lis les codes defaut"),
    PromptCase("cylinder", "check cylinder 2"),
    PromptCase("cylinder", "analyse le cylindre 3"),
    PromptCase("warning", "voyant moteur allumé"),
    PromptCase("warning", "why is the check engine light on"),
    PromptCase("context", "get vin"),
    PromptCase("context", "quelles capacites obd sont supportees"),
    PromptCase("symptom", "la voiture manque de puissance"),
    PromptCase("symptom", "probleme de demarrage"),
    PromptCase("clarify", "it's weird"),
    PromptCase("clarify", "j ai un doute"),
    PromptCase("unknown", "hello"),
    PromptCase("unknown", "repair my whole car please"),
)


def run_prompt_suite(
    *,
    vehicle_id: str,
    category: str | None = None,
    full_diagnosis: bool = False,
) -> int:
    """Run the prompt suite and print a compact result for each prompt."""
    clear_settings_cache()
    settings = get_settings()
    embedding_provider = get_embedding_provider(settings)
    embedding_label = get_embedding_provider_name(embedding_provider)
    reranker_label = get_reranker_provider_name(settings)

    print("=== Prompt Suite ===")
    print(f"Vehicle ID          : {vehicle_id}")
    print(f"Embedding provider  : {embedding_label}")
    print(f"Reranker provider   : {reranker_label}")
    print("OBD/data layer      : mock-backed tools")
    print("")

    selected_cases = [case for case in PROMPT_CASES if category is None or case.category == category]
    if not selected_cases:
        print(f"No prompts found for category: {category}")
        return 1

    for index, case in enumerate(selected_cases, start=1):
        response = diagnose(
            AgentRequest(
                request_id=f"suite_{index:03d}",
                ts=datetime.now(timezone.utc),
                vehicle_id=vehicle_id,
                user_prompt=case.prompt,
            )
        )
        print(f"[{index:02d}] {case.category}")
        print(f"Prompt   : {case.prompt}")
        print(f"Intent   : {response.intent.name}")
        print(f"Goal     : {response.intent.parameters.goal}")
        print(f"Scope    : {response.intent.parameters.scope}")
        print(f"Policy   : {response.intent.parameters.resolution_policy}")
        print(f"Conf     : intent={response.intent.confidence:.2f} final={response.confidence:.2f}")
        if response.intent.parameters.signal:
            print(f"Signal   : {response.intent.parameters.signal}")
        if response.intent.parameters.cylinder_index:
            print(f"Cylinder : {response.intent.parameters.cylinder_index}")
        print(f"Actions  : {' -> '.join(response.actions_taken) if response.actions_taken else 'None'}")
        if response.intent.parameters.clarification_question:
            print(f"Question : {response.intent.parameters.clarification_question}")
        if full_diagnosis:
            print(f"Diagnosis: {response.diagnosis}")
        else:
            print(f"Diagnosis: {_shorten(response.diagnosis)}")
        print("")

    return 0


def _shorten(text: str, limit: int = 140) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a ready-made suite of automotive prompts against the agent.")
    parser.add_argument("--vehicle-id", default="veh_002", help="Vehicle profile to use. Default: veh_002")
    parser.add_argument(
        "--category",
        choices=sorted({case.category for case in PROMPT_CASES}),
        help="Only run one prompt category.",
    )
    parser.add_argument(
        "--full-diagnosis",
        action="store_true",
        help="Print the full diagnosis text instead of a shortened preview.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    return run_prompt_suite(
        vehicle_id=args.vehicle_id,
        category=args.category,
        full_diagnosis=args.full_diagnosis,
    )


if __name__ == "__main__":
    raise SystemExit(main())
