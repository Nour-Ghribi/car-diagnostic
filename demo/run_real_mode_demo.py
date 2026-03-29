from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.embedding_provider import get_embedding_provider, get_embedding_provider_name
from agent.orchestrator import diagnose
from agent.reranker import get_reranker_provider_name
from agent.response_formatter import format_agent_response
from agent.settings import clear_settings_cache, get_settings
from backend.schemas import AgentRequest


def run_demo_once(vehicle_id: str, user_prompt: str, out_stream: TextIO | None = None) -> None:
    """Run one end-to-end diagnostic request and print a CLI-friendly report."""
    stream = out_stream or sys.stdout
    clear_settings_cache()
    settings = get_settings()
    embedding_provider = get_embedding_provider(settings)
    embedding_label = get_embedding_provider_name(embedding_provider)
    reranker_label = get_reranker_provider_name(settings)

    print("=== Real Mode Demo (OBD mocked) ===", file=stream)
    print(f"Embedding provider : {embedding_label}", file=stream)
    print(f"Reranker provider  : {reranker_label}", file=stream)
    print("OBD/data layer     : mock-backed tools", file=stream)
    print("", file=stream)

    request = AgentRequest(
        request_id=f"demo_{int(datetime.now(timezone.utc).timestamp())}",
        ts=datetime.now(timezone.utc),
        vehicle_id=vehicle_id,
        user_prompt=user_prompt,
    )

    try:
        response = diagnose(request)
        print(format_agent_response(response), file=stream)
    except Exception as exc:  # pragma: no cover - defensive CLI path
        print(f"Demo execution failed safely: {exc}", file=stream)


def main() -> int:
    """Interactive CLI entry point for realistic local testing."""
    try:
        vehicle_id = input("Vehicle ID [veh_001]: ").strip() or "veh_001"
        user_prompt = input("Natural language query: ").strip()
        if not user_prompt:
            print("A natural language query is required.")
            return 1
        run_demo_once(vehicle_id, user_prompt)
        return 0
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
