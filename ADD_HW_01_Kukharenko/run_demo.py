"""CLI та генератор п'яти test cases."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from demo_model import DemoModel
from react_agent import OpenAIModel, build_graph, initial_state
from trajectory_logger import TrajectoryLogger

ROOT = Path(__file__).resolve().parent

CASES = [
    ("TC-001", "Оціни BRG-6205 на PLANT-A", "surplus"),
    ("TC-002", "Оціни BRG-6205 на PLANT-B", "justified"),
    ("TC-003", "Чи можна перемістити BRG-6205 з PLANT-A?", "surplus"),
    ("TC-004", "Оціни VLV-DN50 на PLANT-B", "surplus"),
    ("TC-005", "Оціни UNKNOWN-1 на PLANT-C", "not_found"),
]


def run_case(query: str, *, live: bool, trajectory_path: Path) -> tuple[dict, float]:
    model = OpenAIModel(os.getenv("OPENAI_MODEL", "gpt-4.1")) if live else DemoModel()
    logger = TrajectoryLogger(trajectory_path)
    started = time.perf_counter()
    result = build_graph(model, logger).invoke(initial_state(query))
    return result["final_response"], (time.perf_counter() - started) * 1000


def main() -> None:
    parser = argparse.ArgumentParser(description="ReAct-аналізатор ремонтних запасів")
    parser.add_argument("query", nargs="?")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--env-file")
    args = parser.parse_args()
    if args.env_file:
        load_dotenv(args.env_file)
    if args.query:
        answer, _ = run_case(args.query, live=args.live, trajectory_path=ROOT / "trajectory.json")
        print(json.dumps(answer, ensure_ascii=False, indent=2))
        return
    rows = []
    trajectory_dir = ROOT / "artifacts" / "trajectories"
    for case_id, query, expected in CASES:
        actual, latency = run_case(query, live=args.live, trajectory_path=trajectory_dir / f"{case_id}.json")
        rows.append(
            {
                "case_id": case_id,
                "input_query": query,
                "expected_result": expected,
                "actual_result": actual,
                "steps": len(json.loads((trajectory_dir / f"{case_id}.json").read_text(encoding="utf-8"))),
                "tool_calls": actual["sources"],
                "time_ms": round(latency, 3),
                "passed": actual["status"] == expected,
            }
        )
    (ROOT / "test_results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "trajectory.json").write_text(
        (trajectory_dir / "TC-003.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    print(f"Готово: {sum(row['passed'] for row in rows)}/{len(rows)} cases")


if __name__ == "__main__":
    main()
