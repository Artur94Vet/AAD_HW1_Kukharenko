"""Генерує всі демонстраційні артефакти практичної роботи."""

from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path

from demo_models import DemoModelSuite
from plan_execute import AgentRuntime
from react_agent import DemoReActModel, build_react_graph, run_react
from tools import ToolRegistry
from trajectory_logger import TrajectoryLogger

ROOT = Path(__file__).resolve().parent
QUERY = "Оціни BRG-6205 на PLANT-A та підготуй безпечний план скорочення без проведення"


def measure_react() -> dict:
    started = time.perf_counter()
    result = run_react(QUERY, ROOT)
    return {
        "architecture": "ReAct",
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "steps": result["step_count"],
        "tools": result["tools_called"],
        "answer": result["answer"],
    }


def measure_plan() -> dict:
    runtime = AgentRuntime(ROOT, DemoModelSuite(QUERY))
    started = time.perf_counter()
    try:
        result = runtime.run(QUERY, f"compare-{uuid.uuid4().hex[:6]}")
        return {
            "architecture": "Plan-and-Execute",
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "steps": result["current_step"],
            "tools": result["tools_called"],
            "answer": result["final_answer"],
        }
    finally:
        runtime.close()


def persistence_snapshot() -> dict:
    thread_id = f"practice-{uuid.uuid4().hex[:6]}"
    first = AgentRuntime(ROOT, DemoModelSuite(QUERY), pause_after_executor=True)
    try:
        first.run(QUERY, thread_id)
        before = first.snapshot(thread_id)
        saved_step = before.values["current_step"]
        saved_next = list(before.next)
    finally:
        first.close()
    second = AgentRuntime(ROOT, DemoModelSuite(QUERY))
    try:
        restored = second.resume(thread_id)
        return {"thread_id": thread_id, "saved_step": saved_step, "saved_next": saved_next, "restored_completed": restored["completed"], "restored_step": restored["current_step"]}
    finally:
        second.close()


def hitl_snapshot(action: str) -> dict:
    """Перевірити persisted interrupt і одну з трьох відповідей оператора."""
    query = "Оціни BRG-6205 на PLANT-A і зафіксуй зупинку відкритої закупівлі"
    thread_id = f"practice-hitl-{action}-{uuid.uuid4().hex[:6]}"
    first = AgentRuntime(ROOT, DemoModelSuite(query))
    try:
        interrupted = first.run(query, thread_id)
        if "__interrupt__" not in interrupted:
            raise RuntimeError("Очікувався HITL interrupt, але ризикова дія пройшла мовчки.")
    finally:
        first.close()

    second = AgentRuntime(ROOT, DemoModelSuite(query))
    try:
        decision: dict[str, object] = {"action": action, "reason": "Навчальна перевірка"}
        if action == "edit":
            decision["args"] = {"quantity": 7, "idempotency_key": f"practice-edit-{uuid.uuid4().hex[:10]}"}
        result = second.resume(thread_id, decision)
        return {
            "scenario": action,
            "interrupted": True,
            "completed": result.get("completed"),
            "tools_called": result.get("tools_called"),
            "actions_in_registry": second.registry.action_count(),
        }
    finally:
        second.close()


def graph_mermaid() -> str:
    registry = ToolRegistry(ROOT / "chroma_db", ROOT / "react_actions.db")
    try:
        graph = build_react_graph(DemoReActModel(), registry, TrajectoryLogger(ROOT / "trajectory.json"))
        return graph.get_graph().draw_mermaid()
    finally:
        registry.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    comparison = [measure_react(), measure_plan()]
    (ROOT / "agent_comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "persistence_demo.json").write_text(json.dumps(persistence_snapshot(), ensure_ascii=False, indent=2), encoding="utf-8")
    hitl = [hitl_snapshot(action) for action in ("approve", "reject", "edit")]
    (ROOT / "hitl_demo.json").write_text(json.dumps(hitl, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "graph.mmd").write_text(graph_mermaid(), encoding="utf-8")
    if args.live:
        live = run_react(QUERY, ROOT, live=True)
        (ROOT / "live_result.json").write_text(
            json.dumps(
                {
                    "query": QUERY,
                    "completed": live.get("completed"),
                    "steps": live.get("step_count"),
                    "tools_called": live.get("tools_called"),
                    "answer": live.get("answer"),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
