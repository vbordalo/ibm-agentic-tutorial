from __future__ import annotations

import json
import subprocess
import time
import uuid

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


@dataclass
class RunTrace:
    model: str
    temperature: float
    task: str
    dataset_name: str | None = None
    runs_dir: str = "runs"

    run_id: str = field(init=False)
    run_dir: Path = field(init=False)
    events_path: Path = field(init=False)

    started_at: str = field(init=False)
    started_perf: float = field(init=False)

    sequence: int = field(default=0, init=False)
    metrics: dict[str, int] = field(default_factory=dict, init=False)

    def __post_init__(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = uuid.uuid4().hex[:6]

        self.run_id = f"{timestamp}_{suffix}"
        self.started_at = utc_now_iso()
        self.started_perf = time.perf_counter()

        date_dir = datetime.now().strftime("%Y-%m-%d")

        self.run_dir = Path(self.runs_dir) / date_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.events_path = self.run_dir / "events.jsonl"

        self.metrics = {
            "planner_calls": 0,
            "coder_calls": 0,
            "executor_calls": 0,
            "reviewer_calls": 0,
            "execution_errors": 0,
        }

        self.log_event(
            event="run_started",
            data={
                "task": self.task,
                "dataset_name": self.dataset_name,
                "model": self.model,
                "temperature": self.temperature,
                "git_commit": get_git_commit(),
            },
        )

    def log_event(
        self,
        event: str,
        node: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        self.sequence += 1

        record = {
            "seq": self.sequence,
            "timestamp": utc_now_iso(),
            "event": event,
            "node": node,
            "data": data or {},
        }

        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def increment(self, metric: str, amount: int = 1) -> None:
        self.metrics[metric] = self.metrics.get(metric, 0) + amount

    def finish(
        self,
        status: str,
        final_state: dict[str, Any] | None = None,
    ) -> None:
        elapsed_seconds = time.perf_counter() - self.started_perf

        self.log_event(
            event="run_completed",
            data={
                "status": status,
                "elapsed_seconds": elapsed_seconds,
            },
        )

        summary = {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": utc_now_iso(),
            "status": status,
            "elapsed_seconds": elapsed_seconds,
            "system": {
                "git_commit": get_git_commit(),
                "model": self.model,
                "temperature": self.temperature,
            },
            "input": {
                "task": self.task,
                "dataset_name": self.dataset_name,
            },
            "metrics": self.metrics,
            "final_state": final_state or {},
        }

        summary_path = self.run_dir / "summary.json"

        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(
                summary,
                f,
                indent=2,
                ensure_ascii=False,
                default=str,
            )