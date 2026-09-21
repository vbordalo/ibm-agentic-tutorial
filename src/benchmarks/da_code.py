from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import click


@dataclass
class DACodeTask:
    task_id: str
    instruction: str
    task_type: str
    hardness: str
    workspace_path: Path
    gold_path: Path
    eval_config: dict


def load_jsonl_record(path: Path, task_id: str) -> dict:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)

            if record.get("id") == task_id:
                return record

    raise ValueError(
        f"Task ID {task_id!r} was not found in {path}"
    )


def load_da_code_task(
    task_id: str,
    da_code_root: str | Path,
) -> DACodeTask:
    root = Path(da_code_root).expanduser().resolve()

    task_config_path = (
        root
        / "da_code"
        / "configs"
        / "task"
        / "all.jsonl"
    )

    eval_config_path = (
        root
        / "da_code"
        / "configs"
        / "eval"
        / "eval_all.jsonl"
    )

    workspace_path = (
        root
        / "da_code"
        / "source"
        / task_id
    )

    gold_path = (
        root
        / "da_code"
        / "gold"
        / task_id
    )

    task_record = load_jsonl_record(
        task_config_path,
        task_id,
    )

    eval_record = load_jsonl_record(
        eval_config_path,
        task_id,
    )

    if not workspace_path.is_dir():
        raise FileNotFoundError(
            f"Workspace not found: {workspace_path}"
        )

    if not gold_path.is_dir():
        raise FileNotFoundError(
            f"Gold directory not found: {gold_path}"
        )

    return DACodeTask(
        task_id=task_id,
        instruction=task_record["instruction"],
        task_type=task_record["type"],
        hardness=task_record["hardness"],
        workspace_path=workspace_path,
        gold_path=gold_path,
        eval_config=eval_record,
    )


@click.command()
@click.option(
    "--task-id",
    required=True,
    help="DA-Code task identifier, e.g. dm-csv-001.",
)
@click.option(
    "--da-code-root",
    required=True,
    type=click.Path(
        exists=True,
        file_okay=False,
        path_type=Path,
    ),
    help="Path to the cloned DA-Code repository.",
)
def main(
    task_id: str,
    da_code_root: Path,
) -> None:
    task = load_da_code_task(
        task_id=task_id,
        da_code_root=da_code_root,
    )

    click.echo(f"Task ID: {task.task_id}")
    click.echo(f"Type: {task.task_type}")
    click.echo(f"Hardness: {task.hardness}")
    click.echo(f"Workspace: {task.workspace_path}")
    click.echo(f"Gold: {task.gold_path}")
    click.echo()
    click.echo("Instruction:")
    click.echo(task.instruction)
    click.echo()
    click.echo("Workspace files:")

    for path in sorted(task.workspace_path.iterdir()):
        click.echo(f"  - {path.name}")


if __name__ == "__main__":
    main()