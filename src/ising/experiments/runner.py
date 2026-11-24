"""Deterministic replicated trials, checkpointed results and metadata."""
import csv
import hashlib
import json
import multiprocessing
import platform
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np


@dataclass
class TrialResults:
    rows: list[dict]
    parameters: dict

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)


def make_tasks(models, replicates: int, seed: int, options: dict) -> list[dict]:
    if replicates < 1 or not models or seed < 0:
        raise ValueError('Require models, positive replicates and nonnegative seed')
    streams = iter(np.random.SeedSequence(seed).spawn(len(models)*replicates))
    return [dict(model=model, replicate=r, seed=int(next(streams).generate_state(1, dtype=np.uint64)[0]), **options)
            for model in models for r in range(replicates)]


def environment() -> dict:
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], stderr=subprocess.DEVNULL, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit = dirty = None
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parents[1]
    for source in sorted(root.rglob("*.py")):
        digest.update(str(source.relative_to(root)).encode())
        digest.update(source.read_bytes())
    return dict(source_sha256=digest.hexdigest(), python=sys.version, platform=platform.platform(), git_commit=commit, git_dirty=dirty,
                dependencies={name: version(name) for name in ('numpy','scipy','numba')})


def run_trials(trial, tasks: list[dict], parameters: dict, *, workers: int = 1,
               output: str | Path | None = None, resume: bool = False, progress=None) -> TrialResults:
    """Run in deterministic order; write each completed row to a JSONL checkpoint.

    Output is a study directory containing parameters.json, checkpoint.jsonl and trials.csv.
    Resume requires identical parameters; worker count may change.
    """
    if workers < 1 or not tasks or (resume and output is None):
        raise ValueError('Require positive workers, tasks, and output for resume')
    rows = []
    checkpoint = None
    directory = Path(output) if output is not None else None
    if directory is not None:
        metadata = directory / 'parameters.json'
        journal = directory / 'checkpoint.jsonl'
        if resume:
            saved = json.loads(metadata.read_text())
            if saved['parameters'] != parameters:
                raise ValueError('Resume parameters do not match existing study')
            current = environment()
            for key in ('source_sha256', 'python', 'dependencies'):
                if saved['environment'][key] != current[key]:
                    raise ValueError(f'Resume environment differs: {key}')
            # A crash can leave an incomplete final record; truncate only that record.
            if journal.exists():
                lines = journal.read_bytes().splitlines(keepends=True)
                valid = []
                for line in lines:
                    if not line.endswith(b'\n'):
                        break
                    valid.append(line)
                    rows.append(json.loads(line))
                journal.write_bytes(b''.join(valid))
            if len(rows) > len(tasks):
                raise ValueError('Checkpoint has too many records')
            for row, task in zip(rows, tasks):
                if row['seed'] != task['seed'] or row['replicate'] != task['replicate'] or row['N'] != task['model'].N:
                    raise ValueError('Checkpoint does not match task order')
        else:
            directory.mkdir(parents=True, exist_ok=False)
            (directory / 'figures').mkdir()
            metadata.write_text(json.dumps(dict(parameters=parameters, environment=environment()), indent=2)+'\n')
        checkpoint = journal.open('a', encoding='utf-8')
    def collect(iterator):
        for row in iterator:
            if checkpoint is not None:
                checkpoint.write(json.dumps(row, allow_nan=False)+'\n')
                checkpoint.flush()
            rows.append(row)
            if progress is not None:
                progress(len(rows), len(tasks), row)
    try:
        if workers == 1:
            collect(map(trial, tasks[len(rows):]))
        else:
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
                collect(pool.map(trial, tasks[len(rows):]))
    finally:
        if checkpoint is not None:
            checkpoint.close()
    result = TrialResults(rows, parameters)
    if directory is not None:
        temporary = directory / 'trials.csv.tmp'
        temporary.unlink(missing_ok=True)
        result.save(temporary)
        temporary.replace(directory / 'trials.csv')
    return result
