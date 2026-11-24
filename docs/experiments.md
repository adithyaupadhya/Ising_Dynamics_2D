# Experiments and reproducibility

[Home](../README.md) · [Dynamics](dynamics.md) · [Analysis](analysis.md)

The installed `ising` command (equivalently `python -m ising`) provides one interface. Use `ising COMMAND --help` for defaults and all available options. Python experiment functions accept ordinary arguments and do not print or write unless requested.

## Command reference

| Command | Purpose |
|---|---|
| `ising series` | Print a magnetization trajectory |
| `ising scan` | Compare finite-time field scans |
| `ising movie` | Generate a GIF |
| `ising check` | Compare equilibrium estimates with exact enumeration |
| `ising passage` | Measure first-passage times |
| `ising coalescence` | Measure extremal coupling times |
| `ising benchmark` | Compare reference and compiled local updates |

Common model options are `--size` (side length), `--beta` or `--temperature` (reciprocals), and `--field` (rescaled field $\lambda$, not $h_N$). Where applicable, `--dynamics` selects `heatbath`, `metropolis` or `sw`; `--backend` selects `python` or `numba`; `--seed` specifies the random seed. Not all options apply to every command.

## Trajectories and field scans

```bash
ising series --size 32 --beta .5 --field 3 --sweeps 100 --record-every 10
ising scan --size 32 --beta .5 --fields 1 2 3 --sweeps 100
ising movie --size 32 --beta .5 --field 3 --frames 50 --stride 5 -o bubble.gif
```

`series` prints time and magnetization. `scan` runs each dynamics from both extremal initial states for each field and reports mean recorded magnetization over the second half of the trajectory and the final largest-plus fraction. Note that these are finite-time summaries and not equilibrium estimates. `movie` saves a GIF to a new path.

```python
from ising import IsingModel
from ising.experiments import run

state, observations = run(IsingModel(32, .5, 3), 100, record_every=10,
                          observables={"contours": True, "interface_heights": True})
print(observations[-1]["magnetization"])
```

`run` returns a copy of the final padded configuration and a list of observations at time zero, recording intervals and final time. The default omits cluster measurements; pass an explicit dictionary of `observe()` options to collect geometric observables. See [Observables](observables.md).

## Equilibrium diagnostic

```bash
ising check --size 3 --beta .5 --field 3 --burn 500 --steps 2000
```

For $N\leq4$, exact enumeration provides Gibbs expectations of magnetization, bond sum and the all-minus indicator. `equilibrium_check` compares these against each dynamics after burn-in and reports batch-mean standard errors and standardized discrepancies (`z`). A short run or agreement within errors does not prove equilibration.

## First-passage experiments

For magnetization threshold $a\in(-1,1)$, define the upward and downward stopping times

$$\tau_a^+=\inf\{t\geq0:m(\sigma_t)\geq a\},\qquad \tau_a^-=\inf\{t\geq0:m(\sigma_t)\leq a\}.$$

Initialize all minus for upward crossing and all plus for downward crossing. The simulator checks time zero and every elementary attempt (every cluster step for Swendsen-Wang) and not just after every sweep. A crossing is a proxy and does not necessarily imply droplet formation/dissolution or mixing.

```bash
ising passage --size 16 24 32 --beta .5 --field 3 --init minus --threshold 0 --replicates 20 --limit 10000 --workers 4 --seed 42 -o results/nucleation
```

| Option | Meaning |
|---|---|
| `--size` | One or more side lengths |
| `--init` | `minus` or `plus`, determines crossing direction |
| `--threshold` | Magnetization threshold |
| `--replicates` | Independent trials per size |
| `--limit` | Maximum sweep-equivalent observation time |
| `--workers` | Parallel worker processes |
| `--seed` | Master seed |
| `--output` | Study directory |

`event=1` indicates a crossing; `event=0` indicates right censoring at the limit. The recorded time is the exact observed crossing time or censoring time.

## Coalescence experiments

Measure when extremal heat-bath chains driven by the same random updates first agree. The same clock and censoring conventions apply. See the [coupling inequality](dynamics.md#monotone-coupling) for its connection to mixing.

```bash
ising coalescence --size 16 24 32 --beta .5 --field 3 --replicates 20 --limit 10000 --workers 4 --seed 42 -o results/coupling
```

An empirical coupling survival curve estimates the probability appearing in the coupling inequality, it is not a deterministic mixing bound.

## Trial output

| CSV fields | Interpretation |
|---|---|
| `N`, `beta`, `lambda` | Model parameters |
| `backend`, `replicate`, `seed` | Implementation, zero-based replicate index and assigned seed |
| `sweeps`, `updates`, `event` | Observed time, elementary count and event indicator |
| `elapsed_seconds` | Wall-clock trial runtime (may include compilation) |
| Passage: `h`, `dynamics`, `init`, `threshold`, `limit_sweeps`, `final_m` | Field, algorithm, initialization, stopping criterion, limit and final magnetization |
| Coupling: `limit`, `sweep`, `coalesced`, `disagreement_fraction`, `lower_magnetization`, `upper_magnetization` | Limit, time (same as `sweeps`), agreement status and final chain measurements |

Illustrative passage record (selected columns only; not measured data):

| N | replicate | sweeps | updates | event |
|---:|---:|---:|---:|---:|
| 16 | 0 | 12.5 | 3200 | 1 |
| 16 | 1 | 10000 | 2560000 | 0 |

For SW, `backend` is recorded as `n/a`. Analyze censoring explicitly rather than averaging cutoff times as though they were events.

## Python API and checkpointing

```python
from ising.experiments import first_passage

results = first_passage([16, 24, 32], beta=.5, lam=3, replicates=20,
                        limit=10000, seed=42, workers=1)
print(results.rows[0])
results.save("results/passage.csv")
```

`coalescence` provides a corresponding API. Use `output="results/study"` to checkpoint and `resume=True` to resume. `TrialResults.save` writes a separate CSV and will not overwrite files. On Windows and other spawn-based platforms, guard parallel script entry points with `if __name__ == "__main__":`; in notebooks prefer one worker unless multiprocessing is configured.

Both replicated experiments derive independent seeds through NumPy `SeedSequence`. The parent writes in deterministic task order. Worker count does not affect scientific observations; elapsed times may differ. The optional progress callback receives completed count, total count and row.

A study directory contains:

```text
results/nucleation/
    parameters.json
    checkpoint.jsonl
    trials.csv
    figures/
```

`parameters.json` records scientific parameters, dependency versions, Python, platform, source SHA-256 and Git commit/dirty status when available. `checkpoint.jsonl` stores flushed completed records; `trials.csv` is regenerated after all trials finish; `figures/` is reserved for analysis output.

A new run will not overwrite an existing directory. To resume, repeat the original command with `--resume`. Resume checks parameters, source digest, Python and dependency versions and trial identities; an unterminated final checkpoint line is discarded. Interrupted individual trajectories restart from their assigned seeds. Completed trials are retained. Do not run multiple writers against one study. Flushing does not guarantee durability against abrupt hardware failure. Generated results are ignored by Git.
