"""Pass or fail for each specification R1 to R8, computed from the results store.

    python -m levellens.specs            # write results/tables/specs.md and specs.csv, print the verdicts
    python -m levellens.specs --print    # print them, write nothing

The specifications are frozen in docs/design.md. Nothing is trained and no model is loaded here. Every
number comes from `results/runs.jsonl`, the saved predictions in `results/preds/`, the split manifest
and the reports that `scripts/efficiency.py` reads.

Where the text of a specification leaves a choice, this file makes the same choice everywhere:

- A trial run is never part of a verdict (`levellens.results.TRIAL_MARK`).
- "The best" model or setting is the one with the best mean QWK on dev. The number that is judged is
  then read on test. Nothing is chosen on test.
- R3 on ARC and R5 are about three seeds, so their setting is chosen among the settings that have three.
- R3 has three parts, one per dataset. Each part gets a verdict of its own, and R3 is met only when
  all three are.
- R4 compares the two output heads backbone by backbone. It is met when the ordinal head has the lower
  Balanced DRPS on at least two datasets for at least one backbone. The evidence names every backbone,
  so a backbone on which it does not hold stays visible.
- R6 and R7 are computed by `scripts/efficiency.py`. This file quotes its verdict.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import statistics
from pathlib import Path

from levellens.data import info
from levellens.evaluate import paired_bootstrap
from levellens.paths import results_dir, shown
from levellens.results import (
    TABLE_SPLITS, TRIAL_MARK, leaderboard, load_predictions, load_rows, load_runs, predictions_file, runs_file,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path(__file__).resolve().parent / "data" / "manifest.json"
TRAINED = ("race", "arc", "math")  # the three datasets a model is trained on
EXTRAPOLATION = ("scienceqa",)     # scored by a model that never saw them in training
TITLES = {"race": "RACE++", "arc": "ARC", "math": "MATH", "scienceqa": "ScienceQA"}

# The limits, as frozen in docs/design.md
R1_TRAIN_QUESTIONS = 3000
R2_FAMILIES = ("classical", "recurrent", "transformer", "open LLM")
R2_OUTPUT_TYPES = ("classification", "regression", "ordinal")
R3_RACE_SPEARMAN = 0.79
R3_RACE_SPEARMAN_CAPPED = 0.77  # allowed when the training set had to be capped
R3_ARC_SPEARMAN = 0.50
R4_DATASETS = 2
R5_SPEARMAN_SD = 0.05
R5_PUBLISHED_SD = 0.139
SEEDS = 3

PASS, FAIL, NOT_MEASURED = "pass", "fail", "not measured"
COLUMNS = ("id", "specification", "verdict", "evidence")
SPECIFICATIONS = {
    "R1": "Three trained datasets with more than 3,000 training questions each, plus an extrapolation set",
    "R2": "At least four model families and three output types, all on identical splits",
    "R3 RACE++": "Spearman at least 0.79 on the full training set",
    "R3 ARC": "Mean Spearman at least 0.50 over 3 seeds, for the setting with the best mean QWK on dev",
    "R3 MATH": "The best transformer beats the best classical model on QWK, 95% bootstrap interval above zero",
    "R4": "An ordinal head has lower Balanced DRPS than cross-entropy on at least two of three datasets, same backbone",
    "R5": "ARC, same setting as R3: standard deviation of Spearman over 3 seeds at most 0.05",
    "R6": "Every fine-tune at most 1 GPU-hour, the whole project at most 20 GPU-hours",
    "R7": "Demo: at most 1 second per question on a CPU, DistilBERT, timed over 100 items",
    "R8": "Every run starts from one config; every table and figure is rebuilt from the results store with one command",
}


def efficiency_module():
    """Load scripts/efficiency.py, which owns the model families and the verdict on R6 and R7."""
    spec = importlib.util.spec_from_file_location("levellens_scripts_efficiency", REPO_ROOT / "scripts" / "efficiency.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def result_rows(folder: Path | None = None) -> list[dict]:
    """Return the rows that are results: the newest row of each run, trial runs left out."""
    return [row for row in load_rows(folder) if TRIAL_MARK not in (row.get("variant") or "")]


def saved_predictions(rows: list[dict], folder: Path | None = None) -> dict[str, dict]:
    """Return the saved predictions of every run that has them: run id -> split -> {"ids", "y", "output"}."""
    return {
        row["run_id"]: load_predictions(row["run_id"], folder)
        for row in rows if predictions_file(row["run_id"], folder).exists()
    }


def settings(rows: list[dict], dataset: str) -> dict[tuple[str, str, str], list[dict]]:
    """Group the runs of a dataset that differ only in the seed: (model, head, variant) -> rows."""
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        if row["dataset"] == dataset:
            groups.setdefault((row["model"], row["head"], row.get("variant") or ""), []).append(row)
    return groups


def mean_metric(rows: list[dict], split: str, metric: str) -> float | None:
    """Return the mean of a metric over some runs, or None when no run has it."""
    values = [(row["metrics"].get(split) or {}).get(metric) for row in rows]
    values = [value for value in values if value is not None]
    return statistics.fmean(values) if values else None


def label(setting: tuple[str, str, str]) -> str:
    """Return a setting as text: "deberta-v3, CORN head"."""
    model, head, variant = setting
    parts = [model] + ([f"{head.upper()} head"] if head != "none" else []) + ([variant] if variant else [])
    return ", ".join(parts)


def best_setting(rows: list[dict], dataset: str, seeds: int = 1, models: set[str] | None = None):
    """Return the setting of a dataset with the best mean QWK on dev, as (setting, rows), or None.

    Only settings with at least `seeds` runs count, and only the models in `models` when it is given.
    """
    candidates = [
        (mean_metric(group, "dev", "qwk"), setting, group)
        for setting, group in settings(rows, dataset).items()
        if len(group) >= seeds and (models is None or setting[0] in models)
    ]
    candidates = [candidate for candidate in candidates if candidate[0] is not None]
    if not candidates:
        return None
    _, setting, group = max(candidates, key=lambda candidate: candidate[0])
    return setting, group


def verdict(spec_id: str, passed: bool | None, evidence: str) -> dict:
    """Return one line of the table. `passed` is None when the store has nothing to judge the specification on."""
    outcome = NOT_MEASURED if passed is None else PASS if passed else FAIL
    return {"id": spec_id, "specification": SPECIFICATIONS[spec_id], "verdict": outcome, "evidence": evidence}


def r1(rows: list[dict], manifest: dict) -> dict:
    """R1: the size of the three training sets, and the extrapolation set."""
    sizes = {name: manifest[name]["train"]["n"] for name in TRAINED if name in manifest}
    with_rows = {row["dataset"] for row in rows}
    extra = [name for name in EXTRAPOLATION if name in manifest]
    passed = (
        len(sizes) == len(TRAINED)
        and all(size > R1_TRAIN_QUESTIONS and name in with_rows for name, size in sizes.items())
        and bool(extra)
    )
    trained = ", ".join(f"{TITLES[name]} {size:,}" for name, size in sizes.items())
    held_out = ", ".join(f"{TITLES[name]} with {manifest[name]['test']['n']:,} test questions" for name in extra)
    return verdict("R1", passed if rows else None, f"Training questions: {trained}. Extrapolation set: {held_out or 'none'}.")


def output_type(row: dict, predictions: dict[str, dict]) -> str | None:
    """Return what a run predicts: a class, an ordinal level, or a number on a continuous scale."""
    if row["head"] == "corn":
        return "ordinal"
    if row["head"] == "ce":
        return "classification"
    saved = predictions.get(row["run_id"])
    if not saved:
        return None
    output = next(iter(saved.values()))["output"]
    return "regression" if output.ndim == 1 else "classification"  # [n] is a score, [n, K] are probabilities


def split_mismatches(rows: list[dict], predictions: dict[str, dict]) -> list[str]:
    """Return the runs whose saved question ids differ from the other runs of the same dataset and split."""
    reference: dict[tuple[str, str], object] = {}
    wrong = []
    for row in rows:
        for split, parts in predictions.get(row["run_id"], {}).items():
            first = reference.setdefault((row["dataset"], split), parts["ids"])
            if len(parts["ids"]) != len(first) or (parts["ids"] != first).any():
                wrong.append(f"{row['run_id']} on {split}")
    return wrong


def r2(rows: list[dict], predictions: dict[str, dict], families: dict[str, str]) -> dict:
    """R2: the model families, the output types, and that every run was scored on the same questions."""
    found_families = {families.get(row["model"], "other") for row in rows}
    found_types = {output_type(row, predictions) for row in rows} - {None}
    wrong = split_mismatches(rows, predictions)
    checked = sum(1 for row in rows if row["run_id"] in predictions)
    passed = set(R2_FAMILIES) <= found_families and set(R2_OUTPUT_TYPES) <= found_types and not wrong and checked > 0
    same = f"All {checked} runs with saved predictions hold the same question ids for each dataset and split."
    if wrong:
        same = f"Different question ids: {', '.join(wrong)}."
    return verdict("R2", passed if rows else None, (
        f"Families: {', '.join(sorted(found_families))}. Output types: {', '.join(sorted(found_types))}. {same}"
    ))


def r3_race(rows: list[dict]) -> dict:
    """R3 on RACE++: the Spearman of the setting that is best on dev."""
    best = best_setting(rows, "race")
    if best is None:
        return verdict("R3 RACE++", None, "No RACE++ run with a dev split.")
    setting, group = best
    spearman = mean_metric(group, "test", "spearman")
    capped = any((row.get("config") or {}).get("limit_train") for row in group)
    limit = R3_RACE_SPEARMAN_CAPPED if capped else R3_RACE_SPEARMAN
    training = "a capped training set" if capped else "the full training set"
    return verdict("R3 RACE++", spearman is not None and spearman >= limit, (
        f"{label(setting)} is the best setting on dev (QWK {mean_metric(group, 'dev', 'qwk'):.3f}), trained on {training}. "
        f"Spearman on test: {spearman:.3f}, against {limit:.2f}. The published {R3_RACE_SPEARMAN:.2f} was reached with "
        "the reading passage in the input; our input holds the question and its options only."
    ))


def arc_setting(rows: list[dict]):
    """Return the ARC setting that R3 and R5 judge: the best mean QWK on dev among the settings with three seeds."""
    return best_setting(rows, "arc", seeds=SEEDS)


def r3_arc(rows: list[dict]) -> dict:
    """R3 on ARC: the mean Spearman over three seeds."""
    best = arc_setting(rows)
    if best is None:
        return verdict("R3 ARC", None, f"No ARC setting has {SEEDS} seeds.")
    setting, group = best
    spearman = mean_metric(group, "test", "spearman")
    return verdict("R3 ARC", spearman >= R3_ARC_SPEARMAN, (
        f"{label(setting)} has the best mean QWK on dev among the settings with {SEEDS} seeds "
        f"({mean_metric(group, 'dev', 'qwk'):.3f}). Mean Spearman on test over {len(group)} seeds: {spearman:.3f}, "
        f"against {R3_ARC_SPEARMAN:.2f}."
    ))


def r3_math(rows: list[dict], predictions: dict[str, dict], families: dict[str, str]) -> dict:
    """R3 on MATH: the best transformer against the best classical model, with a paired bootstrap on test."""
    chosen = {}
    for family in ("transformer", "classical"):
        best = best_setting(rows, "math", models={model for model, name in families.items() if name == family})
        if best is None:
            return verdict("R3 MATH", None, f"No MATH run of a {family} model with a dev split.")
        chosen[family] = max(best[1], key=lambda row: row["metrics"]["dev"]["qwk"])  # one run of the setting
    a, b = (predictions.get(chosen[family]["run_id"], {}).get("test") for family in ("transformer", "classical"))
    if a is None or b is None:
        return verdict("R3 MATH", None, "The test predictions of the two runs are not saved.")
    if len(a["ids"]) != len(b["ids"]) or (a["ids"] != b["ids"]).any():
        return verdict("R3 MATH", None, "The two runs were not scored on the same test questions.")
    result = paired_bootstrap(a["y"], a["output"], b["output"], len(info("math").level_names), metric="qwk")
    return verdict("R3 MATH", result["low"] is not None and result["low"] > 0, (
        f"{chosen['transformer']['run_id']} (QWK {result['a']:.3f}) against {chosen['classical']['run_id']} "
        f"(QWK {result['b']:.3f}) on test. Difference {result['difference']:.3f}, 95% bootstrap interval "
        f"{result['low']:.3f} to {result['high']:.3f}, {result['samples']:,} resamples of the same questions."
    ))


def r4(rows: list[dict]) -> dict:
    """R4: Balanced DRPS of the ordinal head against cross-entropy, backbone by backbone."""
    wins: dict[str, list[str]] = {}
    compared: dict[str, int] = {}
    for dataset in TRAINED:
        groups = settings(rows, dataset)
        for model in sorted({setting[0] for setting in groups}):
            ordinal, plain = groups.get((model, "corn", "")), groups.get((model, "ce", ""))
            if not ordinal or not plain:
                continue
            a, b = mean_metric(ordinal, "test", "balanced_drps"), mean_metric(plain, "test", "balanced_drps")
            if a is None or b is None:
                continue
            compared[model] = compared.get(model, 0) + 1
            wins.setdefault(model, [])
            if a < b:
                wins[model].append(TITLES[dataset])
    if not compared:
        return verdict("R4", None, "No backbone has both a CORN run and a cross-entropy run.")
    order = sorted(compared, key=lambda model: (-len(wins[model]), model))
    counts = ", ".join(f"{model} {len(wins[model])} of {compared[model]} datasets" for model in order)
    met = [model for model in order if len(wins[model]) >= R4_DATASETS]
    missed = [model for model in order if len(wins[model]) < R4_DATASETS]
    summary = f"Met with {' and '.join(met)}" if met else "Met with no backbone"
    summary += f", not with {' and '.join(missed)}." if missed else "."
    return verdict("R4", bool(met), f"CORN has the lower Balanced DRPS on test: {counts}. {summary}")


def r5(rows: list[dict]) -> dict:
    """R5: the spread of Spearman over the three seeds of the ARC setting of R3."""
    best = arc_setting(rows)
    if best is None:
        return verdict("R5", None, f"No ARC setting has {SEEDS} seeds.")
    setting, group = best
    values = [row["metrics"]["test"]["spearman"] for row in group]
    spread = statistics.stdev(values)
    return verdict("R5", spread <= R5_SPEARMAN_SD, (
        f"{label(setting)}, {len(values)} seeds. Standard deviation of Spearman on test: {spread:.3f}, "
        f"against {R5_SPEARMAN_SD:.2f}. Published BERT on the balanced training set: {R5_PUBLISHED_SD}."
    ))


def r6_r7(checks: dict) -> list[dict]:
    """R6 and R7, quoted from the verdicts of scripts/efficiency.py."""
    six, seven = checks["R6"], checks["R7"]
    if six["measured"]:
        longest = six["longest_fine_tune"]
        evidence = (
            f"{six['fine_tunes']} fine-tunes. The longest is {longest['run_id']} at {longest['hours']:.2f} hours, against 1. "
            f"Whole project: {six['gpu_hours']:.2f} GPU-hours, against 20. Details in results/tables/efficiency.md."
        )
    else:
        evidence = "No fine-tune in the store."
    lines = [verdict("R6", six["pass"] if six["measured"] else None, evidence)]
    if seven["measured"]:
        evidence = (
            f"{seven['model']}, one question at a time on a CPU: {seven['fastest']:.3f} to {seven['slowest']:.3f} seconds over "
            f"{len(seven['seconds_per_question'])} saved models, against 1. The slowest measurement decides."
        )
    else:
        evidence = "No saved model was timed on a CPU."
    lines.append(verdict("R7", seven["pass"] if seven["measured"] else None, evidence))
    return lines


def r8(rows: list[dict], folder: Path | None = None) -> dict:
    """R8: every run carries its config, and the tables built from the store equal the files in the store."""
    with_config = sum(1 for row in rows if isinstance(row.get("config"), dict) and row["config"])
    runs = load_runs(folder)
    tables = Path(folder) / "tables" if folder else results_dir() / "tables"
    compared, different = 0, []
    for dataset in sorted(set(runs["dataset"])) if not runs.empty else []:
        for split in TABLE_SPLITS:
            table = leaderboard(runs, dataset, split)
            if table.empty:
                continue
            path = tables / f"{dataset}_{split}.csv"
            compared += 1
            rebuilt = table.to_csv(index=False, lineterminator="\n")
            if not path.exists() or path.read_text(encoding="utf-8").replace("\r\n", "\n") != rebuilt:
                different.append(path.name)
    passed = bool(rows) and with_config == len(rows) and compared > 0 and not different
    same = f"The {compared} leaderboard tables rebuilt from the store are identical to the files in results/tables/."
    if different:
        same = f"Rebuilt tables that differ from the files: {', '.join(different)}."
    return verdict("R8", passed if rows else None, (
        f"{with_config} of {len(rows)} runs carry the config they started from. {same} "
        "One command each: `python -m levellens.results`, `python -m levellens.figures`."
    ))


def verdicts(folder: Path | None = None) -> list[dict]:
    """Return the verdict on every specification, in the order of docs/design.md."""
    efficiency = efficiency_module()
    rows = result_rows(folder)
    predictions = saved_predictions(rows, folder)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if runs_file(folder).exists():
        _, checks = efficiency.build(Path(folder) if folder else results_dir(), write=False)
    else:
        checks = efficiency.verdicts([], {})  # an empty store: nothing was run, so nothing is measured
    families = efficiency.FAMILIES
    return [
        r1(rows, manifest), r2(rows, predictions, families),
        r3_race(rows), r3_arc(rows), r3_math(rows, predictions, families),
        r4(rows), r5(rows), *r6_r7(checks), r8(rows, folder),
    ]


def summary(lines: list[dict]) -> str:
    """Return one sentence: how many of the eight specifications are met."""
    by_spec: dict[str, list[dict]] = {}
    for line in lines:
        by_spec.setdefault(line["id"].split()[0], []).append(line)
    met = [spec for spec, parts in by_spec.items() if all(part["verdict"] == PASS for part in parts)]
    sentence = f"{len(met)} of {len(by_spec)} specifications are met."
    for spec, parts in by_spec.items():
        if spec in met:
            continue
        if len(parts) == 1:
            sentence += f" {spec}: {parts[0]['verdict']}."
            continue
        passed = [part["id"].split(maxsplit=1)[1] for part in parts if part["verdict"] == PASS]
        other = [f"{part['id'].split(maxsplit=1)[1]} ({part['verdict']})" for part in parts if part["verdict"] != PASS]
        sentence += f" {spec} is met on {' and '.join(passed) or 'no dataset'}; not on {' and '.join(other)}."
    return sentence


def report(lines: list[dict]) -> str:
    """Return the Markdown page with the verdicts."""
    page = [
        "# Specifications R1 to R8: pass or fail", "",
        "Built from `runs.jsonl`, the saved predictions and the efficiency reports by `python -m levellens.specs`. "
        "Do not edit by hand. The specifications are frozen in `docs/design.md`; the rules this table applies "
        "where their text leaves a choice are at the top of `levellens/specs.py`.", "",
        f"**{summary(lines)}**", "",
        "| Id | Specification | Verdict | Evidence |", "|---|---|---|---|",
    ]
    page += [
        f"| {line['id']} | {line['specification']} | **{line['verdict']}** | {line['evidence']} |" for line in lines
    ]
    return "\n".join(page)


def build(folder: Path | None = None, write: bool = True) -> list[dict]:
    """Judge every specification on one results store. Write `specs.md` and `specs.csv` into its tables folder."""
    lines = verdicts(folder)
    if write:
        tables = (Path(folder) if folder else results_dir()) / "tables"
        tables.mkdir(parents=True, exist_ok=True)
        with open(tables / "specs.csv", "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(lines)
        (tables / "specs.md").write_text(report(lines) + "\n", encoding="utf-8", newline="\n")
        for name in ("specs.csv", "specs.md"):
            print(shown(tables / name))
    return lines


def main(argv: list[str] | None = None) -> None:
    """Judge every specification and print the verdicts."""
    parser = argparse.ArgumentParser(description="Pass or fail for each specification R1 to R8")
    parser.add_argument("--print", action="store_true", dest="print_only", help="print the verdicts, write nothing")
    args = parser.parse_args(argv)
    lines = build(write=not args.print_only)
    for line in lines:
        print(f"{line['id']:10s} {line['verdict']:13s} {line['evidence']}")
    print(summary(lines))


if __name__ == "__main__":
    main()
