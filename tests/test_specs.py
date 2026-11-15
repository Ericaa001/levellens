"""The verdicts on R1 to R8: the choices `levellens/specs.py` makes, checked on small made-up stores."""

import numpy as np
import pandas as pd

from levellens import specs
from levellens.results import ROW_FIELDS, append_run, leaderboard, load_runs


def run(dataset, model, head, seed, dev_qwk, spearman, drps=0.5, variant="", limit_train=None):
    """Return a result row with the fields the verdicts read."""
    name = "-".join(part for part in (dataset, model, head, variant) if part)
    metrics = {"n": 10, "qwk": dev_qwk, "spearman": spearman, "balanced_drps": drps}
    row = dict.fromkeys(ROW_FIELDS)
    row.update(
        run_id=f"{name}-s{seed}", dataset=dataset, model=model, head=head, variant=variant, seed=seed,
        config={"limit_train": limit_train}, metrics={"dev": dict(metrics), "test": dict(metrics)},
        train_seconds=1.0, predict_seconds=1.0, device="cpu", n_params=1, owner="A", tier=1,
    )
    return row


def three_seeds(model, head, dev_qwk, spearmans):
    """Return three ARC runs of one setting that differ in the seed."""
    return [run("arc", model, head, seed, dev_qwk, spearman) for seed, spearman in enumerate(spearmans)]


def test_arc_setting_needs_three_seeds():
    """A tuned run with one seed can be the best on dev and still is not the setting R3 and R5 judge."""
    rows = [run("arc", "distilbert", "corn", 0, 0.90, 0.10, variant="optuna")]
    rows += three_seeds("deberta-v3", "corn", 0.60, [0.62, 0.63, 0.64])
    setting, group = specs.arc_setting(rows)
    assert setting == ("deberta-v3", "corn", "") and len(group) == 3
    assert specs.r3_arc(rows)["verdict"] == specs.PASS
    assert specs.r5(rows)["verdict"] == specs.PASS


def test_arc_without_three_seeds_is_not_measured():
    rows = [run("arc", "distilbert", "corn", 0, 0.9, 0.9)]
    assert specs.r3_arc(rows)["verdict"] == specs.NOT_MEASURED
    assert specs.r5(rows)["verdict"] == specs.NOT_MEASURED


def test_r5_fails_when_the_seeds_spread():
    rows = three_seeds("distilbert", "ce", 0.6, [0.50, 0.60, 0.70])  # standard deviation 0.10
    assert specs.r3_arc(rows)["verdict"] == specs.PASS
    assert specs.r5(rows)["verdict"] == specs.FAIL


def test_r3_race_is_read_on_the_setting_chosen_on_dev():
    """The setting with the higher Spearman on test does not count when another one is better on dev."""
    rows = [run("race", "deberta-v3", "corn", 0, 0.70, 0.70), run("race", "deberta-v3", "ce", 0, 0.69, 0.80)]
    line = specs.r3_race(rows)
    assert line["verdict"] == specs.FAIL and "0.700" in line["evidence"]


def test_r3_race_capped_training_set_has_the_lower_limit():
    assert specs.r3_race([run("race", "distilbert", "ce", 0, 0.7, 0.78)])["verdict"] == specs.FAIL
    assert specs.r3_race([run("race", "distilbert", "ce", 0, 0.7, 0.78, limit_train=30000)])["verdict"] == specs.PASS


def test_r4_counts_the_datasets_backbone_by_backbone():
    rows = []
    for dataset, corn_a, corn_b in (("race", 0.4, 0.6), ("arc", 0.4, 0.6), ("math", 0.6, 0.6)):
        rows += [run(dataset, "a", "ce", 0, 0.5, 0.5, drps=0.5), run(dataset, "a", "corn", 0, 0.5, 0.5, drps=corn_a)]
        rows += [run(dataset, "b", "ce", 0, 0.5, 0.5, drps=0.5), run(dataset, "b", "corn", 0, 0.5, 0.5, drps=corn_b)]
    line = specs.r4(rows)
    assert line["verdict"] == specs.PASS
    assert "a 2 of 3 datasets" in line["evidence"] and "b 0 of 3 datasets" in line["evidence"]
    assert "Met with a, not with b." in line["evidence"]
    only_b = [row for row in rows if row["model"] == "b"]
    assert specs.r4(only_b)["verdict"] == specs.FAIL
    assert specs.r4([row for row in rows if row["head"] == "ce"])["verdict"] == specs.NOT_MEASURED


def test_r2_reads_the_output_type_and_checks_the_question_ids():
    families = {"tfidf": "classical", "bilstm": "recurrent", "bert": "transformer", "llm": "open LLM", "r2de": "classical"}
    rows = [
        run("arc", "tfidf", "none", 0, 0.5, 0.5), run("arc", "r2de", "none", 0, 0.5, 0.5),
        run("arc", "bilstm", "corn", 0, 0.5, 0.5), run("arc", "bert", "ce", 0, 0.5, 0.5), run("arc", "llm", "none", 0, 0.5, 0.5),
    ]
    ids = np.array(["q1", "q2", "q3"])
    saved = {row["run_id"]: {"test": {"ids": ids, "output": np.zeros((3, 7))}} for row in rows}
    saved["arc-r2de-none-s0"]["test"]["output"] = np.zeros(3)  # one score per question: a regression
    assert specs.output_type(rows[1], saved) == "regression" and specs.output_type(rows[0], saved) == "classification"
    assert specs.r2(rows, saved, families)["verdict"] == specs.PASS

    saved["arc-bert-ce-s0"]["test"]["ids"] = np.array(["q1", "q2", "q9"])
    line = specs.r2(rows, saved, families)
    assert line["verdict"] == specs.FAIL and "arc-bert-ce-s0 on test" in line["evidence"]


def test_trial_runs_are_left_out(tmp_path):
    """A run whose variant holds "smoke" stays in the store and is in no verdict and no leaderboard."""
    append_run(run("arc", "distilbert", "ce", 0, 0.99, 0.99, variant="smoke-n400"), tmp_path)
    append_run(run("arc", "distilbert", "ce", 0, 0.60, 0.60), tmp_path)
    assert [row["run_id"] for row in specs.result_rows(tmp_path)] == ["arc-distilbert-ce-s0"]
    runs = load_runs(tmp_path)
    assert len(runs) == 2
    assert list(leaderboard(runs, "arc", "test")["Variant"].fillna("")) == [""]


def test_summary_counts_r3_once():
    lines = [
        {"id": "R1", "verdict": specs.PASS}, {"id": "R3 RACE++", "verdict": specs.FAIL},
        {"id": "R3 ARC", "verdict": specs.PASS}, {"id": "R3 MATH", "verdict": specs.PASS},
    ]
    assert specs.summary(lines) == "1 of 2 specifications are met. R3 is met on ARC and MATH; not on RACE++ (fail)."


def test_an_empty_store_measures_nothing(tmp_path):
    lines = specs.verdicts(tmp_path)
    assert [line["id"] for line in lines] == list(specs.SPECIFICATIONS)
    assert {line["verdict"] for line in lines} == {specs.NOT_MEASURED}


def test_the_project_store_gets_a_verdict_on_every_specification():
    lines = specs.verdicts(specs.REPO_ROOT / "results")  # the committed store, not the empty one of the test session
    assert [line["id"] for line in lines] == list(specs.SPECIFICATIONS)
    assert all(line["verdict"] in (specs.PASS, specs.FAIL, specs.NOT_MEASURED) for line in lines)
    assert set(pd.DataFrame(lines).columns) == set(specs.COLUMNS)
