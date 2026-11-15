# Final-model error audit

中文版：[audit.zh-CN.md](audit.zh-CN.md)

**Owner:** Wenyue. **Status on 15 November 2026:** the 15 final-model cases have arrived and their provenance has been verified. The table contains **provisional judgements**, with one reason per case. Wenyue's case-by-case human confirmation is pending. The proposed counts are **7 model error, 0 label noise and 8 unclear**. These are not yet completed human-audit counts and must not be presented as such.

## Frozen cases and provenance

The public [case list](../results/audit/worst_errors.csv) now contains five full-test errors per dataset from the three runs below, replacing the earlier ten BiLSTM cases. The runs match the best development-set quadratic weighted kappa (QWK) among the saved runs. No test result was used to select a model or tune it.

Cases are ordered by descending absolute ordinal error, then descending confidence in the predicted level rounded to four decimals, then item ID, matching `levellens/explain/errors.py`. All 15 IDs, source labels, predictions, probabilities, confidence values and top-five positions were reproduced. The local test splits match the frozen manifest: RACE++ 5,642 items, ARC 3,530 and MATH 5,000. The supplied full texts match those splits and remain private.

| Dataset | Frozen run ID | Prediction-file SHA-256 |
|---|---|---|
| RACE++ | `race-deberta-v3-corn-s0` | `552a88e051d822dd820cf28c0c09fd5199aef3d15b49bed3610b65a8bc9c9e52` |
| ARC | `arc-distilbert-corn-optuna-s0` | `73ff2c5693892798be0e0e71545bb4115b7ba9c1d970c5a76b63ec9ac3b53072` |
| MATH | `math-deberta-v3-corn-s0` | `210c8c5fa73cceb0392a31e349296515ac06c1aa1f40eeae418d15c6d9f087e6` |

## Decision rules

| Category | Evidence needed |
|---|---|
| model error | The source label is plausible for its source exam or level, the model misses it, and no concrete data defect explains the case; record the reasoning demand or relevant cue. |
| label noise | A concrete source-label inconsistency, source metadata error or corrupted item supports a label concern; apparent ease alone is insufficient, and original test labels are retained. |
| unclear | The source level cannot be adjudicated reliably, necessary context is unavailable, or plausible explanations conflict; record the missing evidence. |

These categories are operational judgements, not causal proof. Labels are **target levels**, not difficulty calibrated from student responses. Omitted context and markup are secondary flags, not automatic evidence of label noise. In particular, the short solutions of MATH-2 and MATH-4 motivate uncertainty but do not establish an incorrect source rating.

Wenyue must read and confirm each full case before the human-reviewer fields are filled. Record the actual reviewer and date. A second member should check disputed cases if available. Do not name someone who has not reviewed the case.

## Fifteen-case worksheet

Categories below are provisional. “Pending” in the last column is a workflow state, not a fourth category. Secondary flags follow the table.

| Slot | Dataset | Item ID | Run ID | True / predicted level | Provisional category | Reason / evidence | Human reviewer / date |
|---|---|---|---|---|---|---|---|
| RACE-1 | RACE++ | `race-test-college696-q1` | `race-deberta-v3-corn-s0` | college / middle school | unclear | The question retrieves a weather event from an omitted passage, so its short wording cannot settle the college source level or explain the middle-school prediction. | Pending |
| RACE-2 | RACE++ | `race-test-college696-q0` | `race-deberta-v3-corn-s0` | college / middle school | unclear | The reason for a habitual conversation topic comes from an omitted passage, and simple wording alone does not establish that the college source label is wrong. | Pending |
| RACE-3 | RACE++ | `race-test-college696-q3` | `race-deberta-v3-corn-s0` | college / middle school | unclear | The everyday precaution is interpreted through an omitted passage, leaving insufficient evidence to separate missing context from a questionable college source level. | Pending |
| RACE-4 | RACE++ | `race-test-middle3379-q0` | `race-deberta-v3-corn-s0` | middle school / college | unclear | The question asks for a passage-specific explanation of a metaphor, but the model receives no passage and the supplied metadata alone cannot explain its college prediction. | Pending |
| RACE-5 | RACE++ | `race-test-college696-q2` | `race-deberta-v3-corn-s0` | college / middle school | unclear | A phrase is interpreted within an omitted passage, while its familiar vocabulary is insufficient evidence for changing the recorded college source level. | Pending |
| ARC-1 | ARC | `TIMSS_1995_8_N2` | `arc-distilbert-corn-optuna-s0` | grade 8 / grade 3 | model error | The complete item compares nutrient coverage across meals and the raw TIMSS record specifies grade 8, with no observed data defect supporting the grade-3 prediction. | Pending |
| ARC-2 | ARC | `Mercury_SC_LBS10946` | `arc-distilbert-corn-optuna-s0` | grade 4 / grade 9 | model error | The raw ARC item records grade 4 and presents a complete single-rule inheritance question, so technical vocabulary alone does not justify predicting a grade-9 source. | Pending |
| ARC-3 | ARC | `NCEOGA_2013_8_36` | `arc-distilbert-corn-optuna-s0` | grade 8 / grade 3 | model error | The original North Carolina assessment record specifies grade 8 and the energy-source question is complete, so elementary content alone does not invalidate its source grade. | Pending |
| ARC-4 | ARC | `MCAS_2004_9_6-v1` | `arc-distilbert-corn-optuna-s0` | grade 9 / grade 4 | model error | The complete MCAS item requires identifying a wave example and its raw record specifies grade 9, with no observed source-label conflict to support grade 4. | Pending |
| ARC-5 | ARC | `WASL_2005_8_12` | `arc-distilbert-corn-optuna-s0` | grade 8 / grade 3 | model error | The complete WASL item requires choosing a relevant before-and-after observation and its raw grade-8 record provides no evidence for the grade-3 prediction. | Pending |
| MATH-1 | MATH | `math-test-prealgebra-0581` | `math-deberta-v3-corn-s0` | Level 5 / Level 1 | unclear | The table is supplied only as Asymptote drawing code, so the Level-5 versus Level-1 disagreement cannot be separated reliably from the input representation. | Pending |
| MATH-2 | MATH | `math-test-prealgebra-0784` | `math-deberta-v3-corn-s0` | Level 5 / Level 1 | unclear | The supplied solution uses a short product-rule count, but no item-specific rating rationale establishes whether the recorded Level 5 or the predicted Level 1 is the more defensible target. | Pending |
| MATH-3 | MATH | `math-test-number_theory-0044` | `math-deberta-v3-corn-s0` | Level 5 / Level 1 | model error | Coordinating leap years, interval endpoints and weekday arithmetic makes the source target plausible, while the Level-1 prediction misses this multistep demand and no concrete data defect was found. | Pending |
| MATH-4 | MATH | `math-test-algebra-1029` | `math-deberta-v3-corn-s0` | Level 5 / Level 1 | unclear | The supplied solution is a short symbolic rate substitution, but the absent item-specific rating rationale prevents adjudicating the Level-5 versus Level-1 disagreement. | Pending |
| MATH-5 | MATH | `math-test-number_theory-0134` | `math-deberta-v3-corn-s0` | Level 5 / Level 1 | model error | The complete problem requires combining powers and reducing residues modulo a prime, making Level 5 plausible and the Level-1 prediction an unsupported underestimate. | Pending |

**Secondary flags and scope.** All five RACE++ cases omit the passage from model input. RACE-1, RACE-2, RACE-3 and RACE-5 share passage `college696`; the five selected questions represent only two passages. RACE-4 also contains a source spacing artefact. MATH-1 embeds an unrendered table as Asymptote code. MATH-2 and MATH-4 lack an item-specific rating rationale. No selected case has an identical model input elsewhere in the primary train, dev and test splits. These checks do not exclude near duplicates or prove that the source labels are correct.

## Proposed counts, awaiting human confirmation

| Dataset | Model error | Label noise | Unclear | Cases |
|---|---|---|---|---|
| RACE++ | 0 | 0 | 5 | 5 |
| ARC | 5 | 0 | 0 | 5 |
| MATH | 2 | 0 | 3 | 5 |
| Total | 7 | 0 | 8 | 15 |

Human-confirmed cases: **0 of 15**. Cases awaiting human confirmation: **15**. The zero is for supported label-noise judgements in this proposed classification, not proof that all labels are clean. This severity-selected set cannot estimate population error or label-noise rates.

## Three proposed slide examples

Use these paraphrases only after human confirmation. Do not publish question text, passages or solutions. Keep the item ID and run ID in speaker notes.

| Slot | Source target / prediction | Proposed decision | Slide wording |
|---|---|---|---|
| RACE-1 | college / middle school | unclear | Passage-dependent item with its passage omitted; source stage cannot be settled from the short input. |
| ARC-3 | grade 8 / grade 3 | model error | A basic science fact can still come from a grade-8 exam; target grade and perceived ease differ. |
| MATH-5 | Level 5 / Level 1 | model error | The model underestimates a multistep modular-arithmetic problem. |

Caption: Five worst full-test errors per frozen model, selected by ordinal gap and confidence; proposed case judgements, pending human confirmation. Counts describe these 15 selected questions only.

## Completion check

- [x] All 15 cases match frozen test data and the supplied prediction hashes.
- [x] Top-five selection is reproduced for all three frozen runs.
- [x] Fifteen provisional reasons, category counts and three examples are prepared in both languages.
- [ ] Wenyue has read and confirmed each case; actual reviewer and date are recorded.
- [ ] Final counts replace the provisional status, and slides 6–9 and narration use the confirmed result.
- [ ] `done B3` is posted to issue #2 after human confirmation.
