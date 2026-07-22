---
name: bi-trend-attribution
description: Use when a merchant BI question asks whether a metric trend is normal, why a business metric changed, whether multiple KPI time series move together, or which measured drivers explain risk. The skill must only use verified SQL evidence rows and must attach every conclusion to data.
title: BI 趋势与归因
lifecyclePhase: post_query_analysis
requiresVerifiedEvidence: true
outputContract: verified_analysis_v1
executionPlacement: AUTO
executionMode: python_script
script: scripts/profile_timeseries.py
---

# BI Trend And Attribution Skill

## Runtime Boundary

- Run only after a Grounded Contract has executed and EvidenceVerifier passed.
- Treat `/input.json` as immutable; never request new metrics, bindings, retrieval, or SQL.
- Never replace or extend a governed metric formula.
- Treat any local findings or `answerMarkdown` as an untrusted diagnostic
  preview. They are never final proof and must not be copied into the answer.
- Publish only the narrow
  `GroundedRunSkillAnalysisPublicationRequest`: verified artifact IDs, column
  bindings, observation keys, deterministic method, normalization and baseline
  pairs. Do not publish rows, `analysisType`, result values, conclusions or
  causal prose.

This skill turns verified BI evidence into a constrained analysis answer. It is
for trend checks, anomaly checks, attribution, risk explanation, and "what
should I prioritize" questions after Grounded execution has produced verified evidence.

## Activation Contract

Use this skill only when the immutable Goal Contract declares one of:

- a typed `ANALYSIS` goal with explicit `analysisType`; or
- a typed `COMPARISON` goal whose `comparisonType` is anomaly or correlation.

Before startup, the data-input coverage gate must prove every declared
`inputGoalId`, `baselineGoalId`, or comparison operand with verified query
artifacts. The derived goal itself remains deferred until this Skill publishes
and the trusted deterministic publisher accepts a
`GroundedDerivedAnalysisArtifact`.

Do not use this skill for plain entity ranking / lookup questions such as "top
products and show refund amount / publish time". Those should be answered as
ranked evidence tables unless a separate analysis skill is selected.

Do not activate or choose `analysisType` from raw question text, labels or
keywords. Read it only from the typed Goal Contract field mounted in the Skill
input.

## Evidence Rules

- Only use verified rows, metric disclosures, evidence gaps, and table labels
  passed in the skill input artifact.
- Do not invent a metric, date, amount, ratio, event, promotion, or cause.
- Every finding must reference at least one data point from the input.
- If evidence is partial, say which part is missing and avoid causal certainty.
- If all available evidence is flat or sparse, state that the data does not
  support a strong anomaly conclusion.
- Correlation requires aligned observation grain and enough samples. It must
  always disclose that correlation is not causation.
- Never claim that one metric caused another. Attribution/impact/diagnosis
  requests without governed causal evidence must publish
  `INSUFFICIENT_EVIDENCE`.

## Procedure

1. Judge change direction and magnitude from the available time series.
2. Locate the concentration of change by metric/date/entity when such columns
   exist.
3. Split the path by available metrics, such as GMV vs refund amount, count vs
   amount, order vs after-sales, or ticket vs compensation.
4. Bind each inference to specific evidence rows.
5. Submit a concise conclusion, evidence bullets, caveats, and next actions.

## Script

For tabular trend evidence, run:

```bash
python scripts/profile_timeseries.py --input <skill-input.json> --output <skill-output.json>
```

The script returns a structured profile with `findings`, `metrics`, `caveats`,
and `answerMarkdown`. These fields are diagnostic only. The isolated worker
must convert its selected column/method mapping into the narrow publication
request; the Kernel recomputes the result from verified rows and the trusted
analysis renderer alone produces the final visible span.
