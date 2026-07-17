---
name: bi-trend-attribution
description: Use when a merchant BI question asks whether a metric trend is normal, why a business metric changed, whether multiple KPI time series move together, or which measured drivers explain risk. The skill must only use verified SQL evidence rows and must attach every conclusion to data.
title: BI 趋势与归因
lifecyclePhase: post_query_analysis
requiresVerifiedEvidence: true
outputContract: verified_analysis_v1
executionMode: python_script
script: scripts/profile_timeseries.py
---

# BI Trend And Attribution Skill

## Runtime Boundary

- Run only after a Grounded Contract has executed and EvidenceVerifier passed.
- Treat `/input.json` as immutable; never request new metrics, bindings, retrieval, or SQL.
- Never replace or extend a governed metric formula.
- Return `verified_analysis_v1`: observations, semanticDisclosures, derivedFacts,
  hypotheses, recommendations, evidenceRefs, gaps, and executionConfidence.

This skill turns verified BI evidence into a constrained analysis answer. It is
for trend checks, anomaly checks, attribution, risk explanation, and "what
should I prioritize" questions after Grounded execution has produced verified evidence.

## Activation Contract

Use this skill only when the Core has a grounded analysis request and verified evidence declares:

- `analysisIntent` is `trend_check`, `anomaly_check`, `diagnosis`, or `comparison`, or
- `requiresExplanation` is true and the required evidence intents describe trend, anomaly, diagnosis, attribution, or comparison evidence.

Do not use this skill for plain entity ranking / lookup questions such as "top
products and show refund amount / publish time". Those should be answered as
ranked evidence tables unless a separate analysis skill is selected.

Do not activate from raw question keywords. The Core should decide from the
Grounded Contract, verified evidence, and evidence gaps.

## Evidence Rules

- Only use verified rows, metric disclosures, evidence gaps, and table labels
  passed in the skill input artifact.
- Do not invent a metric, date, amount, ratio, event, promotion, or cause.
- Every finding must reference at least one data point from the input.
- If evidence is partial, say which part is missing and avoid causal certainty.
- If all available evidence is flat or sparse, state that the data does not
  support a strong anomaly conclusion.

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
and `answerMarkdown`. The Core may use `answerMarkdown` directly when
LLM answer synthesis is slow or unavailable.
