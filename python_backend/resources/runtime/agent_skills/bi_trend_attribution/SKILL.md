---
name: bi_trend_attribution
description: Use when a merchant BI question asks whether a metric trend is normal, why a business metric changed, whether multiple KPI time series move together, or which measured drivers explain risk. The skill must only use verified SQL evidence rows and must attach every conclusion to data.
---

# BI Trend And Attribution Skill

This skill turns verified BI evidence into a constrained analysis answer. It is
for trend checks, anomaly checks, attribution, risk explanation, and "what
should I prioritize" questions after QueryGraph execution has produced evidence.

## Activation Contract

Use this skill only when the Planner-produced `questionUnderstanding` declares:

- `analysisIntent` is not `none`, or
- `requiresExplanation` is true.

Do not activate from raw question keywords. The Lead/Answer agent should decide
from structured question understanding.

## Evidence Rules

- Only use verified rows, metric disclosures, evidence gaps, and table labels
  passed in the skill input artifact.
- Do not invent a metric, date, amount, ratio, event, promotion, or cause.
- Every finding must reference at least one data point from the input.
- If evidence is partial, say which part is missing and avoid causal certainty.
- If all available evidence is flat or sparse, state that the data does not
  support a strong anomaly conclusion.

## Workflow

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
and `answerMarkdown`. The Answer agent may use `answerMarkdown` directly when
LLM answer synthesis is slow or unavailable.
