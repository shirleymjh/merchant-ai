---
name: merchant_daily_briefing
description: Use when a merchant asks for a daily/weekly operating briefing, store health summary, or what to prioritize today. The skill summarizes verified business signals into action priorities.
---

# Merchant Daily Briefing Skill

This skill turns verified merchant BI evidence into a concise operating
briefing.

## Activation Contract

Use this skill only when the Planner-produced `questionUnderstanding` declares
a reusable or fixed analysis workflow and the question asks for a store health
summary, daily report, weekly report, operating briefing, or priority list.

Do not use it for a single metric lookup unless the user asks for diagnosis or
prioritization.

## Evidence Rules

- Only use verified rows, metric disclosures, evidence gaps, and table labels
  passed in the skill input artifact.
- Do not mark a missing topic as normal.
- Separate facts, risks, and suggested actions.
- Every priority must be tied to a data row or an explicit evidence gap.

## Workflow

1. Summarize trade, refund/after-sales, customer service/compensation, product,
   and fulfillment signals that are present in the evidence.
2. Rank the top merchant actions by impact and confidence.
3. Point out missing topics that prevent a full health judgment.
4. Bind each action to specific evidence rows.
5. Submit a short briefing with priorities, caveats, and next drill-downs.
