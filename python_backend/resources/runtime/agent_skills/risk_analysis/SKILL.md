---
name: risk-analysis
description: Use when verified BI evidence must rank or explain merchant operational risk across products, orders, refunds, compensation, tickets, coupons, or fulfillment. The skill must only use verified evidence rows and must separate observed facts from risk hypotheses.
title: 风险分析
executionMode: structured_renderer
renderer: verified_evidence
---

# Risk Analysis Skill

## Activation Contract

Use this skill when the Grounded Contract and verified evidence require risk
ranking, diagnosis, or anomaly analysis across two or more BI evidence domains.

Do not activate from raw question keywords alone. The Core must select this
skill from grounded bindings, verified evidence, and evidence gaps.

## Evidence Rules

- Only use verified SQL/compute rows, metric disclosures, and evidence gaps.
- Do not invent a cause, loss amount, rate, status, or item not present in the
  evidence.
- Every risk statement must cite the metric or row that supports it.
- If an important evidence branch failed or returned zero rows, keep it as a
  caveat rather than treating it as low risk.

## Procedure

1. Identify risk dimensions available in evidence: volume, amount, rate,
   compensation, tickets, coupon spend, fulfillment, product lifecycle.
2. Rank entities by observed severity, not by missing evidence.
3. Separate high-confidence facts from possible causes.
4. Return priority items, supporting evidence, caveats, and next actions.
