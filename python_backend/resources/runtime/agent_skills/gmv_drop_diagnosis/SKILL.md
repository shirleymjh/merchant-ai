---
name: gmv-drop-diagnosis
description: Use when a merchant asks why GMV, sales amount, paid orders, or order volume dropped or moved abnormally. The skill is a fixed merchant-operation SOP and must only use verified evidence rows.
title: 指标下降诊断
executionMode: structured_renderer
renderer: verified_evidence
---

# GMV Drop Diagnosis Skill

This skill turns verified merchant BI evidence into a constrained diagnosis for
GMV, sales amount, paid orders, or order volume decline.

## Activation Contract

Use this skill only when the Core has a Grounded Contract plus verified evidence
and the question asks for trend,
anomaly, diagnosis, comparison, or attribution around GMV, sales amount, or
orders.

Do not use it for plain count lookups, product lists, or refund-only questions.

## Evidence Rules

- Only use verified rows, metric disclosures, evidence gaps, and table labels
  passed in the skill input artifact.
- Do not invent promotions, traffic changes, product events, or causes.
- Every conclusion must reference at least one data point from the input.
- If the evidence only shows the metric changed, say that the cause still needs
  product/channel/date breakdown evidence.

## Procedure

1. Judge the direction and magnitude of GMV/order change.
2. Locate whether the change is concentrated by date, product, category,
   channel, or another available dimension.
3. Split the path into available drivers, such as order count, paid amount,
   customer unit price, refund amount, or coupon impact.
4. Bind each inference to specific evidence rows.
5. Submit a concise diagnosis, caveats, and merchant actions.
