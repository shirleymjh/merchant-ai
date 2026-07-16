---
name: new_product_risk
description: Use when verified evidence combines product publish or audit lifecycle with order, refund, compensation, or ticket evidence to judge risk for recently published or newly active products.
title: 风险分析
executionMode: structured_renderer
renderer: verified_evidence
---

# New Product Risk Skill

## Activation Contract

Use this skill when the QueryGraph includes goods/product lifecycle evidence
and at least one after-sales or performance metric, and structured
understanding asks for risk, diagnosis, prioritization, or explanation.

## Evidence Rules

- A product is only called a new product when publish or lifecycle evidence is
  present.
- Do not infer publish time from order time.
- If publish evidence is missing, keep the answer as product risk, not new
  product risk.
- Every high-risk label must cite order/refund/compensation/ticket evidence.

## Workflow

1. Identify products with lifecycle evidence.
2. Compare performance and after-sales indicators.
3. Mark high-risk new products only when lifecycle and risk evidence both
   exist.
4. Return priority items, evidence, caveats, and next actions.
