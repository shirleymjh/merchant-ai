---
name: new-product-risk
description: Use when verified evidence combines product publish or audit lifecycle with order, refund, compensation, or ticket evidence to judge risk for recently published or newly active products.
title: 风险分析
lifecyclePhase: post_query_analysis
requiresVerifiedEvidence: true
outputContract: verified_analysis_v1
executionPlacement: AUTO
executionMode: structured_renderer
renderer: verified_evidence
---

# New Product Risk Skill

## Runtime Boundary

- Run only after a Grounded Contract has executed and EvidenceVerifier passed.
- Treat `/input.json` as immutable; never request new metrics, bindings, retrieval, or SQL.
- Never replace or extend a governed metric formula.
- Return `verified_analysis_v1`: observations, semanticDisclosures, derivedFacts,
  hypotheses, recommendations, evidenceRefs, gaps, and executionConfidence.

## Activation Contract

Use this skill when the Grounded Contract and verified evidence include
goods/product lifecycle evidence and at least one after-sales or performance
metric, and the user asks for risk, diagnosis, prioritization, or explanation.

## Evidence Rules

- A product is only called a new product when publish or lifecycle evidence is
  present.
- Do not infer publish time from order time.
- If publish evidence is missing, keep the answer as product risk, not new
  product risk.
- Every high-risk label must cite order/refund/compensation/ticket evidence.

## Procedure

1. Identify products with lifecycle evidence.
2. Compare performance and after-sales indicators.
3. Mark high-risk new products only when lifecycle and risk evidence both
   exist.
4. Return priority items, evidence, caveats, and next actions.
