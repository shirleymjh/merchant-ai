---
name: ratio-analysis
description: Use when the answer depends on a derived ratio, percentage, or population subset calculation. The skill must expose numerator, denominator, formula, and coverage gaps.
title: 派生指标分析
lifecyclePhase: post_query_analysis
requiresVerifiedEvidence: true
outputContract: verified_analysis_v1
executionPlacement: AUTO
executionMode: structured_renderer
renderer: verified_evidence
---

# Ratio Analysis Skill

## Runtime Boundary

- Run only after a Grounded Contract has executed and EvidenceVerifier passed.
- Treat `/input.json` as immutable; never request new metrics, bindings, retrieval, or SQL.
- Never replace or extend a governed metric formula.
- Return `verified_analysis_v1`: observations, semanticDisclosures, derivedFacts,
  hypotheses, recommendations, evidenceRefs, gaps, and executionConfidence.

## Activation Contract

Use this skill when the Grounded Contract and verified evidence contain a
derived ratio/percentage metric, share, or 占比 calculation.

## Evidence Rules

- Always name the numerator, denominator, formula, and computed metric.
- Do not treat a missing numerator or denominator as zero.
- If the base population is scope-constrained, state the scope source.
- If the numerator and denominator use different grains, call out that caveat.

## Procedure

1. Identify base population and event/subset population.
2. Report numerator evidence, denominator evidence, and formula.
3. Interpret the ratio only after confirming both sides are covered.
4. List evidence gaps separately.
