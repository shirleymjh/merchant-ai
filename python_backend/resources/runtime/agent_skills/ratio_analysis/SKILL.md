---
name: ratio_analysis
description: Use when the answer depends on a derived ratio, percentage, or population subset calculation. The skill must expose numerator, denominator, formula, and coverage gaps.
---

# Ratio Analysis Skill

## Activation Contract

Use this skill when a QueryGraph has a derived ratio/percentage metric, or
`questionUnderstanding.calculationIntents` contains a ratio, percentage, share,
or占比 style calculation.

## Evidence Rules

- Always name the numerator, denominator, formula, and computed metric.
- Do not treat a missing numerator or denominator as zero.
- If the base population is scope-constrained, state the scope source.
- If the numerator and denominator use different grains, call out that caveat.

## Workflow

1. Identify base population and event/subset population.
2. Report numerator evidence, denominator evidence, and formula.
3. Interpret the ratio only after confirming both sides are covered.
4. List evidence gaps separately.
