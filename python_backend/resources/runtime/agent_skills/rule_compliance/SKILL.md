---
name: rule_compliance
description: Use when a merchant BI answer combines platform rules or policy guidance with measured evidence. The skill must keep recalled rule evidence separate from SQL facts.
---

# Rule Compliance Skill

## Activation Contract

Use this skill when `plan_requires_rule_evidence(plan)` is true and verified
data evidence is also present.

Rule-only questions should use the RuleAnswerAgent path. This skill is for
rule + data or rule + data + analysis answers.

## Evidence Rules

- Rule statements must come from recalled rule evidence.
- Data statements must come from verified SQL/compute evidence.
- Do not infer policy violation solely from a high metric value unless the rule
  evidence explicitly defines that threshold or condition.
- If the rule evidence is partial, say that the compliance conclusion is
  limited by the recalled rule coverage.

## Workflow

1. Summarize applicable rule evidence.
2. Summarize measured data evidence.
3. Compare them only where rule evidence gives a basis.
4. Return compliance caveats and operational follow-ups.
