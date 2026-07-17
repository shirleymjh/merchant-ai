---
name: refund-rate-diagnosis
description: Use when a merchant asks why refund rate, refund amount, after-sales, or return pressure rose. The skill follows a fixed refund diagnosis SOP and must disclose numerator/denominator evidence.
title: 指标变化诊断
executionMode: structured_renderer
renderer: verified_evidence
---

# Refund Rate Diagnosis Skill

This skill turns verified merchant BI evidence into a constrained refund or
after-sales diagnosis.

## Activation Contract

Use this skill only when the Core has a Grounded Contract plus verified evidence
and the question asks for refund-rate,
refund amount, return, after-sales, or refund anomaly diagnosis.

Do not use it for platform rule-only questions or plain refund detail lookup.

## Evidence Rules

- Only use verified rows, metric disclosures, evidence gaps, and table labels
  passed in the skill input artifact.
- Refund rate conclusions must disclose both numerator and denominator evidence
  when available.
- Do not treat missing order count or missing refund count as zero.
- If product, reason, ticket, or compensation evidence is missing, present it as
  a follow-up gap rather than a cause.

## Procedure

1. Confirm whether refund rate, refund count, or refund amount changed.
2. Check whether the movement is driven by numerator growth, denominator drop,
   or both.
3. Locate concentration by product, category, date, reason, ticket, or
   compensation evidence when those dimensions exist.
4. Bind every attribution to specific evidence rows.
5. Submit priority actions and disclose unresolved evidence gaps.
