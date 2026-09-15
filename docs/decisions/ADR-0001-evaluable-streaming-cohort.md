# ADR-0001 — Evaluable streaming cohort for per-user drift latency

- **Statut** : accepté
- **Date** : 2026-08-18

## Contexte

The research loop measures, *per user and in interaction units*, detection latency,
adaptation latency and recovery latency around a preference shift. This requires each
evaluated user to have a test-period sequence long enough to (a) fill a short-term
window (5–20 interactions) and (b) exhibit shift → detection → adaptation → recovery
over many subsequent steps.

Phase 2 validation showed the `min 20 interactions` floor is enforced **globally**,
but the global temporal split slices each user's history by time. In the **test**
split the median user has only **6** interactions (p25=3, p95=23, max=149). Most test
users therefore cannot support a per-user latency measurement at all.

## Options envisagées

| Option | Avantages | Inconvénients |
| --- | --- | --- |
| Evaluate latency on *all* test users | no selection | undefined for ~most users; latency dominated by users with 3–6 interactions where short+long windows don't fit |
| Re-split per-user (local 70/10/20 per user) | every user has test history | breaks the *global* chronological stream and the shared drift-onset schedule; changes the frozen data |
| **Evaluable cohort: test users with ≥ `N_min_test` test-period interactions** | latency is well-defined; long-term rep available (96.5% seen in train) | latency reported on a subpopulation, not all users |

## Décision

Adopt the **evaluable cohort**. The streaming per-user drift/latency evaluation runs on
test users whose test-period interaction count ≥ `N_min_test`. Their *full* cross-split
history feeds the long-term representation `L_u(t)`. `N_min_test` is a config parameter
(candidate default ≈ 20; larger for latency-focused analyses), tuned/justified on the
**validation** split only, never on test.

Aggregate next-item quality (NDCG@10 etc.) is additionally reported on the **full** test
set so the cohort restriction is transparent and does not inflate headline quality.

## Conséquences

- Adds a cohort-selection step (config-driven, seed-logged) before streaming evaluation;
  the cohort size and its activity profile are reported.
- Latency metrics (H1–H3) are claims about the evaluable subpopulation — stated as such
  in the thesis, in the threats-to-validity section.
- Explicitly **excludes** cold-start and low-activity test users from *latency* claims
  (not from aggregate quality). This is a scoping choice consistent with the brief's
  requirement of ≥20 prior interactions, not an oversight.
- Pairs with the natural-vs-synthetic distinction: the same cohort is used for controlled
  injection (Phase 7) so ground-truth onsets fall on users who can actually recover.
