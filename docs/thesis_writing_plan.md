# Thesis writing plan — expanding the draft to 40–80 pages

A section-by-section blueprint for turning the current ~2,000-word skeleton into a full
PFE (~60–70 pages of body). For each section: the **target length**, the **source
material** already in the repo that feeds it, its **status**, and **what to add**.

Legend — status: ✅ drafted (lift from source, light edit) · ◐ partial (expand) ·
✍️ to write (source exists as data/notes) · 🧩 needs your voice (personal/administrative).

Rough page budget (single-spaced, figures included): **~64 pages of body + front/back
matter**. Sections are ordered as they appear in `build_latex_style_pfe_pdf.py`.

---

## Front matter (~7 pp) 🧩/✅
| Item | pp | Source | Status |
|---|--:|---|---|
| Cover page | 1 | builder | ✅ (fill supervisor/jury/date) |
| Abstract + Résumé | 1 | builder (updated) | ✅ |
| Acknowledgements | 1 | — | 🧩 personal |
| Table of contents, lists of tables/figures/acronyms | 2–3 | builder | ✅ (auto-update after final pagination) |
| Declaration of AI use | 1 | — | 🧩 (state tools, dates; supervisor may require prompt evidence) |

## General Introduction (~2–3 pp) ✅→◐
Source: current builder intro + `scope.md`. **Add:** the reader's roadmap (one paragraph
per chapter), the concrete research question, and why an *independent* PFE (no host
company) still addresses a real sector problem. Keep it non-technical.

## Chapter 1 — Context and State of the Art (~12 pp) ◐ *biggest expansion*
The current chapter is ~1.5 pp; this is where most new pages come from. Turn the
literature **list** into a critical, thematic **review**.
| §  | Content | pp | Source | To add |
|---|---|--:|---|---|
| 1.1 | Introduction & problem statement | 1.5 | builder | why drift ≠ noise; the detection-vs-response tension |
| 1.2 | Sequential recommendation | 2.5 | `bibliography.md` #1–9 | discuss GRU4Rec, SASRec, BERT4Rec, Caser, NARM — *what each adds and its limit* |
| 1.3 | Long- & short-term preference modelling | 2.5 | `bibliography.md` #10–14 | LSTUR, SLi-Rec, CLSR, **SLSRec** — the learned-gate lineage; position our gap |
| 1.4 | Concept-drift detection | 2.5 | `bibliography.md` #15–21 | DDM, ADWIN, Page-Hinkley, the two surveys — why this literature was under-used in RecSys |
| 1.5 | Drift in recommenders + research gap | 2.5 | `bibliography.md` #22–25 | timeSVD++, forgetting MF, stream-based surveys; end on the **explicit-detector-vs-learned-gate** gap |
| 1.6 | Research questions & hypotheses | 1 | `experiment_protocol.md` | one table: RQ → hypothesis → metric |

**Method:** each cited work gets 2–4 sentences (contribution + limitation for *our*
question). Use the "Remaining gap for this PFE" column already in the builder's literature
table as the spine.

## Chapter 2 — Methodological Framework (~11 pp) ✍️ *source is ready*
Almost fully specified in **`method_equations.md`** and **`ranking_protocol.md`** — this is
transcription + prose, not new research.
| § | Content | pp | Source |
|---|---|--:|---|
| 2.1 | Overview: representation → sense → confirm → adapt | 1 | `architecture.md` |
| 2.2 | Data & preprocessing (dataset, k-core, split, floors) | 2 | `data_card.md`, `ranking_protocol.md` |
| 2.3 | Backbone: short-term GRU, long-term pool, fusion, scoring (all equations) | 2 | `method_equations.md` §1 |
| 2.4 | Drift signal D_u(t) + the L–S ablation | 1.5 | `method_equations.md` §2 |
| 2.5 | Detector: causal z-score + persistence state machine | 1.5 | `method_equations.md` §3 |
| 2.6 | Fusion-weight policies (static / window-B / continuous / proposed) | 1.5 | `method_equations.md` §4 |
| 2.7 | Hyperparameters (full table) + evaluation protocol | 1.5 | `method_equations.md` §5, `ranking_protocol.md` |

## Chapter 3 — Experimental Results (~18 pp) ✍️ *the arc, ready in narrative*
Expand **`results_narrative.md` §3.1–3.8** into full prose, one section per finding, each
with its table/figure and a short "reading" paragraph.
| § | Content | pp | Source | Figure/Table |
|---|---|--:|---|---|
| 3.1 | Protocol recap | 0.5 | narrative 3.1 | — |
| 3.2 | Backbone validation + **external SASRec/GRU4Rec** | 2 | `rerun_fixed_backbone.md`, `sequential_baselines.md` | floors table; SASRec/GRU4Rec table |
| 3.3 | Drift signal | 1.5 | narrative 3.3 | Fig. signal distribution |
| 3.4 | Detection + **seed CIs** | 2 | `seed_robustness.md` | detection/FA table + trade-off fig |
| 3.5 | External **detector** baselines | 2 | `detector_baselines.md` | detector-comparison fig |
| 3.6 | window-B negative finding + diagnosis | 2 | `window_b_analysis.md` | pre/post table |
| 3.7 | Dense cohort — the fusion precondition | 1.5 | `dense_cohort_probe.md` | α-sweep table |
| 3.8 | Coherent drift + the **adaptation win** (3 scenarios) | 3 | `dense_adaptation_result.md` | recovery-curves fig; 30-seed CI table |
| 3.9 | Encoder ablation (adaptation-as-insurance) | 2 | `dense_adaptation_result.md` (ablation) | EWMA-vs-attention table |
| 3.10 | Results summary | 0.5 | narrative | — |

## Chapter 4 — Ethics, Data Protection and Responsible Use (~3 pp) ✅
Already written in the builder (GDPR basis, consent/representativeness, false-alarm harm).
**Optional add:** a short paragraph on reproducibility-as-ethics (auditable runs).

## Chapter 5 — Discussion and Limitations (~7 pp) ✍️ *split out from narrative*
Currently folded into the General Conclusion; the supervisor asked for a **dedicated**
chapter. Lift **`results_narrative.md` "Chapter 5 — Discussion & Limitations"**.
| § | Content | pp |
|---|---|--:|
| 5.1 | What is established (3 results) | 1.5 |
| 5.2 | The boundaries: detection≤CUSUM; adaptation's preconditions | 2 |
| 5.3 | Adaptation as robust insurance (encoder mirror-image) | 1.5 |
| 5.4 | Threats to validity (cohort size, single backbone seed, synthetic drift, offline) | 1.5 |
| 5.5 | Why the negative→positive arc matters (intellectual honesty) | 0.5 |

## General Conclusion (~2 pp) ✅
Builder version is current. Tighten to: contribution, the mapped boundaries, and the
future-work list (stronger encoder, learned-gate head-to-head, full scale, second dataset).

## Bibliography (~4 pp) ◐
Fold the **33 entries** from `bibliography.md` into the builder (7 are already there).
**Action:** verify each against the version you read, ensure every in-text citation has an
entry. This is P2.9.

## Appendices (~6 pp) ✍️
| App. | Content | Source |
|---|---|---|
| A | Traceability & reproduction (run structure, how to re-run each phase) | `README.md`, `JOURNAL.md`, scripts |
| B | Full hyperparameter tables | `method_equations.md` §5 |
| C | Supplementary results (per-scenario recovery curves, sensitivity) | run artifacts |
| D | Submission checklist | builder |

---

## Suggested writing order (highest leverage first)
1. **Chapter 3** — the story is done in `results_narrative.md`; expanding it is fast and
   it is the core of the thesis. (~1–2 days)
2. **Chapter 2** — transcription from `method_equations.md`. (~1 day)
3. **Chapter 5** — lift and expand the Discussion. (~half day)
4. **Chapter 1** — the real writing effort: the critical literature review. (~2 days)
5. **Front/back matter, bibliography, appendices, polish.** (~1 day)

## What I can draft for you
Say the word and I'll turn any of these into full prose in the builder — most useful
targets: **Chapter 3** (I have all the numbers and figures), **Chapter 2** (equations →
prose), or **Chapter 5** (Discussion). Chapter 1's literature review is the part best
written in your own voice, but I can produce a per-reference paragraph scaffold you edit.
