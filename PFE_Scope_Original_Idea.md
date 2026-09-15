# Project Scope — Drift-Aware Long/Short-Term Fusion Recommender

## Author
Gil-Allen MOUNZEO

## Program
PGE5 — Aivancity School for AI, Paris

## Date
August 2026

---

## Why — Research Objective

This project studies how a recommender system that combines long-term and short-term user preferences can adapt more rapidly when a user's interests evolve.

The central problem is **not concept drift in recommender systems in general**. It is the specific problem of **preference drift inside long/short-term fusion recommenders**.

A long-term representation captures stable historical preferences, while a short-term representation captures recent behaviour. When a user's interests change, these two representations may become increasingly inconsistent. If the fusion mechanism continues to rely too heavily on the historical signal, recommendation quality may degrade and adaptation may occur too slowly.

The project therefore develops and evaluates a **drift-aware long/short-term fusion methodology** that:

1. represents long-term and short-term preferences separately;
2. monitors their divergence over time;
3. detects meaningful and persistent preference shifts;
4. dynamically adapts the long/short-term fusion weight;
5. measures how quickly the system detects and responds to the shift;
6. limits false alarms caused by temporary or noisy behaviour.

### Formal Research Question

**Can drift awareness reduce detection and adaptation latency in long/short-term sequential recommenders while limiting recommendation-quality degradation and false adaptation during evolving user preferences?**

---

## What — Research Constructs and Variables

The project focuses on four main constructs.

### 1. Long-term preference

A representation of persistent user interests derived from historical interactions.

### 2. Short-term preference

A representation of recent user behaviour intended to capture current or emerging interests.

### 3. Preference drift / long-short divergence

A measurable change indicating that the recent preference representation is becoming inconsistent with the user's longer-term preference.

A general formulation is:

$$D_u(t) = d(L_u(t), S_u(t))$$

where $L_u(t)$ is the long-term preference representation, $S_u(t)$ is the short-term representation, and $d(\cdot)$ is a divergence or distance measure.

### 4. Adaptive fusion

The mechanism controlling the contribution of long-term and short-term preferences:

$$H_u(t) = \alpha_u(t)L_u(t) + (1-\alpha_u(t))S_u(t)$$

High $\alpha$ means greater reliance on long-term preference; low $\alpha$ means greater reliance on recent preference.

The key research problem is determining **when and how $\alpha_u(t)$ should change in response to preference drift**.

---

## Primary Research Variables

### Detection latency

Number of interactions between the effective onset of a preference shift and its detection:

$$L_{detect}=t_{detect}-t_{shift}$$

### Adaptation latency

Number of interactions between drift detection and an effective change in fusion behaviour:

$$L_{adapt}=t_{adapt}-t_{detect}$$

### Recovery latency

Number of interactions between the onset of a shift and recovery of recommendation quality:

$$L_{recovery}=t_{recovery}-t_{shift}$$

### Recommendation quality

Primary: NDCG@10. Secondary: Recall@10, Hit Rate@10, and MRR where appropriate.

### Stability / false adaptation

Measure how often the system changes its fusion behaviour in response to temporary or noisy deviations that do not represent a persistent preference change.

### Computational overhead

Inference latency, memory usage, and additional cost of drift monitoring and adaptive fusion.

---

## Where — Research Environment

The project is an academic research project conducted at Aivancity School for AI, Paris, France.

Experiments are performed in a controlled offline environment using publicly available Amazon Product Reviews data. The project is a research prototype rather than a live production recommender.

No live Amazon infrastructure or human-subject testing is required.

---

## When — Project Timeline

| Phase | Period | Activity |
|---|---|---|
| Phase 1 | February 2026 | Problem definition, literature review, research framing |
| Phase 2 | March–April 2026 | Data preparation and baseline long/short-term recommender |
| Phase 3 | May–June 2026 | Drift-awareness and adaptive fusion methodology |
| Phase 4 | July 2026 | Controlled experiments, ablations, and evaluation |
| Phase 5 | August 2026 | Results analysis, final report, and submission |

---

## Who — Study Population

The study focuses on individual Amazon users with sufficient historical interactions to construct long-term and short-term preference representations. The current data pipeline requires a minimum of 20 interactions per user.

No users are recruited or contacted. All experiments use retrospective public data.

---

# How — Research Design and Methodology

The methodology is organized around a closed adaptive loop:

**User interactions → Long-term preference + Short-term preference → Drift monitoring → Preference-shift detection → Fusion adaptation → Next-item recommendation → Post-adaptation evaluation**

## Phase 1 — Data Preparation

The current Amazon loader is retained as the basis of the dataset.

Current parameters:

- Categories: Electronics; Clothing_Shoes_and_Jewelry; Books
- Date range: 2014-01-01 to 2018-12-31
- Minimum user interactions: 20
- Minimum item interactions: 5
- Only verified purchases are retained
- Feedback: explicit Amazon rating stored as `preference_score`
- Item identifier: `parent_asin`
- Duplicate key: `(user_id, item_id, timestamp)`
- Random seed: 42

The dataset is sorted chronologically and split without shuffling.

## Phase 2 — Temporal Baseline

A baseline next-item recommender explicitly separates long-term and short-term preference. A fixed or otherwise non-drift-aware fusion strategy is established first.

## Phase 3 — Preference Drift Monitoring

At each user interaction time $t$:

1. construct the long-term representation;
2. construct the short-term representation;
3. calculate their divergence;
4. maintain a history of the drift signal;
5. determine whether the divergence is persistent enough to indicate a meaningful preference shift.

The detector must not react to a single anomalous interaction. It should distinguish stable behaviour, temporary deviation, emerging shift, and persistent shift.

## Phase 4 — Adaptive Fusion

When a meaningful shift is detected, the system changes the fusion weight $\alpha_u(t)$. The primary adaptation mechanism is **fusion control**, not full model retraining.

Candidate strategies include gradual reduction of long-term weight, temporary short-term dominance, progressive restoration toward the long-term signal, and incremental updating of the long-term representation after a persistent shift.

## Phase 5 — Controlled Preference-Shift Evaluation

Controlled scenarios provide known shift onset times for precise latency measurement:

- sudden shift;
- gradual shift;
- recurring shift.

Category changes may make scenarios interpretable, but the research definition of drift is based on **user preference evolution**, not category labels alone. Controlled drift must be clearly distinguished from naturally occurring preference changes.

## Phase 6 — Natural Preference Evolution

Where sufficient examples exist, evaluate the methodology on naturally occurring chronological changes in user behaviour as a complement to controlled experiments.

---

# Baselines

### Baseline A — Static Long/Short Fusion

Long-term and short-term preferences are fused using a fixed strategy.

### Baseline B — Sliding / Recent-Window Strategy

Uses a fixed recent history without explicit drift awareness.

### Baseline C — Periodic Adaptation

Updates the recommender on a fixed schedule without explicitly detecting preference shifts.

### Baseline D — Adaptive Fusion Without Drift Awareness

Allows the fusion weight to vary but does not use an explicit preference-drift signal.

### Proposed Method — Drift-Aware Long/Short Fusion

Uses preference divergence and persistence-aware drift detection to dynamically adapt the long/short-term fusion.

---

# Ablation Study

| Configuration | Drift Awareness | Adaptive Fusion | Purpose |
|---|---|---|---|
| A | No | No | Static reference |
| B | No | Yes | Effect of adaptation without drift awareness |
| C | Yes | No | Effect of detection alone |
| D | Yes | Yes | Full proposed methodology |
| E | Yes | Yes | Alternative divergence / threshold strategy |

Additional experiments should study short-term window size, long-term history definition, divergence measure, fixed vs dynamic threshold, persistence requirement, and abrupt vs gradual fusion adaptation.

---

# Evaluation

## Primary Evaluation Questions

**Q1 — Detection speed:** Does the proposed method detect preference shifts faster than non-drift-aware baselines?

**Q2 — Adaptation speed:** After detection, does the fusion mechanism adapt faster?

**Q3 — Recommendation recovery:** Does faster adaptation reduce recommendation-quality degradation and recovery time?

**Q4 — Stability:** Does the methodology avoid excessive adaptation to temporary or noisy behaviour?

**Q5 — General behaviour across users:** Does the method remain useful for both stable and highly volatile users?

## Main Metrics

| Metric | Purpose |
|---|---|
| Detection latency | Speed of recognizing a preference shift |
| Adaptation latency | Speed of changing fusion behaviour |
| Recovery latency | Time to recover recommendation quality |
| NDCG@10 | Primary recommendation-quality measure |
| Recall@10 | Recommendation retrieval quality |
| Hit Rate@10 | Recommendation success |
| False adaptation rate | Robustness to temporary/noisy behaviour |
| Detection precision/recall | Drift detection quality |
| Inference latency | Computational overhead |
| Memory usage | Computational overhead |

---

# Temporal Evaluation Rules

The evaluation must simulate the chronological user experience.

At time $t$, the system may only use information available at or before $t$.

For each user:

**History up to t → construct $L_u(t)$ and $S_u(t)$ → calculate drift signal → decide whether adaptation is required → generate next-item recommendation → observe next interaction → update the state.**

No future interactions may be used for representation construction, threshold tuning, drift decisions, adaptation, or model selection.

The temporal data split is: earliest 70% for training, next 10% for validation, latest 20% for testing.

---

# Dataset

The primary evaluation dataset is Amazon Product Reviews using Electronics, Clothing_Shoes_and_Jewelry, and Books with verified-purchase ratings from 2014–2018.

The processed interaction schema includes:

- `user_id`
- `item_id`
- `category`
- `preference_score`
- `timestamp`
- `user_idx`
- `item_idx`

The dataset is used for next-item recommendation and temporal user-preference analysis.

---

# Out of Scope

The project explicitly excludes:

- generic concept-drift methodology for arbitrary recommender systems;
- item-side/catalog drift;
- cold-start recommendation;
- group or multi-user recommendation;
- cross-platform user drift;
- live production deployment;
- real-user A/B testing;
- permanent long-term profile rewriting after every detected deviation;
- treating product-category changes as the sole definition of preference drift.

---

# Expected Contribution

The intended contribution is a **drift-aware methodology for long/short-term fusion recommenders**.

The methodology should demonstrate whether monitoring the divergence between long-term and short-term preference signals can enable a recommender to:

1. detect meaningful preference changes earlier;
2. adapt its fusion weight more rapidly;
3. reduce recommendation degradation after a shift;
4. recover recommendation quality faster;
5. remain stable when behavioural changes are temporary or noisy.

The project therefore focuses on:

$$\boxed{\text{Preference Evolution} \rightarrow \text{Drift Awareness} \rightarrow \text{Fusion Adaptation} \rightarrow \text{Faster Recovery}}$$

rather than on drift detection as an isolated task.

---

# Main Research Hypotheses

### H1
Drift-aware long/short-term fusion detects meaningful preference shifts with lower detection latency than static or non-drift-aware fusion.

### H2
Drift-aware fusion reduces adaptation and recovery latency following a persistent preference shift.

### H3
Drift-aware fusion reduces post-shift recommendation degradation compared with static long/short-term fusion.

### H4
Gradual fusion-weight adaptation provides a better stability/adaptation trade-off than abrupt switching.

### H5
Persistence-aware drift detection reduces false adaptations caused by temporary behavioural deviations.

### H6
Adaptive thresholds improve robustness across users with different levels of behavioural volatility.

---

# Final Research Positioning

The project should consistently be described as:

**Drift-Aware Long/Short-Term Fusion for Sequential Recommenders**

The central question is:

> **When a user's recent interests become inconsistent with their historical preferences, can a recommender recognize that the current long/short-term fusion is becoming inappropriate and adapt the balance quickly enough to preserve recommendation quality?**

The detector is therefore not the final objective.

The key methodological relationship is:

$$\boxed{D_u(t) \rightarrow \alpha_u(t) \rightarrow Recommendation_t}$$

where the drift signal influences the long/short-term fusion itself.
