# Bibliography (working reference list)

Expanded reference set for Chapters 1–2 (target 30–40, up from the original 7),
grouped by theme. Every entry is a real, checkable work; **verify each one personally
before submission** (venue, year, page/DOI) — this list is the research scaffold, not a
substitute for that check. Entries flagged ⚠ need a field confirmed (marked inline).

## Sequential recommendation (backbones / baselines — P1.1)

1. Hidasi, B., Karatzoglou, A., Baltrunas, L., & Tikk, D. (2016). *Session-based
   Recommendations with Recurrent Neural Networks* (GRU4Rec). ICLR.
2. Hidasi, B., & Karatzoglou, A. (2018). *Recurrent Neural Networks with Top-k Gains
   for Session-based Recommendations* (GRU4Rec+). CIKM.
3. Tang, J., & Wang, K. (2018). *Personalized Top-N Sequential Recommendation via
   Convolutional Sequence Embedding* (Caser). WSDM.
4. Kang, W.-C., & McAuley, J. (2018). *Self-Attentive Sequential Recommendation*
   (SASRec). ICDM.
5. Sun, F., Liu, J., Wu, J., Pei, C., Lin, X., Ou, W., & Jiang, P. (2019). *BERT4Rec:
   Sequential Recommendation with Bidirectional Encoder Representations from
   Transformer*. CIKM.
6. Li, J., Ren, P., Chen, Z., Ren, Z., Lian, T., & Ma, J. (2017). *Neural Attentive
   Session-based Recommendation* (NARM). CIKM.
7. Rendle, S., Freudenthaler, C., & Schmidt-Thieme, L. (2010). *Factorizing Personalized
   Markov Chains for Next-Basket Recommendation* (FPMC). WWW.
8. Zhou, K., Wang, H., Zhao, W. X., et al. (2020). *S³-Rec: Self-Supervised Learning for
   Sequential Recommendation with Mutual Information Maximization*. CIKM.
9. Zhao, W. X., Mu, S., Hou, Y., et al. (2021). *RecBole: Towards a Unified,
   Comprehensive and Efficient Framework for Recommendation Algorithms*. CIKM.
   *(the framework providing GRU4Rec/SASRec/BERT4Rec baselines — P1.1)*

## Long- and short-term preference modelling (the fusion literature)

10. An, M., Wu, F., Wu, C., Zhang, K., Liu, Z., & Xie, X. (2019). *Neural News
    Recommendation with Long- and Short-term User Representations* (LSTUR). ACL.
11. Yu, Z., Lian, J., Mahmoody, A., Liu, G., & Xie, X. (2019). *Adaptive User Modeling
    with Long and Short-Term Preferences for Personalized Recommendation* (SLi-Rec).
    IJCAI.
12. Zheng, Y., Liu, S., Li, Z., & Wu, S. (2022). *Disentangling Long and Short-Term
    Interests for Recommendation* (CLSR). WWW.
13. Lv, F., Wu, T., et al. (2019). *SDM: Sequential Deep Matching Model for Online
    Large-scale Recommender System*. CIKM.
14. Zhou, W., Shen, Y., Ji, J., Feng, Y., Tang, X., He, X., Feng, L., & Zhu, Z. (2026).
    *SLSRec: Self-Supervised Contrastive Learning for Adaptive Fusion of Long- and
    Short-Term User Interests*. arXiv:2604.04530 (preprint, 6 Apr 2026). Disentangles
    long/short interests via self-supervised contrastive learning and combines them with
    an **attention-based fusion network** (a *learned* weighting). Positions our gap
    precisely: we drive the fusion weight from an *explicit online drift-detector* rather
    than a learned gate — the direct comparison for P1.1.

## Concept-drift detection (the detector literature — P1.1)

15. Gama, J., Medas, P., Castillo, G., & Rodrigues, P. (2004). *Learning with Drift
    Detection* (DDM). SBIA.
16. Baena-García, M., del Campo-Ávila, J., Fidalgo, R., et al. (2006). *Early Drift
    Detection Method* (EDDM). ECML/PKDD workshop.
17. Bifet, A., & Gavaldà, R. (2007). *Learning from Time-Changing Data with Adaptive
    Windowing* (ADWIN). SDM.
18. Page, E. S. (1954). *Continuous Inspection Schemes* (the Page–Hinkley test).
    Biometrika, 41(1/2), 100–115.
19. Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M., & Bouchachia, A. (2014). *A
    Survey on Concept Drift Adaptation*. ACM Computing Surveys, 46(4).
20. Lu, J., Liu, A., Dong, F., Gu, F., Gama, J., & Zhang, G. (2019). *Learning under
    Concept Drift: A Review*. IEEE TKDE, 31(12).
21. Bifet, A., Holmes, G., Kirkby, R., & Pfahringer, B. (2010). *MOA: Massive Online
    Analysis*. JMLR, 11.

## Drift / non-stationarity in recommender systems

22. Koren, Y. (2009). *Collaborative Filtering with Temporal Dynamics* (timeSVD++). KDD.
23. Jugovac, M., Jannach, D., & Karimi, M. (2018). *StreamingRec: A Framework for
    Benchmarking Stream-based News Recommenders*. RecSys.
24. Matuszyk, P., Vinagre, J., Spiliopoulou, M., Jorge, A. M., & Gama, J. (2015).
    *Forgetting Methods for Incremental Matrix Factorization in Recommender Systems*.
    SAC.
25. Al-Ghossein, M., Abdessalem, T., & Barré, A. (2021). *A Survey on Stream-Based
    Recommender Systems*. ACM Computing Surveys, 54(5).

## Dataset

26. Hou, Y., Li, J., He, Z., Yan, A., Chen, X., & McAuley, J. (2024). *Bridging Language
    and Items for Retrieval and Recommendation* (**Amazon Reviews'23** — the dataset
    actually used; McAuley Lab, UC San Diego). arXiv:2403.03952. **Primary dataset
    citation.**
27. Ni, J., Li, J., & McAuley, J. (2019). *Justifying Recommendations using
    Distantly-Labeled Reviews and Fine-Grained Aspects* (Amazon Review Data 2018).
    EMNLP-IJCNLP. *(earlier release lineage — not the version used here).*
28. He, R., & McAuley, J. (2016). *Ups and Downs: Modeling the Visual Evolution of
    Fashion Trends with One-Class Collaborative Filtering*. WWW. *(earliest Amazon
    release lineage).*

## Evaluation methodology

29. Krichene, W., & Rendle, S. (2020). *On Sampled Metrics for Item Recommendation*.
    KDD. *(justifies our full, unsampled ranking — P0.5)*.
30. Järvelin, K., & Kekäläinen, J. (2002). *Cumulated Gain-Based Evaluation of IR
    Techniques* (NDCG). ACM TOIS, 20(4).
31. Cañamares, R., & Castells, P. (2020). *On Target Item Sampling in Offline
    Recommender System Evaluation*. RecSys.
32. Dallmann, A., Zoller, D., & Hotho, A. (2021). *A Case Study on Sampling Strategies
    for Evaluating Neural Sequential Item Recommendation Models*. RecSys.

## Optimisation / training

33. Kingma, D. P., & Ba, J. (2015). *Adam: A Method for Stochastic Optimization*. ICLR.
34. Rendle, S., Freudenthaler, C., Gantner, Z., & Schmidt-Thieme, L. (2009). *BPR:
    Bayesian Personalized Ranking from Implicit Feedback*. UAI.

---

**Original 7 (retained, verified):** the entries the draft already carried — keep the
ones that survive the personal re-check and fold them into the numbering above.
