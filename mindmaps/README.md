# Mermaid Mindmaps — Visual Study Guide

> Open this file in any Mermaid-compatible viewer (GitHub, VS Code with Mermaid extension, mermaid.live) to see the rendered mindmaps.

*Last updated: 2026-03-29*

---

## Section 0: AI/ML Primer

```mermaid
mindmap
  root((AI / ML\nPrimer))
    Supervised Learning
      Classification
        Spam detection
        Fraud detection
        Medical diagnosis
        Image recognition
      Regression
        Price prediction
        ETA estimation
        Demand forecasting
    Unsupervised Learning
      Clustering
        Customer segmentation
        Topic modeling
      Dimensionality Reduction
        PCA
        t-SNE
      Anomaly Detection
        Security threats
        Manufacturing defects
    Reinforcement Learning
      Game AI - AlphaGo
      Robotics
      Self-driving cars
      Dynamic pricing
    Key Concepts
      Features = inputs
      Labels = correct answers
      Training = learning patterns
      Inference = making predictions
      Loss = how wrong the model is
      Gradient Descent = adjust to reduce loss
    Training Loop
      Input batch
      Predict
      Compare to label
      Calculate loss
      Adjust weights
      Repeat
    Pitfalls
      Underfitting - too simple
      Overfitting - memorized
      Data leakage
      Class imbalance
```

---

## Section 1: CPU vs GPU Model Selection

```mermaid
mindmap
  root((CPU vs GPU\nModel Selection))
    CPU - Use When
      Small models 1B-8B params
      Quantized models 4bit 8bit
      Encoder models
        BERT
        DistilBERT
        Sentence transformers
      Task types
        Embeddings
        Classification
        Reranking
        Search
      Deployment
        Edge devices
        On-prem
        Low traffic under 50 QPS
      Runtimes
        ONNX Runtime
        OpenVINO
        llama.cpp
    GPU - Use When
      Large models 13B+ params
      High throughput generation
      Low latency under 100ms
      Task types
        Text generation
        Image generation
        Video generation
        Multimodal
      Deployment
        Production chatbots
        Multi-user serving
        High concurrency
      Runtimes
        vLLM
        TensorRT
        TGI
    Decision Factors
      Model size in parameters
      Queries per second
      Latency requirements
      Budget constraints
      Operational complexity
```

---

## Section 2: ML Metrics Framework

```mermaid
mindmap
  root((ML Metrics\nFramework))
    Layer 1 - Business Revenue
      Revenue uplift
      Margin improvement
      Cost savings
      Customer LTV
      Churn reduction
    Layer 2 - User Engagement
      CTR
      Session length
      Retention rate
      Completion rate
      NPS / CSAT
    Layer 3 - Model Quality
      F1 Score
      AUC-ROC
      NDCG
      BLEU / ROUGE
      Hallucination rate
    Layer 4 - Operational
      Latency p50 p99
      Uptime / SLA
      Throughput QPS
      Error rate
    Layer 5 - Risk Trust Safety
      Fairness metrics
      Data drift
      Toxicity scores
      Compliance flags
      Bias auditing
    Framework Pattern
      North Star = 1 business metric
      Leading Indicators = engagement + quality
      Guardrails = ops + trust + safety
    By System Type
      Recommendations
        CTR, add-to-cart, conversion
      Fraud Detection
        Precision, Recall, fraud $ blocked
      Search
        MRR, NDCG, search success
      Ads
        CTR, CPC, CPM, ROAS
      Copilots
        Task completion, CSAT
```

---

## Section 3: System Design Steps

```mermaid
mindmap
  root((System Design\n6 Steps))
    Step 1 - Requirements
      Business objective
      User problem
      Scale - users, items, QPS
      Latency SLA
      Data availability
      Constraints
      Assumptions
    Step 2 - Metrics
      Offline metrics
        F1, AUC, NDCG, RMSE
      Online metrics
        CTR, conversion, CSAT
      Guardrails
        Latency, safety, drift
    Step 3 - Architecture
      Data ingestion
      Feature pipeline
      Training pipeline
      Model registry
      Serving layer
      A/B platform
      Monitoring
    Step 4 - Model Building
      Data generation
        User logs
        Human annotation
        Heuristic labels
        Active learning
      Featurization
        User features
        Item features
        Context features
        Embeddings
      Model training
        Baseline first
        Iterate features
        Iterate architecture
        Iterate training
      Evaluation
        Component metrics
        System metrics
        Business metrics
    Step 5 - Deep Dive
      Feature engineering
      Serving latency
      Cold start
      Fairness / bias
      Scaling strategy
      Failure modes
    Step 6 - Monitoring
      Model metrics
      Product metrics
      Ops health
      Trust / safety
      Retraining triggers
```

---

## Section 4: Offline vs Online Metrics

```mermaid
mindmap
  root((Offline vs\nOnline Metrics))
    Offline Metrics
      Properties
        Fast - seconds
        Cheap - no traffic
        Safe - no user impact
        Repeatable
        First gate
      Limitations
        Selection bias
        Static snapshot
        System effects missed
        Proxy not truth
      By System
        Classification - F1, AUC, PR-AUC
        Regression - MAE, RMSE, MAPE
        Ranking - NDCG, Recall@K
        NLP - BLEU, ROUGE, EM
        GenAI - Groundedness, halluc rate
    Online Metrics
      Properties
        Real users
        Business impact
        Deployment decision
        Requires A/B test
        Statistical rigor
      By System
        Recs - CTR, conversion, revenue
        Search - success rate, reformulation
        Fraud - blocked dollars, complaint rate
        Ads - CTR, CPC, CPM, ROAS
        LLM - task completion, CSAT
    Critical Insight
      Offline wins != Online wins
      Selection bias
      System interactions
      User behavior shifts
```

---

## Section 5: Business Metrics

```mermaid
mindmap
  root((Business\nMetrics))
    CTR - Click Through Rate
      Formula: Clicks / Impressions
      Measures engagement
      Leading indicator
    CPC - Cost Per Click
      Formula: Spend / Clicks
      Efficiency of spend
    CPM - Cost Per Mille
      Formula: Spend / Impressions x 1000
      Brand awareness metric
    CPA - Cost Per Acquisition
      Formula: Spend / Conversions
      Acquisition efficiency
    CVR - Conversion Rate
      Formula: Conversions / Clicks
      Funnel effectiveness
    ROAS - Return on Ad Spend
      Formula: Revenue / Ad Spend
      Ultimate business metric
    Funnel Flow
      Impressions - CPM
      Clicks - CTR, CPC
      Conversions - CVR, CPA
      Revenue - ROAS
    Key Lesson
      High CTR can reduce revenue
      Engagement != Business outcome
      Always track full funnel
```

---

## Section 6: Offline Metrics Deep Dive

```mermaid
mindmap
  root((Offline Metrics\nDeep Dive))
    What They Are
      Computed on held-out test data
      Before deployment
      Model never saw this data
    Why Use Them
      Fast - seconds to minutes
      Cheap - just compute
      Safe - no user risk
      Repeatable
      First gate before online
    Pipeline
      Raw data
      Label / annotate
      Train-Val-Test split 70-15-15
      Train model
      Evaluate on test set
      Compute metrics
    Limitations
      Selection bias
      Static data
      System effects
      Proxy for business
    Analogy
      Practice exam vs real exam
      Driving simulator vs real road
    Fit in System Design
      Build candidate model
      Offline eval - PASS or FAIL
      Online A/B test
      Full rollout
```

---

## Section 7: Metrics by Model Type

```mermaid
mindmap
  root((Metrics by\nModel Type))
    Classification
      Confusion Matrix
        TP, FP, FN, TN
      Accuracy - overall correctness
      Precision - positive prediction quality
        Use when FP costly
        Spam filter
      Recall - catch rate
        Use when FN costly
        Cancer screening
      F1 - balanced score
      AUC-ROC - threshold independent
      PR-AUC - imbalanced data
    Regression
      MAE - average absolute error
      RMSE - penalizes large errors
      MAPE - percentage error
      R-squared - variance explained
    Ranking
      Precision@K - top K relevance
      Recall@K - coverage in top K
      NDCG - position-weighted quality
      MAP - avg precision across queries
      MRR - first relevant position
    NLP / GenAI
      Exact Match
      BLEU - translation overlap
      ROUGE - summary overlap
      Groundedness - source backed
      Hallucination rate
      BERTScore - semantic similarity
    Clustering
      Silhouette - separation quality
      Purity - class alignment
      Adjusted Rand Index
      Davies-Bouldin Index
```

---

## Model Selection Guide

```mermaid
mindmap
  root((Model Selection\nby Use Case))
    Spam / Fraud Detection
      Start: Logistic Regression
      Iterate: XGBoost / LightGBM
      Advanced: Deep NN
      Metric: Precision, Recall, PR-AUC
    Recommendations
      Start: Collaborative filtering
      Iterate: Two-tower model
      Advanced: Multi-task DNN
      Metric: NDCG, Recall@K
    Search Ranking
      Start: BM25 + simple reranker
      Iterate: LambdaMART
      Advanced: Cross-encoder transformer
      Metric: MRR, NDCG@5
    ETA / Price Prediction
      Start: Linear Regression
      Iterate: XGBoost
      Advanced: Deep ensemble
      Metric: RMSE, MAPE
    Ads CTR Prediction
      Start: Logistic Regression
      Iterate: Wide and Deep
      Advanced: DCN / DIN
      Metric: AUC, Log Loss
    Content Moderation
      Start: Rule-based + simple classifier
      Iterate: Fine-tuned BERT
      Advanced: Multi-modal transformer
      Metric: Precision, Recall
    Chatbot / QA
      Start: Retrieval + template
      Iterate: Fine-tuned LLM
      Advanced: RAG + RLHF
      Metric: Groundedness, CSAT
    Customer Segmentation
      Start: K-Means
      Iterate: DBSCAN
      Advanced: Gaussian Mixture
      Metric: Silhouette, Purity
```

---

## Section 9: Interview Cheat Sheet

```mermaid
mindmap
  root((Interview\nCheat Sheet))
    Time Budget - 35 min
      Requirements 3-5 min
      Metrics 2-3 min
      Architecture 5-8 min
      Model Building 5-8 min
      Deep Dive 5-10 min
      Monitoring 2-3 min
    Metrics Pattern
      North Star - 1 business metric
      Leading - engagement + quality
      Guardrails - ops + trust
    Quick Reference
      Technically better? -> Offline
      Helped real users? -> Online
      Anything broken? -> Guardrails
      Making money? -> North Star
    Pro Tips
      Always start with baseline
      State assumptions explicitly
      Explain WHY each iteration helps
      Tie metrics to business
      Discuss failure modes
      Show monitoring is continuous
```
