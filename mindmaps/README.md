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

## Search Relevance Architecture

```mermaid
mindmap
  root((Search Relevance\nArchitecture))
    Online Serving Path
      Query arrives from user
      Candidate Retrieval
        BM25 keyword match
        ANN embedding search
        1M docs to 1000 candidates
        Metric: Recall@1000
        Latency: 10-50ms
      Document Scoring
        Two-tower model
        1000 to 200
        Metric: AUC
        Latency: 20-50ms
      Filtering
        Out of stock removal
        Policy violations
        Geo-restrictions
        Metric: Compliance rate
      Ranker
        Cross-encoder transformer
        50+ features
        200 to 20
        Metric: NDCG@10
        Latency: 30-100ms
      Ranked Results to User
    Offline Training Path
      User Interactions
        Clicks
        Purchases
        Dwell time
        Skips
      Training Data Generator
        Join features with labels
        Daily batch job
      Model Training
        LambdaRank
        Cross-encoder
        30 days of data
      Model Publishing
        Offline eval gate
        NDCG@10 must improve
      Model Repository
        Versioned models
        Shadow mode testing
    Document Indexing
      Crawl product catalog
      Extract features
      Generate embeddings
      Build inverted index + ANN index
    Monitoring
      NDCG@10 from clicks
      CTR on top 3
      Zero-result rate under 2%
      p99 latency under 200ms
      Model freshness under 7 days
    Key Interview Points
      Draw both loops first
      Explain the funnel 1M to 1000 to 200 to 20
      Justify two-stage ranking
      Discuss cold-start for new docs
      Address position bias in training
```

---

## Metrics Explainer — What Each Metric Actually Means

> If you only read one section, read this. Every metric explained in plain English with real-world analogies.

### Classification Metrics Explained

```mermaid
mindmap
  root((Classification\nMetrics Explained))
    Accuracy
      What: Out of ALL predictions how many were correct
      Formula: Correct / Total
      Analogy: Your overall test score
      Gotcha: Useless if 99% of data is one class
      Example: 95% accuracy on fraud but only 1% is fraud - misleading
    Precision
      What: When you PREDICTED positive how often were you RIGHT
      Formula: True Positives / All Predicted Positives
      Analogy: When you say YES how often are you correct
      High precision means: Few false alarms
      Use when: False alarms are expensive
      Example: Spam filter - marking real email as spam annoys users
      Example: Fraud alert - blocking legit purchases loses revenue
    Recall
      What: Out of all ACTUAL positives how many did you CATCH
      Formula: True Positives / All Actual Positives
      Analogy: What percentage of fish in the lake did your net catch
      High recall means: You miss very few real cases
      Use when: Missing a case is dangerous
      Example: Cancer screening - missing cancer can kill someone
      Example: Fraud - missing fraud means money lost
    F1 Score
      What: Single score balancing Precision AND Recall
      Formula: 2 x Precision x Recall / Precision + Recall
      Analogy: Overall grade combining accuracy and coverage
      Use when: You need ONE number for Precision vs Recall
      Why harmonic mean: Penalizes if either is very low
      Example: F1 of 0.8 means good balance of catching and accuracy
    AUC-ROC
      What: How well model SEPARATES positives from negatives across ALL thresholds
      Score range: 0.5 = random guess to 1.0 = perfect
      Analogy: If you randomly pick one positive and one negative how often does model rank the positive higher
      Use when: You want threshold-independent quality measure
      Example: Fraud model AUC 0.95 means it almost always ranks fraud higher than non-fraud
    PR-AUC
      What: Like AUC but focused on the POSITIVE class
      Use when: Data is very imbalanced like 0.1% fraud
      Why better than AUC-ROC: AUC-ROC can look good even when model is bad at rare class
      Example: Fraud detection with 1 in 1000 fraud rate
```

### Regression Metrics Explained

```mermaid
mindmap
  root((Regression\nMetrics Explained))
    MAE - Mean Absolute Error
      What: Average of how far off each prediction is
      Formula: Average of absolute differences
      Analogy: On average how many minutes late or early is the bus
      Units: Same as your target - dollars, minutes etc
      Use when: You want simple interpretable error
      Example: MAE of $15K on house prices means on average off by $15K
    RMSE - Root Mean Square Error
      What: Like MAE but PUNISHES big errors more
      Formula: Square root of average of squared errors
      Analogy: Like MAE but a prediction off by 100 is punished MORE than ten predictions off by 10
      Use when: Big errors are much worse than small ones
      Example: ETA off by 60 min is much worse than 6 ETAs off by 10 min
      Key insight: RMSE is always bigger than or equal to MAE
    MAPE - Mean Absolute Percentage Error
      What: Average percentage you are off by
      Formula: Average of percentage errors
      Analogy: Are you off by 5% or 50%
      Use when: Relative accuracy matters more than absolute
      Example: Predicting demand - being off by 10 units matters differently for 100 units vs 10K units
      Gotcha: Breaks when actual value is zero
    R-squared
      What: How much of the variation in data your model explains
      Range: 0 = no better than guessing the average to 1 = perfect
      Analogy: How much of the exam answers could you explain vs just guessing
      Use when: You want to compare model power across different datasets
```

### Ranking Metrics Explained

```mermaid
mindmap
  root((Ranking\nMetrics Explained))
    NDCG - Normalized Discounted Cumulative Gain
      What: Are the BEST items near the TOP of your list
      Key idea: Position matters - relevant item at position 1 counts more than at position 10
      Analogy: Google search - you want the best result FIRST not buried on page 3
      Score range: 0 to 1 where 1 = perfect ranking
      The Discount: Each position gets less credit - position 1 gets full credit position 10 gets very little
      Example: Two search results both show 3 relevant items but the one with relevant items at positions 1-2-3 scores higher than one at positions 5-8-10
      Used in: Search ranking, recommendation feeds, ad ranking
    Precision at K
      What: Of the top K items you showed how many were actually relevant
      Analogy: You recommended 10 movies - how many did the user actually like
      Example: Precision@10 = 0.7 means 7 of your top 10 recommendations were good
      Used in: Top-N recommendations, search results page
    Recall at K
      What: Of ALL relevant items how many appeared in your top K
      Analogy: There are 50 good movies - how many did your top 10 list include
      Example: Recall@10 = 0.2 means your top 10 captured 20% of all good items
      Used in: Candidate retrieval - did we fetch enough good items
    MRR - Mean Reciprocal Rank
      What: How quickly does the FIRST relevant result appear
      Formula: 1 divided by position of first relevant result averaged across queries
      Analogy: When you google something how far down do you scroll to find the answer
      Example: First relevant at position 1 = score 1.0 and at position 3 = score 0.33
      Used in: QA systems, autocomplete, single-answer search
    MAP - Mean Average Precision
      What: Average quality of ranking across many different queries
      Analogy: Not just one search query but your AVERAGE ranking quality across thousands of searches
      Used in: Information retrieval benchmarks, search engine evaluation
```

### NLP and GenAI Metrics Explained

```mermaid
mindmap
  root((NLP / GenAI\nMetrics Explained))
    BLEU
      What: How much does model output OVERLAP with a reference translation word by word
      Analogy: Comparing your essay to the answer key - how many phrases match
      Range: 0 to 1 where higher is better
      Used in: Machine translation
      Gotcha: Can miss that two different phrasings mean the same thing
      Example: Reference is Bonjour and model says Hello - BLEU would be 0 even though meaning is same
    ROUGE
      What: How much of the REFERENCE summary appears in the model output
      Difference from BLEU: BLEU checks precision and ROUGE checks recall
      Analogy: Did the summary capture all the key points from the original
      Variants: ROUGE-1 is unigrams and ROUGE-L is longest common subsequence
      Used in: Text summarization
    Exact Match
      What: Does the predicted answer EXACTLY equal the correct answer
      Very strict: One word different means score is 0
      Used in: QA where there is one correct answer like a date or name
      Example: Correct answer is Paris - if model says Paris France then EM is 0
    Groundedness
      What: Is every claim in the answer SUPPORTED by the source document
      Analogy: Did the student only write facts from the textbook or make things up
      Used in: RAG systems, enterprise chatbots, any LLM using retrieved documents
      Example: Source says founded in 1998 - grounded answer says 1998 - hallucinated answer says 1995
    Hallucination Rate
      What: How often does the model INVENT facts not in the source
      Why critical: Users trust AI answers - wrong facts cause real damage
      Used in: Any production LLM deployment
      Example: Model says CEO is John Smith when source never mentions any CEO
    BERTScore
      What: Semantic similarity using embeddings - do they MEAN the same thing even if words differ
      Advantage over BLEU: Catches paraphrases and synonyms
      Example: happy and joyful would score high even though different words
      Used in: Open-ended generation, paraphrase detection
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
