## Table of Contents

1. [Data Preprocessing with DuckDB](#data-preprocessing-with-duckdb-optimization)
   - [Load and Filter Data with DuckDB](#1-load-and-filter-data-with-duckdb)
   - [Clean Text Data with Parallel Processing](#2-clean-text-data-with-parallel-processing)
   - [Create Initial Metrics with DuckDB](#3-create-initial-metrics-with-duckdb)

2. [Approach 1: Direct LLM Analysis](#approach-1-direct-llm-analysis)
   - [Optimized Sample Selection with DuckDB](#1-optimized-sample-selection-with-duckdb)
   - [Gemini API Integration with Structured Response](#2-gemini-api-integration-with-structured-response)
   - [Process Results and Enrichment](#3-process-results-and-enrichment)

3. [Approach 2: LDA + K-Means Clustering](#approach-2-lda--k-means-clustering)
   - [Text Vectorization](#1-text-vectorization)
   - [LDA Topic Modeling and Topic Similarity Analysis](#2-lda-topic-modeling-and-topic-similarity-analysis)
   - [K-Means Clustering on Consolidated LDA Results](#3-k-means-clustering-on-consolidated-lda-results)
   - [Gemini Labeling of Clusters with Structured Response](#4-gemini-labeling-of-clusters-with-structured-response)

4. [Approach 3: Embeddings + Clustering](#approach-3-embeddings--clustering)
   - [Generate Embeddings](#1-generate-embeddings)
   - [Dimensionality Reduction](#2-dimensionality-reduction-optional)
   - [Advanced Clustering with Similarity Analysis](#3-advanced-clustering-with-similarity-analysis)
   - [Extract Cluster Representatives with Advanced Analysis](#4-extract-cluster-representatives-with-advanced-analysis)
   - [Gemini Labeling of Clusters with Structured Response](#5-gemini-labeling-of-clusters-with-structured-response)

5. [Results Comparison and Visualization](#results-comparison-and-visualization)
   - [Compile Results](#1-compile-results)
   - [Generate Comparison Report and Visualizations](#2-generate-comparison-report-and-visualizations)
   - [Interactive Dashboard](#3-interactive-dashboard-optional)

6. [Topic Novelty Filtering Approach](#topic-novelty-filtering-approach)
   - [Generate Baseline Topics List with Gemini](#1-generate-baseline-topics-list-with-gemini)
   - [Implement Topic Novelty Scoring](#2-implement-topic-novelty-scoring)
   - [Apply Filtering to All Approaches](#3-apply-filtering-to-all-approaches)
   - [Generate Combined Report with Novelty Metrics](#4-generate-combined-report-with-novelty-metrics)

7. [Evaluation and Metrics](#evaluation-and-metrics)
   - [Topic Coherence and Quality Analysis](#1-topic-coherence-and-quality-analysis)
   - [Performance Metrics](#2-performance-metrics)
   - [Result Overlaps](#3-result-overlaps)

8. [Out of Scope: Advanced Approaches](#out-of-scope-advanced-approaches)
   - [Full Reinforcement Learning Approach](#1-full-reinforcement-learning-approach)
   - [Negative-Only Learning Approach](#2-negative-only-learning-approach)
   - [Generator-Critic LLM Approach](#3-generator-critic-llm-approach)

9. [Next Steps](#next-steps)