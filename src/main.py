import os
import time
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import duckdb

from data_preprocessing import main as preprocess_data
from approach1_direct_llm import direct_llm_analysis
from approach2_lda_kmeans import lda_kmeans_clustering
from approach3_embeddings import embeddings_clustering

def main():
    """
    Main function to run the Farcaster Trending Topics Analysis.
    
    This script orchestrates the full pipeline:
    1. Data preprocessing with DuckDB
    2. Run all three approaches to identify trending topics
    3. Compare results and generate summary report
    
    Each step caches its results to disk to enable resuming from failures.
    """
    print("Starting Farcaster Trending Topics Analysis...")
    overall_start_time = time.time()
    
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    os.makedirs('output/cache', exist_ok=True)
    
    # Step 1: Data Preprocessing
    print("\n=== STEP 1: DATA PREPROCESSING ===")
    parquet_cache_path = 'output/interim_data/cleaned_data.parquet'
    
    if os.path.exists(parquet_cache_path):
        print("Loading preprocessed data from parquet cache...")
        recent_df = pd.read_parquet(parquet_cache_path)
        # Reconnect to DuckDB and register the dataframe
        conn = duckdb.connect(database=':memory:')
        conn.execute("SET memory_limit='180GB'")
        conn.register('cleaned_casts', recent_df)
        
        # Print a summary of the loaded data
        print(f"\nCached Data Summary:")
        print(f"  - Total posts: {len(recent_df):,}")
        print(f"  - Date range: {recent_df['datetime'].min()} to {recent_df['datetime'].max()}")
        print(f"  - Average engagement score: {recent_df['engagement_score'].mean():.2f}")
        
        # Load other cached metrics if available
        if os.path.exists('output/interim_data/user_metrics.parquet'):
            user_metrics = pd.read_parquet('output/interim_data/user_metrics.parquet')
            print(f"  - Unique users: {len(user_metrics):,}")
    else:
        # Run preprocessing with parquet saving enabled
        conn, recent_df = preprocess_data(save_interim_data=True)
    
    # Step 2: Run approach 1 - Direct LLM Analysis
    print("\n=== STEP 2: APPROACH 1 - DIRECT LLM ANALYSIS ===")
    cache_path = 'output/cache/approach1.json'
    
    if os.path.exists(cache_path):
        print("Loading approach 1 results from cache...")
        with open(cache_path, 'r') as f:
            approach1_results = json.load(f)
        
        # Print a summary of the cached approach 1 results
        print(f"\nApproach 1 (Direct LLM) Cache Summary:")
        print(f"  - Total topics identified: {len(approach1_results.get('topics', []))}")
        print(f"  - Analysis period: {approach1_results.get('analysis_period', 'N/A')}")
        print(f"  - Total posts analyzed: {approach1_results.get('total_posts_analyzed', 'N/A')}")
        if 'topics' in approach1_results and len(approach1_results['topics']) > 0:
            print(f"  - Top topic: {approach1_results['topics'][0]['name']}")
    else:
        approach1_results = direct_llm_analysis(conn, recent_df)
        # Cache the results
        print("Caching approach 1 results...")
        with open(cache_path, 'w') as f:
            json.dump(approach1_results, f, indent=2)
    
    # Step 3: Run approach 2 - LDA + K-Means Clustering
    print("\n=== STEP 3: APPROACH 2 - LDA + K-MEANS CLUSTERING ===")
    
    cache_path = 'output/cache/approach2.json'
    
    if os.path.exists(cache_path):
        print("Loading approach 2 results from cache...")
        with open(cache_path, 'r') as f:
            approach2_results = json.load(f)
            
        # Print a summary of the cached approach 2 results
        print(f"\nApproach 2 (LDA + K-Means) Cache Summary:")
        
        if isinstance(approach2_results, dict) and 'topics' in approach2_results:
            # Handle the standard format returned by lda_kmeans_clustering function
            print(f"  - Total topics identified: {len(approach2_results['topics'])}")
            if len(approach2_results['topics']) > 0:
                print(f"  - Top topic: {approach2_results['topics'][0]['name']}")
                print(f"  - Analysis period: {approach2_results['analysis_period']}")
        else:
            # Old cache format (list of cluster topics)
            print(f"  - Total topics identified: {len(approach2_results)}")
            if len(approach2_results) > 0 and isinstance(approach2_results, list) and 'topic_data' in approach2_results[0]:
                print(f"  - Top topic: {approach2_results[0]['topic_data']['topic_name']}")
                print(f"  - Topic size: {approach2_results[0]['size']} posts")
            else:
                print("  - Cache format not recognized, will regenerate results")
    else:
        # Run the clustering algorithm
        approach2_results = lda_kmeans_clustering(recent_df)
        
        # Cache the results
        print("Caching approach 2 results...")
        with open(cache_path, 'w') as f:
            json.dump(approach2_results, f, indent=2)
    
    # Step 4: Run approach 3 - Embeddings + Clustering
    print("\n=== STEP 4: APPROACH 3 - EMBEDDINGS + CLUSTERING ===")
    cache_path = 'output/cache/approach3.json'
    
    if os.path.exists(cache_path):
        print("Loading approach 3 results from cache...")
        with open(cache_path, 'r') as f:
            approach3_results = json.load(f)
            
        # Print a summary of the cached approach 3 results
        print(f"\nApproach 3 (Embeddings) Cache Summary:")
        print(f"  - Total topics identified: {len(approach3_results)}")
        if len(approach3_results) > 0:
            print(f"  - Top topic: {approach3_results[0]['topic_data']['topic_name']}")
            print(f"  - Top topic sentiment: {approach3_results[0]['topic_data']['sentiment']}")
    else:
        approach3_results = embeddings_clustering(recent_df)
        # Cache the results
        print("Caching approach 3 results...")
        with open(cache_path, 'w') as f:
            json.dump(approach3_results, f, indent=2)
    
    # Step 5: Compare results and create summary report
    print("\n=== STEP 5: COMPARING RESULTS AND CREATING SUMMARY ===")
    compare_results(approach1_results, approach2_results, approach3_results)
    
    overall_duration = time.time() - overall_start_time
    print(f"\nTotal analysis completed in {overall_duration:.2f} seconds ({overall_duration/60:.2f} minutes)")

def compare_results(approach1_results, approach2_results, approach3_results):
    """
    Compare results from all three approaches and create a summary.
    
    Args:
        approach1_results: Results from direct LLM analysis
        approach2_results: Results from LDA + K-Means clustering
        approach3_results: Results from embeddings clustering
    """
    # Create a combined results dictionary
    combined_results = {
        "analysis_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "approaches": {
            "direct_llm": {
                "topics": approach1_results.get("topics", []),
                "count": len(approach1_results.get("topics", [])),
                "analysis_period": approach1_results.get("analysis_period", "")
            },
            "lda_kmeans": {
                "topics": (
                    # If approach2_results is a dict with 'topics' key (standard format)
                    approach2_results.get("topics", []) 
                    if isinstance(approach2_results, dict) and "topics" in approach2_results
                    # If approach2_results is a list of topics with topic_data (old format)
                    else [
                        {
                            "name": topic["topic_data"]["topic_name"] if "topic_data" in topic else "Unknown Topic",
                            "explanation": topic["topic_data"]["explanation"] if "topic_data" in topic else "",
                            "estimated_percentage": topic.get("size", 0),
                            "engagement_level": topic["topic_data"]["engagement_level"] if "topic_data" in topic else "Medium",
                            "sentiment": topic["topic_data"]["sentiment"] if "topic_data" in topic else "Neutral"
                        } 
                        for topic in approach2_results
                    ] if isinstance(approach2_results, list)
                    # Fallback for empty or unexpected format
                    else []
                ),
                "count": (
                    len(approach2_results.get("topics", [])) 
                    if isinstance(approach2_results, dict) and "topics" in approach2_results
                    else len(approach2_results) if isinstance(approach2_results, list)
                    else 0
                )
            },
            "embeddings": {
                "topics": (
                    # If approach3_results is a list of topics with the expected format
                    [
                        {
                            "name": topic["topic_data"]["topic_name"] if "topic_data" in topic else "Unknown Topic",
                            "explanation": topic["topic_data"]["explanation"] if "topic_data" in topic else "",
                            "key_terms": topic["topic_data"].get("key_terms", []) if "topic_data" in topic else [],
                            "engagement_insight": topic["topic_data"].get("engagement_insight", "") if "topic_data" in topic else "",
                            "sentiment": topic["topic_data"].get("sentiment", "Neutral") if "topic_data" in topic else "Neutral"
                        } 
                        for topic in approach3_results
                    ] if isinstance(approach3_results, list)
                    # If it's a dict with topics key
                    else approach3_results.get("topics", []) if isinstance(approach3_results, dict) and "topics" in approach3_results
                    # Fallback for empty or unexpected format
                    else []
                ),
                "count": (
                    len(approach3_results) if isinstance(approach3_results, list)
                    else len(approach3_results.get("topics", [])) if isinstance(approach3_results, dict) and "topics" in approach3_results
                    else 0
                )
            }
        }
    }
    
    # Save combined results
    with open("output/combined_results.json", "w") as f:
        json.dump(combined_results, f, indent=2)
    
    # Print summary with improved formatting
    print("\n======== TRENDING TOPICS FINAL SUMMARY ========")
    print(f"Analysis date: {combined_results['analysis_date']}")
    print(f"Analysis period: {combined_results['approaches']['direct_llm']['analysis_period']}")
    print(f"\n==== Approach 1 (Direct LLM) Topics: ====")
    for i, topic in enumerate(combined_results["approaches"]["direct_llm"]["topics"]):
        print(f"{i+1}. {topic['name']}")
        print(f"   - Engagement: {topic['engagement_level']}")
        print(f"   - Est. percentage: {topic['estimated_percentage']}")
        if 'explanation' in topic:
            print(f"   - Why trending: {topic['explanation'][:100]}..." if len(topic['explanation']) > 100 else f"   - Why trending: {topic['explanation']}")
        print()
    
    print(f"\n==== Approach 2 (LDA + K-Means) Topics: ====")
    for i, topic in enumerate(combined_results["approaches"]["lda_kmeans"]["topics"]):
        print(f"{i+1}. {topic['name']}")
        print(f"   - Engagement: {topic['engagement_level']}")
        print(f"   - Est. percentage: {topic['estimated_percentage']}")
        if 'explanation' in topic:
            print(f"   - Why trending: {topic['explanation'][:100]}..." if len(topic['explanation']) > 100 else f"   - Why trending: {topic['explanation']}")
        print()
    
    print(f"\n==== Approach 3 (Embeddings) Topics: ====")
    for i, topic in enumerate(combined_results["approaches"]["embeddings"]["topics"]):
        print(f"{i+1}. {topic['name']}")
        print(f"   - Sentiment: {topic['sentiment']}")
        if 'explanation' in topic:
            print(f"   - Why trending: {topic['explanation'][:100]}..." if len(topic['explanation']) > 100 else f"   - Why trending: {topic['explanation']}")
        print()
    
    # Create a simple visualization comparing topics across approaches
    plt.figure(figsize=(14, 10))
    
    # Set up data for plotting
    approaches = ["Direct LLM", "LDA + K-Means", "Embeddings"]
    topic_counts = [
        combined_results["approaches"]["direct_llm"]["count"],
        combined_results["approaches"]["lda_kmeans"]["count"],
        combined_results["approaches"]["embeddings"]["count"]
    ]
    
    # Plot topic counts
    ax = sns.barplot(x=approaches, y=topic_counts, palette="viridis")
    
    # Add value labels on top of bars
    for i, v in enumerate(topic_counts):
        ax.text(i, v + 0.1, str(v), ha='center', fontsize=10)
    
    plt.title("Number of Topics Identified by Each Approach", fontsize=16, fontweight="bold")
    plt.ylabel("Number of Topics", fontsize=14)
    plt.tight_layout()
    plt.savefig("output/topic_count_comparison.png", dpi=300)
    
    print("\nAnalysis complete! Results saved to 'output' directory.")

if __name__ == "__main__":
    main()