#!/usr/bin/env python3
"""
Merge Generator-Critic Results

This script merges results from multiple chunks processed with the generator-critic approach.
Use this when you've processed a large dataset in chunks with different output prefixes.

Usage:
  python src/merge_gc_results.py --chunks chunk1,chunk2,chunk3 --output combined
  
Options:
  --chunks     Comma-separated list of chunk prefixes (required)
  --output     Output prefix for combined results (default: "combined")
  --keep-all   Keep all topics without additional filtering (default: false)
  --top        Number of top topics to keep from each chunk (default: 5)
"""

import os
import json
import argparse
from typing import List, Dict, Any
import datetime


def merge_gc_results(chunk_prefixes: List[str], output_prefix: str, keep_all: bool = False, top_n: int = 5) -> None:
    """
    Merge results from multiple generator-critic chunks.
    
    Args:
        chunk_prefixes: List of chunk prefixes to merge
        output_prefix: Prefix for output files
        keep_all: Whether to keep all topics without filtering
        top_n: Number of top topics to keep from each chunk if not keep_all
    """
    print(f"Merging results from {len(chunk_prefixes)} chunks: {', '.join(chunk_prefixes)}")
    
    # Storage for combined data
    all_topics = []
    all_evaluated_topics = []
    total_posts = 0
    date_periods = []
    
    # Load data from each chunk
    for prefix in chunk_prefixes:
        # Define filenames based on prefix
        results_file = f"output/{prefix}_approach4_results.json"
        raw_file = f"output/{prefix}_generator_critic_raw.json"
        
        # Check if files exist
        if not os.path.exists(results_file):
            print(f"Warning: Results file not found for chunk '{prefix}': {results_file}")
            continue
            
        # Load results file
        with open(results_file, 'r') as f:
            chunk_results = json.load(f)
            
        # Extract data
        chunk_topics = chunk_results.get("topics", [])
        chunk_posts = chunk_results.get("total_posts_analyzed", 0)
        chunk_period = chunk_results.get("analysis_period", "")
        
        # Load raw file with evaluated topics if it exists
        chunk_evaluated = []
        if os.path.exists(raw_file):
            try:
                with open(raw_file, 'r') as f:
                    raw_data = json.load(f)
                    chunk_evaluated = raw_data.get("evaluated_topics", [])
                    print(f"Loaded {len(chunk_evaluated)} evaluated topics from {prefix}")
            except Exception as e:
                print(f"Warning: Error loading raw data for chunk '{prefix}': {e}")
        
        # Filter to top N topics if requested
        if not keep_all and top_n > 0 and len(chunk_topics) > top_n:
            print(f"Limiting chunk '{prefix}' from {len(chunk_topics)} to top {top_n} topics")
            chunk_topics = chunk_topics[:top_n]
        
        # Add to combined data
        all_topics.extend(chunk_topics)
        all_evaluated_topics.extend(chunk_evaluated)
        total_posts += chunk_posts
        if chunk_period:
            date_periods.append(chunk_period)
    
    # Create combined period
    if date_periods:
        # Find earliest and latest dates from all periods
        all_dates = []
        for period in date_periods:
            try:
                dates = period.split(" to ")
                if len(dates) == 2:
                    all_dates.extend(dates)
            except:
                continue
                
        # If we have dates, find min and max
        if all_dates:
            try:
                # Try to parse dates
                parsed_dates = [datetime.datetime.strptime(date.strip(), "%Y-%m-%d") for date in all_dates]
                min_date = min(parsed_dates).strftime("%Y-%m-%d")
                max_date = max(parsed_dates).strftime("%Y-%m-%d")
                combined_period = f"{min_date} to {max_date}"
            except:
                # Fallback if date parsing fails
                combined_period = " and ".join(date_periods)
        else:
            combined_period = " and ".join(date_periods)
    else:
        combined_period = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # Create combined results
    combined_results = {
        "topics": all_topics,
        "analysis_period": combined_period,
        "total_posts_analyzed": total_posts,
        "source_chunks": chunk_prefixes
    }
    
    # Create combined raw data
    combined_raw = {
        "evaluated_topics": all_evaluated_topics,
        "final_results": combined_results,
        "merge_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Create output directory if needed
    os.makedirs("output", exist_ok=True)
    
    # Save combined results
    combined_file = f"output/{output_prefix}_approach4_results.json"
    combined_raw_file = f"output/{output_prefix}_generator_critic_raw.json"
    
    with open(combined_file, 'w') as f:
        json.dump(combined_results, f, indent=2)
        
    with open(combined_raw_file, 'w') as f:
        json.dump(combined_raw, f, indent=2)
    
    print(f"Merged {len(all_topics)} topics from {len(chunk_prefixes)} chunks")
    print(f"Total posts analyzed: {total_posts}")
    print(f"Output saved to: {combined_file}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge Generator-Critic results from multiple chunks")
    parser.add_argument("--chunks", type=str, required=True, 
                        help="Comma-separated list of chunk prefixes to merge")
    parser.add_argument("--output", type=str, default="combined",
                        help="Output prefix for combined results (default: 'combined')")
    parser.add_argument("--keep-all", action="store_true",
                        help="Keep all topics without additional filtering")
    parser.add_argument("--top", type=int, default=5,
                        help="Number of top topics to keep from each chunk (default: 5)")
    
    args = parser.parse_args()
    
    # Parse chunk prefixes
    chunk_prefixes = [prefix.strip() for prefix in args.chunks.split(",")]
    
    # Merge results
    merge_gc_results(chunk_prefixes, args.output, args.keep_all, args.top)