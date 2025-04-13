import pandas as pd
import numpy as np
import duckdb
import os
import sys
from pathlib import Path
import time

"""
This script fixes the engagement metrics in the preprocessed data.
The issue is that the reactions are not being joined to the casts correctly.
"""

def fix_engagement_data():
    print("Starting engagement data fix...")
    start_time = time.time()
    
    # Initialize DuckDB connection for efficient processing
    conn = duckdb.connect(database=':memory:')
    conn.execute("SET memory_limit='180GB'")  # Reserve some memory for other processes
    
    # Check if preprocessed data exists
    if not os.path.exists('output/interim_data/cleaned_data.parquet'):
        print("Error: Preprocessed data not found. Run data_preprocessing.py first.")
        return
    
    # Load the cleaned data
    print("Loading cleaned data...")
    cleaned_df = pd.read_parquet('output/interim_data/cleaned_data.parquet')
    print(f"Loaded {len(cleaned_df):,} posts")
    
    # Register the dataframe with DuckDB
    conn.register('cleaned_posts', cleaned_df)
    
    # Read reactions file
    print("Reading reactions file...")
    conn.execute("CREATE VIEW reactions AS SELECT * FROM read_parquet('farcaster_reactions.parquet')")
    
    # Convert Farcaster timestamp to datetime
    conn.execute("""
    CREATE VIEW reactions_with_datetime AS 
    SELECT *, 
           TIMESTAMP '2021-01-01 00:00:00' + (CAST("Timestamp" AS BIGINT) * INTERVAL '1 second') AS datetime
    FROM reactions
    """)
    
    # Find the most recent timestamp in the posts dataset
    max_timestamp = conn.execute("""
    SELECT MAX(datetime) FROM cleaned_posts
    """).fetchone()[0]
    
    # Calculate time threshold (48 hours before latest timestamp)
    time_threshold = max_timestamp - pd.Timedelta(hours=48)
    
    print(f"Most recent data timestamp: {max_timestamp}")
    print(f"Analyzing reactions from {time_threshold} to {max_timestamp}")
    
    # Create a filtered view of recent reactions
    conn.execute(f"""
    CREATE VIEW recent_reactions AS
    SELECT * FROM reactions_with_datetime
    WHERE datetime >= '{time_threshold}'
    """)
    
    # Count reactions within time window
    reaction_count = conn.execute("""
    SELECT COUNT(*) FROM recent_reactions
    """).fetchone()[0]
    
    print(f"Found {reaction_count:,} reactions in the time window")
    
    # Log some sample target cast IDs to debug
    print("Sample target cast IDs from reactions:")
    sample_targets = conn.execute("""
    SELECT TargetCastId, COUNT(*) as reaction_count
    FROM recent_reactions
    GROUP BY TargetCastId
    ORDER BY reaction_count DESC
    LIMIT 5
    """).df()
    print(sample_targets)
    
    # Log some sample hashes from posts
    print("Sample hashes from posts:")
    sample_hashes = conn.execute("""
    SELECT Hash, COUNT(*) as post_count
    FROM cleaned_posts
    GROUP BY Hash
    LIMIT 5
    """).df()
    print(sample_hashes)
    
    # Check for matching IDs
    print("Checking for matches between reactions and posts...")
    match_count = conn.execute("""
    SELECT COUNT(*) FROM recent_reactions r
    JOIN cleaned_posts p ON r.TargetCastId = p.Hash
    """).fetchone()[0]
    
    print(f"Found {match_count:,} matching reactions to posts")
    
    # Try a different joining approach
    print("Trying alternative join keys...")
    
    # Check if Hash_1 could be the matching field
    if 'Hash_1' in conn.execute("SELECT * FROM recent_reactions LIMIT 1").df().columns:
        hash1_match_count = conn.execute("""
        SELECT COUNT(*) FROM recent_reactions r
        JOIN cleaned_posts p ON r.Hash_1 = p.Hash
        """).fetchone()[0]
        print(f"Hash_1 matches: {hash1_match_count:,}")
    
    # Check if case sensitivity might be an issue
    upper_match_count = conn.execute("""
    SELECT COUNT(*) FROM recent_reactions r
    JOIN cleaned_posts p ON UPPER(r.TargetCastId) = UPPER(p.Hash)
    """).fetchone()[0]
    
    print(f"Case-insensitive matches: {upper_match_count:,}")
    
    # Calculate engagement metrics with the working join condition
    print("Calculating corrected engagement metrics...")
    
    # Determine which join condition works best
    join_condition = "r.TargetCastId = p.Hash"  # Default
    if match_count < upper_match_count:
        join_condition = "UPPER(r.TargetCastId) = UPPER(p.Hash)"
        print(f"Using case-insensitive join: {join_condition}")
    
    # Calculate engagement metrics
    conn.execute(f"""
    CREATE VIEW corrected_metrics AS
    SELECT 
        p.Hash,
        p.datetime,
        p.Text,
        p.cleaned_text,
        p.Fid,
        COALESCE(r.total_reactions, 0) AS total_reactions,
        COALESCE(r.likes_count, 0) AS likes_count,
        COALESCE(r.recasts_count, 0) AS recasts_count,
        COALESCE(r.likes_count, 0) + (3 * COALESCE(r.recasts_count, 0)) AS engagement_score
    FROM cleaned_posts p
    LEFT JOIN (
        SELECT 
            TargetCastId,
            COUNT(*) AS total_reactions,
            SUM(CASE WHEN ReactionType = 'Like' THEN 1 ELSE 0 END) AS likes_count,
            SUM(CASE WHEN ReactionType = 'Recast' THEN 1 ELSE 0 END) AS recasts_count
        FROM recent_reactions
        GROUP BY TargetCastId
    ) r ON {join_condition}
    """)
    
    # Create updated dataframe
    corrected_df = conn.execute("""
    SELECT * FROM corrected_metrics
    """).df()
    
    # Show stats on corrected metrics
    print("\nCorrected Engagement Statistics:")
    for col in ['total_reactions', 'likes_count', 'recasts_count', 'engagement_score']:
        print(f"\n{col}:")
        print(f"  - Min: {corrected_df[col].min()}")
        print(f"  - Max: {corrected_df[col].max()}")
        print(f"  - Mean: {corrected_df[col].mean():.2f}")
        print(f"  - Sum: {corrected_df[col].sum()}")
        print(f"  - Zeros: {(corrected_df[col] == 0).sum()} ({(corrected_df[col] == 0).sum() / len(corrected_df) * 100:.1f}%)")
    
    # Show top 5 posts by engagement
    print("\nTop 5 Posts by Corrected Engagement:")
    top_posts = corrected_df.nlargest(5, 'engagement_score')
    for i, row in top_posts.iterrows():
        print(f"\nEngagement: {row['engagement_score']:.1f} | Likes: {row['likes_count']} | Recasts: {row['recasts_count']}")
        print(f"Text: {row['Text'][:100]}...")
    
    # Save corrected dataframe
    print("\nSaving corrected data...")
    os.makedirs('output/interim_data', exist_ok=True)
    corrected_df.to_parquet('output/interim_data/cleaned_data_fixed.parquet', index=False)
    
    print(f"\nFix completed in {time.time() - start_time:.2f} seconds.")
    print(f"Corrected data saved to output/interim_data/cleaned_data_fixed.parquet")
    print("To use this fixed data, update the main.py file to load from this file instead.")

if __name__ == "__main__":
    fix_engagement_data()