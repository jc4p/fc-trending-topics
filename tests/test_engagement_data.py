import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import the data preprocessing function
from src.data_preprocessing import main as preprocess_data

def test_engagement_data():
    """
    Test script to debug engagement metrics in the preprocessed data
    """
    print("Loading and preprocessing data...")
    
    # Check if interim data exists to save processing time
    if os.path.exists('output/interim_data/cleaned_data.parquet'):
        print("Loading from saved interim data...")
        try:
            recent_df = pd.read_parquet('output/interim_data/cleaned_data.parquet')
            print(f"Loaded preprocessed data with {len(recent_df)} rows")
        except Exception as e:
            print(f"Error loading interim data: {e}")
            print("Preprocessing from scratch...")
            _, recent_df = preprocess_data(save_interim_data=True)
    else:
        print("No interim data found, preprocessing from scratch...")
        _, recent_df = preprocess_data(save_interim_data=True)
    
    # Basic dataset info
    print(f"\nDataset Information:")
    print(f"Total rows: {len(recent_df):,}")
    print(f"Columns: {recent_df.columns.tolist()}")
    
    # Check engagement columns
    print("\nEngagement Column Statistics:")
    engagement_cols = ['total_reactions', 'likes_count', 'recasts_count', 'engagement_score']
    
    for col in engagement_cols:
        if col in recent_df.columns:
            print(f"\n{col}:")
            print(f"  - Data type: {recent_df[col].dtype}")
            print(f"  - Min: {recent_df[col].min()}")
            print(f"  - Max: {recent_df[col].max()}")
            print(f"  - Mean: {recent_df[col].mean():.2f}")
            print(f"  - Median: {recent_df[col].median()}")
            print(f"  - Sum: {recent_df[col].sum()}")
            print(f"  - Zeros: {(recent_df[col] == 0).sum()} ({(recent_df[col] == 0).sum() / len(recent_df) * 100:.1f}%)")
            print(f"  - Nulls: {recent_df[col].isna().sum()} ({recent_df[col].isna().sum() / len(recent_df) * 100:.1f}%)")
        else:
            print(f"\n{col}: Not found in dataset")
    
    # Show sample of high-engagement posts
    if 'engagement_score' in recent_df.columns:
        print("\nTop 5 Posts by Engagement:")
        top_engagement = recent_df.nlargest(5, 'engagement_score')
        for i, row in top_engagement.iterrows():
            print(f"\nEngagement: {row['engagement_score']:.1f} | Likes: {row['likes_count']} | Recasts: {row['recasts_count']}")
            print(f"Text: {row['Text'][:100]}...")
    
    # Check clustering sample
    print("\nSimulating clustering with sample data:")
    sample_df = recent_df.sample(min(1000, len(recent_df)))
    
    # Create a mock cluster
    sample_df['cluster'] = np.random.randint(0, 5, size=len(sample_df))
    
    # Calculate cluster engagement stats
    print("\nCluster Engagement Statistics:")
    for cluster_id in sample_df['cluster'].unique():
        cluster_data = sample_df[sample_df['cluster'] == cluster_id]
        print(f"\nCluster {cluster_id}:")
        print(f"  - Size: {len(cluster_data)} documents")
        print(f"  - Total likes: {int(cluster_data['likes_count'].sum())}")
        print(f"  - Total recasts: {int(cluster_data['recasts_count'].sum())}")
        print(f"  - Avg likes: {float(cluster_data['likes_count'].mean()):.2f}")
        print(f"  - Avg recasts: {float(cluster_data['recasts_count'].mean()):.2f}")
        print(f"  - Avg engagement: {float(cluster_data['engagement_score'].mean()):.2f}")
    
    print("\nDebug complete!")

def check_reactions_data():
    """
    Directly check the reactions Parquet file to see if there's data
    """
    print("\nChecking reactions data file directly...")
    
    try:
        import duckdb
        
        # Create a connection
        conn = duckdb.connect(database=':memory:')
        
        # Try to read the reactions file
        try:
            print("Trying to read reactions file...")
            reactions = conn.execute("SELECT * FROM read_parquet('farcaster_reactions.parquet') LIMIT 10").df()
            print(f"Successfully read reactions file with columns: {reactions.columns.tolist()}")
            print(f"Sample rows: {len(reactions)}")
            
            # Count total reactions
            total_count = conn.execute("SELECT COUNT(*) FROM read_parquet('farcaster_reactions.parquet')").fetchone()[0]
            print(f"Total reactions in file: {total_count:,}")
            
            # Check types of reactions
            reaction_types = conn.execute("""
                SELECT ReactionType, COUNT(*) as count
                FROM read_parquet('farcaster_reactions.parquet')
                GROUP BY ReactionType
                ORDER BY count DESC
            """).df()
            
            print("\nReaction types:")
            print(reaction_types)
            
            # Check timestamps
            timestamp_stats = conn.execute("""
                SELECT 
                    MIN(CAST("Timestamp" AS BIGINT)) as min_ts,
                    MAX(CAST("Timestamp" AS BIGINT)) as max_ts
                FROM read_parquet('farcaster_reactions.parquet')
            """).fetchone()
            
            print("\nTimestamp range:")
            min_date = conn.execute(f"SELECT TIMESTAMP '2021-01-01 00:00:00' + ({timestamp_stats[0]} * INTERVAL '1 second')").fetchone()[0]
            max_date = conn.execute(f"SELECT TIMESTAMP '2021-01-01 00:00:00' + ({timestamp_stats[1]} * INTERVAL '1 second')").fetchone()[0]
            print(f"Min: {timestamp_stats[0]} ({min_date})")
            print(f"Max: {timestamp_stats[1]} ({max_date})")
            
        except Exception as e:
            print(f"Error reading reactions file: {e}")
            
            # Check if file exists
            import os
            if os.path.exists('farcaster_reactions.parquet'):
                print("File exists but could not be read properly")
                
                # Try to get file info
                file_size = os.path.getsize('farcaster_reactions.parquet')
                print(f"File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
            else:
                print("Reactions file does not exist!")
        
    except Exception as e:
        print(f"Error in checking reactions data: {e}")

if __name__ == "__main__":
    test_engagement_data()
    check_reactions_data()