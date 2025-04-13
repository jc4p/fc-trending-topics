"""
Test script to debug reply metrics calculation issues.
This script directly interfaces with the data processing pipeline
to isolate and fix the reply percentage calculation problem.
"""

import sys
import os
import pandas as pd
import duckdb
import time
from datetime import datetime, timedelta

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def test_reply_metrics():
    """Test and debug the reply metrics calculation."""
    print("\n=== TESTING REPLY METRICS CALCULATION ===")
    
    # Initialize DuckDB connection
    conn = duckdb.connect(database=':memory:')
    conn.execute("SET memory_limit='180GB'")  # Reserve some memory for other processes
    conn.execute("SET temp_directory='/tmp'")  # Set temp directory for spilling
    
    # Register parquet files with DuckDB
    print("Loading data with DuckDB...")
    
    # Register parquet files - use absolute paths
    conn.execute("CREATE VIEW casts AS SELECT * FROM read_parquet('/home/ubuntu/fc-trending-topics/casts.parquet')")
    conn.execute("CREATE VIEW reactions AS SELECT * FROM read_parquet('/home/ubuntu/fc-trending-topics/farcaster_reactions.parquet')")
    
    # Convert Farcaster timestamp to datetime within DuckDB
    conn.execute("""
    CREATE VIEW casts_with_datetime AS 
    SELECT *, 
           TIMESTAMP '2021-01-01 00:00:00' + (CAST("Timestamp" AS BIGINT) * INTERVAL '1 second') AS datetime
    FROM casts
    """)
    
    # Find the most recent timestamp in the casts dataset
    max_timestamp = conn.execute("""
    SELECT MAX(datetime) FROM casts_with_datetime
    """).fetchone()[0]
    
    # Calculate time threshold (96 hours before latest timestamp)
    time_threshold = max_timestamp - timedelta(hours=96)
    
    print(f"Most recent data timestamp: {max_timestamp}")
    print(f"Analyzing data from {time_threshold} to {max_timestamp}")
    
    # Create a filtered view of recent casts
    conn.execute(f"""
    CREATE VIEW recent_casts AS
    SELECT * FROM casts_with_datetime
    WHERE datetime >= '{time_threshold}'
    """)
    
    # Count total posts
    total_posts = conn.execute("""
    SELECT COUNT(*) FROM recent_casts
    """).fetchone()[0]
    
    print(f"Total posts in filtered dataset: {total_posts}")
    
    # Count posts with ParentCastId set (direct count)
    reply_count_direct = conn.execute("""
    SELECT COUNT(*) 
    FROM recent_casts
    WHERE ParentCastId IS NOT NULL AND LENGTH(ParentCastId) > 0
    """).fetchone()[0]
    
    # Calculate reply percentage
    reply_pct = (reply_count_direct * 100.0 / total_posts) if total_posts > 0 else 0.0
    
    print(f"\nDirect counts:")
    print(f"  - Reply posts: {reply_count_direct}")
    print(f"  - Total posts: {total_posts}")
    print(f"  - Reply percentage: {reply_pct:.2f}%")
    
    # Get a sample of ParentCastId values to inspect
    sample_parents = conn.execute("""
    SELECT ParentCastId, COUNT(*) as reply_count
    FROM recent_casts
    WHERE ParentCastId IS NOT NULL AND LENGTH(ParentCastId) > 0
    GROUP BY ParentCastId
    ORDER BY reply_count DESC
    LIMIT 5
    """).df()
    
    print(f"\nSample ParentCastId values (most replied to):")
    print(sample_parents)
    
    # Extract parent hash from ParentCastId
    conn.execute("""
    CREATE VIEW casts_with_parent_hash AS
    SELECT 
        *,
        CASE 
            WHEN POSITION(':' IN ParentCastId) > 0 
            THEN SUBSTRING(ParentCastId, POSITION(':' IN ParentCastId) + 1)
            ELSE ParentCastId 
        END AS parent_hash
    FROM recent_casts
    """)
    
    # Count distinct parent hashes
    distinct_parents = conn.execute("""
    SELECT COUNT(DISTINCT parent_hash)
    FROM casts_with_parent_hash
    WHERE parent_hash IS NOT NULL AND parent_hash != ''
    """).fetchone()[0]
    
    print(f"\nDistinct parent posts: {distinct_parents}")
    
    # Now create the final metrics for reports
    reply_metrics = (total_posts, reply_count_direct, reply_pct, distinct_parents)
    
    print(f"\nFinal reply metrics tuple:")
    print(f"  - reply_metrics[0] (total_posts): {reply_metrics[0]}")
    print(f"  - reply_metrics[1] (reply_count): {reply_metrics[1]}")
    print(f"  - reply_metrics[2] (reply_percentage): {reply_metrics[2]:.2f}%")
    print(f"  - reply_metrics[3] (unique_parents): {reply_metrics[3]}")
    
    # Check the metrics reporting function
    print(f"\nSimulated output to user:")
    print(f"  - Total posts: {reply_metrics[0]:,}")
    print(f"  - Reply percentage: {reply_metrics[2]:.1f}%")
    print(f"  - Unique posts replied to: {reply_metrics[3]:,}")
    
    return reply_metrics

if __name__ == "__main__":
    test_reply_metrics()