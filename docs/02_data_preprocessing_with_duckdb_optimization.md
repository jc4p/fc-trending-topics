## Data Preprocessing with DuckDB Optimization

1. **Load and Filter Data with DuckDB**
   ```python
   import pandas as pd
   import numpy as np
   import duckdb
   from datetime import datetime, timedelta
   import time

   # Start timing for performance metrics
   start_time = time.time()
   
   # Initialize DuckDB connection with appropriate memory settings
   # Leveraging the 200GB RAM capacity
   conn = duckdb.connect(database=':memory:')
   conn.execute("SET memory_limit='180GB'")  # Reserve some memory for other processes
   conn.execute("SET temp_directory='/tmp'")  # Set temp directory for spilling
   
   print("Loading data with DuckDB...")
   
   # Register parquet files with DuckDB
   # This is more efficient than loading into pandas first
   conn.execute("CREATE VIEW casts AS SELECT * FROM read_parquet('casts.parquet')")
   conn.execute("CREATE VIEW reactions AS SELECT * FROM read_parquet('farcaster_reactions.parquet')")
   
   # Create indexes for faster joins
   conn.execute("CREATE INDEX cast_hash_idx ON casts(Hash)")
   conn.execute("CREATE INDEX reaction_target_idx ON reactions(TargetCastId)")
   
   # Convert Farcaster timestamp to datetime within DuckDB
   # This is much faster than applying the conversion in pandas
   conn.execute("""
   CREATE VIEW casts_with_datetime AS 
   SELECT *, 
          TIMESTAMP '2021-01-01 00:00:00' + (CAST("Timestamp" AS BIGINT) * INTERVAL '1 second') AS datetime
   FROM casts
   """)
   
   conn.execute("""
   CREATE VIEW reactions_with_datetime AS 
   SELECT *, 
          TIMESTAMP '2021-01-01 00:00:00' + (CAST("Timestamp" AS BIGINT) * INTERVAL '1 second') AS datetime
   FROM reactions
   """)
   
   # Find the most recent timestamp in the casts dataset
   max_timestamp = conn.execute("""
   SELECT MAX(datetime) FROM casts_with_datetime
   """).fetchone()[0]
   
   # Calculate time threshold (48 hours before latest timestamp)
   time_threshold = max_timestamp - timedelta(hours=48)
   
   print(f"Most recent data timestamp: {max_timestamp}")
   print(f"Analyzing data from {time_threshold} to {max_timestamp}")
   
   # Count datasets
   cast_count = conn.execute(f"""
   SELECT COUNT(*) FROM casts_with_datetime 
   WHERE datetime >= '{time_threshold}'
   """).fetchone()[0]
   
   reaction_count = conn.execute(f"""
   SELECT COUNT(*) FROM reactions_with_datetime 
   WHERE datetime >= '{time_threshold}'
   """).fetchone()[0]
   
   print(f"Dataset size for analysis: {cast_count:,} posts, {reaction_count:,} reactions")
   
   # Create a filtered view of recent casts and reactions
   conn.execute(f"""
   CREATE VIEW recent_casts AS
   SELECT * FROM casts_with_datetime
   WHERE datetime >= '{time_threshold}'
   """)
   
   conn.execute(f"""
   CREATE VIEW recent_reactions AS
   SELECT * FROM reactions_with_datetime
   WHERE datetime >= '{time_threshold}'
   """)
   
   # Calculate engagement metrics with a single SQL query
   # This is much more efficient than pandas groupby operations
   conn.execute("""
   CREATE VIEW engagement_metrics AS
   SELECT 
       c.Hash,
       c.Fid,
       c.Text,
       c.datetime,
       COALESCE(r.total_reactions, 0) AS total_reactions,
       COALESCE(r.likes_count, 0) AS likes_count,
       COALESCE(r.recasts_count, 0) AS recasts_count,
       COALESCE(r.likes_count, 0) + (3 * COALESCE(r.recasts_count, 0)) AS engagement_score
   FROM recent_casts c
   LEFT JOIN (
       SELECT 
           TargetCastId,
           COUNT(*) AS total_reactions,
           SUM(CASE WHEN ReactionType = 'Like' THEN 1 ELSE 0 END) AS likes_count,
           SUM(CASE WHEN ReactionType = 'Recast' THEN 1 ELSE 0 END) AS recasts_count
       FROM recent_reactions
       GROUP BY TargetCastId
   ) r ON c.Hash = r.TargetCastId
   """)
   
   # Convert to pandas for compatibility with further processing
   # Only pull the data we need into memory
   print("Converting to pandas for further processing...")
   recent_df = conn.execute("""
   SELECT * FROM engagement_metrics
   """).df()
   
   # Add any additional columns needed for analysis
   # Clean and process data as needed
   recent_df['Text'] = recent_df['Text'].fillna('')
   
   # To efficiently retrieve reactions data later
   def get_reactions_for_cast(cast_hash):
       """Fast retrieval of reactions for a specific cast using DuckDB"""
       return conn.execute(f"""
       SELECT * FROM recent_reactions
       WHERE TargetCastId = '{cast_hash}'
       """).df()
   
   print(f"Data preprocessing complete - took {time.time() - start_time:.2f} seconds")
   print(f"Processed {len(recent_df):,} casts with engagement metrics")
   
   # Optional: Compute summary stats with DuckDB
   engagement_stats = conn.execute("""
   SELECT 
       AVG(engagement_score) AS avg_engagement,
       MEDIAN(engagement_score) AS median_engagement,
       MAX(engagement_score) AS max_engagement,
       PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY engagement_score) AS p95_engagement,
       SUM(CASE WHEN likes_count > 0 OR recasts_count > 0 THEN 1 ELSE 0 END) AS casts_with_engagement,
       COUNT(*) AS total_casts
   FROM engagement_metrics
   """).fetchall()
   
   print(f"Engagement statistics:")
   print(f"  - Average engagement score: {engagement_stats[0][0]:.2f}")
   print(f"  - Median engagement score: {engagement_stats[0][1]:.2f}")
   print(f"  - 95th percentile engagement: {engagement_stats[0][3]:.2f}")
   print(f"  - Casts with any engagement: {engagement_stats[0][4]:,} ({engagement_stats[0][4]/engagement_stats[0][5]*100:.1f}%)")
   ```

2. **Clean Text Data with Parallel Processing**
   ```python
   import re
   from concurrent.futures import ProcessPoolExecutor
   import multiprocessing
   
   # Define text cleaning function
   def clean_text(text):
       if pd.isna(text) or text == "":
           return ""
       # Remove URLs, mentions, special characters
       text = re.sub(r'http\S+', '', text)
       text = re.sub(r'@\w+', '', text)
       text = re.sub(r'[^\w\s]', '', text)
       return text.lower().strip()
   
   # Use DuckDB for text cleaning if the dataset is small enough
   # This leverages DuckDB's parallel processing capabilities
   if len(recent_df) < 10_000_000:  # Threshold for in-database processing
       print("Cleaning text with DuckDB...")
       
       # Define SQL UDF for text cleaning
       conn.create_function('clean_text_udf', clean_text)
       
       # Apply cleaning in DuckDB
       conn.execute("""
       CREATE OR REPLACE VIEW clean_texts AS
       SELECT 
           *,
           CASE 
               WHEN "Text" IS NULL OR "Text" = '' THEN ''
               ELSE clean_text_udf("Text")
           END AS cleaned_text
       FROM engagement_metrics
       """)
       
       # Get data with cleaned text
       recent_df = conn.execute("""
       SELECT * FROM clean_texts
       WHERE cleaned_text != ''
       """).df()
       
   else:
       # For very large datasets, use multiprocessing
       print("Cleaning text with multiprocessing...")
       
       # Determine optimal chunk size and number of workers
       num_cores = multiprocessing.cpu_count()
       chunk_size = max(1000, len(recent_df) // (num_cores * 10))
       
       # Split the dataframe into chunks
       text_chunks = [recent_df['Text'].iloc[i:i + chunk_size] 
                     for i in range(0, len(recent_df), chunk_size)]
       
       # Process chunks in parallel
       start_time = time.time()
       with ProcessPoolExecutor(max_workers=num_cores) as executor:
           cleaned_chunks = list(executor.map(
               lambda chunk: chunk.apply(clean_text), 
               text_chunks
           ))
       
       # Combine results
       recent_df['cleaned_text'] = pd.concat(cleaned_chunks)
       recent_df = recent_df[recent_df['cleaned_text'] != ""]
       
       print(f"Text cleaning complete - took {time.time() - start_time:.2f} seconds")
   
   print(f"Retained {len(recent_df):,} posts after text cleaning")
   
   # Calculate basic text stats for analysis
   text_stats = conn.execute("""
   SELECT 
       AVG(LENGTH("Text")) AS avg_length,
       MEDIAN(LENGTH("Text")) AS median_length,
       MAX(LENGTH("Text")) AS max_length,
       MIN(LENGTH("Text")) AS min_length
   FROM recent_casts
   WHERE "Text" IS NOT NULL AND "Text" != ''
   """).fetchone()
   
   print(f"Text statistics:")
   print(f"  - Average length: {text_stats[0]:.1f} chars")
   print(f"  - Median length: {text_stats[1]:.1f} chars")
   print(f"  - Range: {text_stats[3]} to {text_stats[2]} chars")
   ```

3. **Create Initial Metrics with DuckDB**
   ```python
   # Use DuckDB for efficient metric computation
   print("Computing metrics with DuckDB...")
   
   # Register the cleaned dataframe back to DuckDB for further processing
   conn.register('cleaned_casts', recent_df)
   
   # User activity metrics - much faster with DuckDB
   user_metrics = conn.execute("""
   SELECT
       Fid,
       COUNT(*) AS cast_count,
       AVG(engagement_score) AS avg_engagement,
       SUM(engagement_score) AS total_engagement,
       MIN(datetime) AS first_cast,
       MAX(datetime) AS last_cast
   FROM cleaned_casts
   GROUP BY Fid
   ORDER BY cast_count DESC
   """).df()
   
   # Identify conversation patterns and reply chains
   reply_metrics = conn.execute("""
   SELECT
       COUNT(*) AS total_posts,
       SUM(CASE WHEN "ParentCastId" IS NOT NULL THEN 1 ELSE 0 END) AS reply_count,
       (SUM(CASE WHEN "ParentCastId" IS NOT NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*)) AS reply_percentage,
       -- Top replied-to posts
       (
           SELECT COUNT(DISTINCT "ParentCastId")
           FROM cleaned_casts
           WHERE "ParentCastId" IS NOT NULL
       ) AS unique_parent_count
   FROM cleaned_casts
   """).fetchone()
   
   # Extract reply chains for analysis
   reply_df = conn.execute("""
   SELECT *
   FROM cleaned_casts
   WHERE "ParentCastId" IS NOT NULL
   """).df()
   
   # Find conversation clusters (threads with multiple replies)
   conversation_clusters = conn.execute("""
   WITH reply_counts AS (
       SELECT
           "ParentCastId" AS parent_id,
           COUNT(*) AS reply_count
       FROM cleaned_casts
       WHERE "ParentCastId" IS NOT NULL
       GROUP BY "ParentCastId"
       HAVING COUNT(*) > 1
       ORDER BY COUNT(*) DESC
       LIMIT 100
   )
   SELECT
       r.parent_id,
       r.reply_count,
       p.Text AS parent_text,
       p.Fid AS parent_user,
       p.datetime AS parent_time,
       p.engagement_score AS parent_engagement
   FROM reply_counts r
   LEFT JOIN cleaned_casts p ON r.parent_id = p.Hash
   ORDER BY r.reply_count DESC
   """).df()
   
   print(f"Reply metrics:")
   print(f"  - Total posts: {reply_metrics[0]:,}")
   print(f"  - Reply percentage: {reply_metrics[2]:.1f}%")
   print(f"  - Unique posts replied to: {reply_metrics[3]:,}")
   print(f"  - Found {len(conversation_clusters)} major conversation threads")
   
   # Time-based activity analysis
   time_metrics = conn.execute("""
   WITH hourly_counts AS (
       SELECT
           DATE_TRUNC('hour', datetime) AS hour,
           COUNT(*) AS post_count,
           AVG(engagement_score) AS avg_engagement,
           SUM(CASE WHEN "ParentCastId" IS NOT NULL THEN 1 ELSE 0 END) AS reply_count
       FROM cleaned_casts
       GROUP BY DATE_TRUNC('hour', datetime)
       ORDER BY DATE_TRUNC('hour', datetime)
   )
   SELECT
       hour,
       post_count,
       avg_engagement,
       reply_count,
       reply_count * 100.0 / post_count AS reply_percentage
   FROM hourly_counts
   ORDER BY hour
   """).df()
   
   # Keep these metrics available for later topic analysis
   ```