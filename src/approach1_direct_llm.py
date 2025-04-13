import random
import time
import json
import pandas as pd
from enum import Enum
from typing_extensions import TypedDict
from datetime import datetime
import os
from google.generativeai import GenerativeModel
from google.generativeai import types

# Define TypedDict classes for structured output
class KeyTerm(TypedDict):
    term: str
    frequency: int  # Estimated frequency

class KeyEntity(TypedDict):
    name: str
    type: str  # Person, Project, Company, etc.
    relevance: str  # High, Medium, Low

class Topic(TypedDict):
    name: str  # 5 words max
    explanation: str  # Brief explanation of why trending
    estimated_percentage: str  # Percentage of posts
    key_terms: list[KeyTerm]
    key_entities: list[KeyEntity]
    engagement_level: str  # High, Medium, Low based on likes/recasts

class TrendingTopics(TypedDict):
    topics: list[Topic]
    analysis_period: str
    total_posts_analyzed: int

def direct_llm_analysis(conn, recent_df):
    """
    Approach 1: Direct LLM Analysis
    
    Uses Gemini to analyze a sample of posts and extract trending topics directly.
    
    Args:
        conn: DuckDB connection
        recent_df: DataFrame with cleaned posts
        
    Returns:
        dict: Structured trending topics result
    """
    print("Selecting optimal sample for LLM analysis...")
    start_time = time.time()
    
    # Increase sample size to better utilize Gemini's context window
    # We'll use a larger sample to ensure proper representation of all content types
    # This gives better context about the active discussions and reduces sampling bias
    # Based on token count feedback, we need to reduce from 16000 to stay under the token limit
    sample_size = min(14000, len(recent_df))  # Using 14000 to stay safely under the token limit
    
    print("Identifying top-level posts and their replies...")
    
    # Check if we have parent/reply information
    has_parent_info = conn.execute("""
    SELECT COUNT(*) > 0 FROM INFORMATION_SCHEMA.COLUMNS 
    WHERE TABLE_NAME = 'cleaned_casts' AND COLUMN_NAME = 'ParentCastId'
    """).fetchone()[0]
    
    # Check if conversation metrics are already available from data preprocessing
    has_conversation_metrics = conn.execute("""
    SELECT COUNT(*) > 0 
    FROM INFORMATION_SCHEMA.COLUMNS 
    WHERE TABLE_NAME = 'cleaned_casts' 
    AND COLUMN_NAME IN ('reply_count', 'unique_repliers', 'conversation_hours')
    """).fetchone()[0]
    
    if has_conversation_metrics:
        print("Using pre-calculated conversation metrics from data preprocessing")
        
        # Calculate the 99th percentile of engagement scores
        p99 = conn.execute("""
        SELECT QUANTILE_CONT(engagement_score, 0.99) AS p99_score
        FROM cleaned_casts
        """).fetchone()[0]
        
        print(f"99th percentile of engagement scores: {p99:.2f}")
        
        # Get min and max datetime for recency calculation
        min_max_dt = conn.execute("""
        SELECT MIN(datetime), MAX(datetime) FROM cleaned_casts
        """).fetchone()
        min_dt, max_dt = min_max_dt
        
        # Calculate epoch seconds for time difference
        time_range_seconds = conn.execute(f"""
        SELECT EXTRACT(EPOCH FROM (TIMESTAMP '{max_dt}' - TIMESTAMP '{min_dt}'))
        """).fetchone()[0]
        
        # First separate top-level posts and replies like Approach 3 does
        conn.execute("""
        CREATE OR REPLACE TEMP TABLE post_types AS
        SELECT
            *,
            CASE WHEN ParentCastId IS NULL OR TRIM(ParentCastId) = '' 
                 THEN FALSE ELSE TRUE END AS is_reply
        FROM cleaned_casts
        """)
        
        # Create index for faster filtering
        conn.execute("""
        CREATE INDEX IF NOT EXISTS post_types_is_reply_idx ON post_types(is_reply)
        """)
        
        # Use the improved conversation metrics formula from Approach 3
        # This better balances the weights between different engagement types
        conn.execute(f"""
        CREATE OR REPLACE TABLE sample_weights AS
        SELECT
            *,
            -- Recency weight (0.1 to 1.1) - pre-calculate time range for better performance
            0.1 + ((EXTRACT(EPOCH FROM (datetime - TIMESTAMP '{min_dt}')) / 
                  {time_range_seconds}))
                AS recency_weight,
            
            -- Engagement weight (0.1 to 1.1)
            -- Cap engagement at 99th percentile to avoid outliers
            0.1 + (LEAST(engagement_score, {p99}) / {p99})
                AS engagement_weight,
                
            -- Conversation weight (0.1 to 1.1) using approach3's formula
            -- Enhanced conversation weighting that puts more emphasis on unique repliers
            0.1 + LEAST((
                reply_count * 5 +  -- Base reply weight (5 points per reply)
                unique_repliers * 10 +  -- More weight on diverse conversation (10 points per unique replier)
                LEAST(conversation_hours, 24) * 2  -- Time component (up to 48 points for 24-hour conversations)
            ) / 500.0, 1.0) AS conversation_weight
        FROM post_types
        WHERE is_reply = FALSE  -- Only focus on top-level posts for analysis, like Approach 3
        """)
        
        # Create an index for faster sorting
        conn.execute("""
        CREATE INDEX IF NOT EXISTS sample_weights_datetime_idx ON sample_weights(datetime)
        """)
        
        # Calculate combined weight with better balance between metrics (approach 3 style)
        conn.execute("""
        CREATE OR REPLACE TABLE combined_weights AS
        SELECT
            *,
            -- Combined weight with Approach 3's improved balancing:
            -- - 30% recency (prioritize recent content but not overly)
            -- - 30% engagement (likes/recasts/etc - still important signals)
            -- - 40% conversation (replies, unique repliers - strongest signal of interesting content)
            (0.3 * recency_weight) + (0.3 * engagement_weight) + (0.4 * conversation_weight) AS combined_weight,
            
            -- Also add a diversity score for later stratified sampling
            -- This will help ensure we get posts from different time periods and engagement levels
            NTILE(10) OVER (ORDER BY datetime) AS time_bucket,
            NTILE(10) OVER (ORDER BY engagement_score) AS engagement_bucket,
            NTILE(5) OVER (ORDER BY COALESCE(reply_count, 0)) AS conversation_bucket
        FROM sample_weights
        """)
        
        # Create an index for sorting by combined weight
        conn.execute("""
        CREATE INDEX IF NOT EXISTS combined_weights_weight_idx ON combined_weights(combined_weight DESC)
        """)
    else:
        print("No ParentCastId information available - using standard sampling")
        
        # Optimize query performance with parallel execution
        conn.execute("""
        PRAGMA threads=30;  -- Use more threads for parallelization
        """)
        
        # First calculate the 99th percentile of engagement scores
        p99 = conn.execute("""
        SELECT QUANTILE_CONT(engagement_score, 0.99) AS p99_score
        FROM cleaned_casts
        """).fetchone()[0]
        
        print(f"99th percentile of engagement scores: {p99:.2f}")
        
        # Get min and max datetime for recency calculation
        min_max_dt = conn.execute("""
        SELECT MIN(datetime), MAX(datetime) FROM cleaned_casts
        """).fetchone()
        min_dt, max_dt = min_max_dt
        
        # Calculate epoch seconds for time difference
        time_range_seconds = conn.execute(f"""
        SELECT EXTRACT(EPOCH FROM (TIMESTAMP '{max_dt}' - TIMESTAMP '{min_dt}'))
        """).fetchone()[0]
        
        # Use standard weights without conversation metrics - use materialized table
        conn.execute(f"""
        CREATE OR REPLACE TABLE sample_weights AS
        SELECT
            *,
            -- Recency weight (0.1 to 1.1) - pre-calculate time range for better performance
            0.1 + ((EXTRACT(EPOCH FROM (datetime - TIMESTAMP '{min_dt}')) / 
                  {time_range_seconds}))
                AS recency_weight,
            
            -- Engagement weight (0.1 to 1.1)
            -- Cap engagement at 99th percentile to avoid outliers
            0.1 + (LEAST(engagement_score, {p99}) / {p99})
                AS engagement_weight
        FROM cleaned_casts
        """)
        
        # Create an index for faster sorting
        conn.execute("""
        CREATE INDEX IF NOT EXISTS sample_weights_datetime_idx ON sample_weights(datetime)
        """)
        
        # Calculate combined weight without conversation score
        conn.execute("""
        CREATE OR REPLACE TABLE combined_weights AS
        SELECT
            *,
            -- Combined weight (70% recency, 30% engagement)
            -- Increased recency weight to favor newer content when conversation metrics aren't available
            (0.7 * recency_weight) + (0.3 * engagement_weight) AS combined_weight
        FROM sample_weights
        """)
        
        # Create an index for sorting by combined weight
        conn.execute("""
        CREATE INDEX IF NOT EXISTS combined_weights_weight_idx ON combined_weights(combined_weight DESC)
        """)
    
    # Setting to use parallel processing for faster weighted sampling
    conn.execute("""
    PRAGMA threads=30;  -- Use more threads for weighted sampling
    """)
    
    # Use Approach 3's improved stratified sampling methodology
    # This ensures better distribution across conversation metrics and engagement levels
    
    # First determine how much of each bucket type to sample
    # We want a higher representation of posts with conversations
    conversation_buckets = 5  # From the NTILE(5) bucketing above
    
    # Calculate number of samples per conversation bucket, 
    # emphasizing higher conversation buckets (like Approach 3)
    # Calculate once as a SQL operation for better performance
    conn.execute(f"""
    CREATE TEMP TABLE bucket_allocation AS
    WITH bucket_counts AS (
        SELECT 
            conversation_bucket,
            COUNT(*) AS bucket_size,
            -- Give progressively more weight to higher conversation buckets
            CASE 
                WHEN conversation_bucket = 1 THEN 0.05  -- Very few from no-conversation bucket
                WHEN conversation_bucket = 2 THEN 0.15  -- Some from low-conversation bucket
                WHEN conversation_bucket = 3 THEN 0.20  -- Medium from mid-conversation bucket 
                WHEN conversation_bucket = 4 THEN 0.25  -- More from high-conversation bucket
                WHEN conversation_bucket = 5 THEN 0.35  -- Most from very-high-conversation bucket
                ELSE 0.2 -- Fallback
            END AS bucket_weight
        FROM combined_weights
        GROUP BY conversation_bucket
    )
    SELECT
        conversation_bucket,
        bucket_weight,
        CAST(ROUND({sample_size} * bucket_weight) AS INTEGER) AS sample_count
    FROM bucket_counts
    """)
    
    # Now perform the stratified sampling using the allocations
    conn.execute(f"""
    CREATE OR REPLACE TABLE weighted_samples AS
    WITH stratified_samples AS (
        SELECT b.conversation_bucket, b.sample_count,
            c.*,
            -- Add randomness within each bucket weighted by the combined score
            row_number() OVER (
                PARTITION BY c.conversation_bucket 
                ORDER BY c.combined_weight * random() DESC
            ) as row_rank
        FROM combined_weights c
        JOIN bucket_allocation b ON c.conversation_bucket = b.conversation_bucket
    )
    SELECT *
    FROM stratified_samples
    WHERE row_rank <= sample_count
    ORDER BY combined_weight DESC
    LIMIT {sample_size}
    """)
    
    # Create an index on the combined_weight for faster sorting
    # (We no longer use sampling_key in the stratified sampling approach)
    conn.execute("""
    CREATE INDEX IF NOT EXISTS weighted_samples_weight_idx ON weighted_samples(combined_weight DESC)
    """)
    
    # Get the sampled data with parallel execution
    conn.execute("""
    PRAGMA threads=30;  -- Use more threads for retrieving sampled data
    """)
    
    sampled_casts = conn.execute("""
    SELECT * FROM weighted_samples
    """).df()
    
    # Format texts with timestamps, engagement info, and conversation metrics
    formatted_casts = []
    for _, row in sampled_casts.iterrows():
        dt_str = row['datetime'].strftime('%Y-%m-%d %H:%M:%S') if hasattr(row['datetime'], 'strftime') else str(row['datetime'])
        
        # Import pandas if needed
        import pandas as pd
        
        # Handle NaN values safely
        likes = 0 if pd.isna(row.get('likes_count', 0)) else int(row.get('likes_count', 0))
        recasts = 0 if pd.isna(row.get('recasts_count', 0)) else int(row.get('recasts_count', 0))
        
        # Check if we have conversation metrics
        has_convo_metrics = all(field in row for field in ['reply_count', 'unique_repliers', 'conversation_hours'])
        
        # Create base formatted text with engagement metrics
        formatted_text = f"{dt_str} [👍{likes}|↗️{recasts}]"
        
        # Add conversation metrics if available
        if has_convo_metrics:
            # Handle NaN values safely using pandas isna
            if pd.isna(row.get('reply_count')):
                reply_count = 0
            else:
                reply_count = int(row.get('reply_count', 0))
                
            if pd.isna(row.get('unique_repliers')):
                unique_repliers = 0
            else:
                unique_repliers = int(row.get('unique_repliers', 0))
                
            if pd.isna(row.get('conversation_hours')):
                convo_hours = 0.0
            else:
                convo_hours = float(row.get('conversation_hours', 0))
            
            # Only add conversation metrics if there are replies
            if reply_count > 0:
                # Format conversation duration 
                if convo_hours < 1:
                    time_str = f"{int(convo_hours * 60)}m"
                elif convo_hours < 24:
                    time_str = f"{int(convo_hours)}h"
                else:
                    time_str = f"{int(convo_hours/24)}d"
                    
                # Add conversation metrics
                formatted_text += f" [🗨️{reply_count}|👥{unique_repliers}|⏱️{time_str}]"
        
        # Add the actual post text
        formatted_text += f": {row['Text']}"
        formatted_casts.append(formatted_text)
    
    # Get sample statistics and verify we only have top-level posts
    sample_stats = conn.execute("""
    SELECT
        COUNT(*) AS sample_count,
        AVG(engagement_score) AS avg_engagement,
        AVG(LENGTH("Text")) AS avg_text_length,
        SUM(LENGTH("Text")) AS total_chars,
        SUM(CASE WHEN is_reply = TRUE THEN 1 ELSE 0 END) AS reply_count
    FROM weighted_samples
    """).fetchone()
    
    # Verify no replies made it into our sample
    if sample_stats[4] > 0:
        print(f"WARNING: Sample contains {sample_stats[4]} replies - these should have been filtered out")
    else:
        print("✅ Sample contains only top-level posts as intended")
    
    print(f"Sampled {sample_stats[0]:,} posts for direct LLM analysis")
    print(f"Average engagement in sample: {sample_stats[1]:.2f}")
    print(f"Average text length: {sample_stats[2]:.1f} chars")
    print(f"Total characters: {sample_stats[3]:,} chars")
    print(f"Sampling completed in {time.time() - start_time:.2f} seconds")
    
    # Check if conversation metrics exist in the dataframe (already calculated in data_preprocessing.py)
    has_conversation_metrics = all(field in recent_df.columns for field in ['reply_count', 'unique_repliers', 'conversation_hours'])
    
    # Enable parallel processing for representativeness calculation
    conn.execute("""
    PRAGMA threads=30;  -- Use more threads for representativeness calculation
    """)
    
    # Create materialized tables for counts to improve performance
    # Calculate sample representativeness with metrics already calculated in data_preprocessing
    if has_conversation_metrics:
        print("Using pre-calculated conversation metrics from data_preprocessing.py")
        
        # Create materialized tables for sample and population metrics
        conn.execute("""
        CREATE OR REPLACE TABLE sample_metrics AS
        SELECT
            COUNT(*) AS total,
            SUM(COALESCE(reply_count, 0)) AS total_replies,
            AVG(COALESCE(reply_count, 0)) AS avg_replies,
            SUM(CASE WHEN COALESCE(reply_count, 0) > 0 THEN 1 ELSE 0 END) AS posts_with_replies,
            AVG(COALESCE(unique_repliers, 0)) AS avg_unique_repliers,
            AVG(engagement_score) AS avg_engagement,
            MEDIAN(engagement_score) AS median_engagement,
            COUNT(DISTINCT Fid) AS unique_users
        FROM weighted_samples
        """)
        
        conn.execute("""
        CREATE OR REPLACE TABLE population_metrics AS
        SELECT
            COUNT(*) AS total,
            SUM(COALESCE(reply_count, 0)) AS total_replies,
            AVG(COALESCE(reply_count, 0)) AS avg_replies,
            SUM(CASE WHEN COALESCE(reply_count, 0) > 0 THEN 1 ELSE 0 END) AS posts_with_replies,
            AVG(COALESCE(unique_repliers, 0)) AS avg_unique_repliers,
            AVG(engagement_score) AS avg_engagement,
            MEDIAN(engagement_score) AS median_engagement,
            COUNT(DISTINCT Fid) AS unique_users
        FROM cleaned_casts
        """)
        
        # Retrieve all metrics in a single optimized query
        metrics = conn.execute("""
        SELECT
            s.avg_engagement AS s_avg_engagement,
            p.avg_engagement AS p_avg_engagement,
            s.median_engagement AS s_median_engagement,
            p.median_engagement AS p_median_engagement,
            s.unique_users AS s_unique_users,
            p.unique_users AS p_unique_users,
            (s.posts_with_replies * 100.0 / s.total) AS s_reply_percentage,
            (p.posts_with_replies * 100.0 / p.total) AS p_reply_percentage,
            s.avg_replies AS s_avg_replies,
            p.avg_replies AS p_avg_replies,
            s.avg_unique_repliers AS s_avg_unique_repliers,
            p.avg_unique_repliers AS p_avg_unique_repliers
        FROM sample_metrics s, population_metrics p
        """).fetchone()
        
        representation = metrics[:8]  # Keep only the original 8 metrics for backward compatibility
        
        # Print additional conversation metrics with clearer labels
        print("\nConversation metrics in sample vs population:")
        print(f"  - Avg replies per post: {metrics[8]:.2f} in sample vs {metrics[9]:.2f} in population")
        print(f"  - Avg unique repliers per post: {metrics[10]:.2f} in sample vs {metrics[11]:.2f} in population")
    else:
        # If conversation metrics don't exist, use a simplified version
        # Create materialized tables with simplified metrics
        print("No pre-calculated conversation metrics found, using simplified metrics")
        
        conn.execute("""
        CREATE OR REPLACE TABLE sample_metrics AS
        SELECT
            AVG(engagement_score) AS avg_engagement,
            MEDIAN(engagement_score) AS median_engagement,
            COUNT(DISTINCT Fid) AS unique_users
        FROM weighted_samples
        """)
        
        conn.execute("""
        CREATE OR REPLACE TABLE population_metrics AS
        SELECT
            AVG(engagement_score) AS avg_engagement,
            MEDIAN(engagement_score) AS median_engagement,
            COUNT(DISTINCT Fid) AS unique_users
        FROM cleaned_casts
        """)
        
        # Retrieve metrics in a single optimized query
        metrics = conn.execute("""
        SELECT
            s.avg_engagement AS s_avg_engagement,
            p.avg_engagement AS p_avg_engagement,
            s.median_engagement AS s_median_engagement,
            p.median_engagement AS p_median_engagement,
            s.unique_users AS s_unique_users,
            p.unique_users AS p_unique_users,
            0.0 AS s_reply_percentage,
            0.0 AS p_reply_percentage
        FROM sample_metrics s, population_metrics p
        """).fetchone()
        
        representation = metrics
    
    print("Sample representativeness metrics:")
    print(f"  - Engagement: sample avg {representation[0]:.2f} vs population avg {representation[1]:.2f}")
    print(f"  - Unique users: {representation[4]:,} in sample vs {representation[5]:,} in population")
    print(f"  - % of posts with replies: {representation[6]:.1f}% in sample vs {representation[7]:.1f}% in population")
    
    # Gemini API Integration
    print("Calling Gemini API for topic extraction...")
    
    # Get API key from environment
    import os
    import google.generativeai as genai
    
    api_key = os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        # Try to read from .env file as fallback
        try:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.environ.get('GOOGLE_API_KEY')
        except ImportError:
            print("dotenv package not installed, can't load from .env file")
    
    # Configure Gemini with API key        
    if api_key:
        print(f"Configuring Gemini with API key from environment")
        genai.configure(api_key=api_key)
    else:
        print("WARNING: No GOOGLE_API_KEY found in environment or .env file")
        print("Gemini might not work without an API key")
    
    # Initialize Gemini with structured output
    model = GenerativeModel('gemini-2.0-flash')
    
    # Create a more detailed prompt with time context and structured output
    date_range = f"{recent_df['datetime'].min().strftime('%Y-%m-%d')} to {recent_df['datetime'].max().strftime('%Y-%m-%d')}"
    
    # Create structured prompt with token count management
    # We need to ensure we don't exceed Gemini's 1M token limit
    # Average token-to-character ratio is ~4 characters per token for English text
    # Gemini-2.0-flash has a 1M token limit; limit total to ~700K tokens to leave room for model response
    max_estimated_chars = 700000 * 4  # ~700K tokens
    
    # Estimate the non-post content in the prompt (instructions, etc.)
    # This is roughly 3K tokens or 12K characters
    instruction_chars = 12000
    
    # Calculate how many characters we can use for posts
    available_chars = max_estimated_chars - instruction_chars
    
    # Truncate the formatted_casts list if needed
    total_chars = sum(len(post) for post in formatted_casts)
    if total_chars > available_chars:
        print(f"WARNING: Total characters ({total_chars}) exceeds available space ({available_chars})")
        print(f"Truncating sample to fit within token limits...")
        
        # Find how many posts we can include
        running_total = 0
        cutoff_index = len(formatted_casts)
        for i, post in enumerate(formatted_casts):
            running_total += len(post)
            if running_total > available_chars:
                cutoff_index = i
                break
        
        # Truncate the list
        formatted_casts = formatted_casts[:cutoff_index]
        print(f"Reduced from {len(formatted_casts)} to {cutoff_index} posts to fit token limit")
    
    formatted_posts = "\n".join(formatted_casts)
    prompt = f"""
    Analyze the following Farcaster social media posts from the period: {date_range}.
    These are samples from a dataset of {len(recent_df)} posts from this timeframe.
    The actual number of posts provided in this prompt is {len(formatted_casts)}.
    
    Your task is to identify the top 5 TRULY TRENDING topics of discussion with supporting evidence.
    
    TRENDING topics have these characteristics:
    - HIGH ENGAGEMENT: Topics with many likes and recasts
    - RECENCY: Topics that are active in the most recent timeframe
    - GROWTH: Topics that show increasing activity over time
    - CONVERSATION: Topics that generate many replies and discussions
    
    Generate your response based on the following TypedDict schema in Python:
    
    class KeyTerm(TypedDict):
        term: str
        frequency: int  # Estimated frequency
    
    class KeyEntity(TypedDict):
        name: str
        type: str  # Person, Project, Company, etc.
        relevance: str  # High, Medium, Low
    
    class Topic(TypedDict):
        name: str  # 5 words max
        explanation: str  # Brief explanation of why trending
        estimated_percentage: str  # Percentage of posts
        key_terms: list[KeyTerm]
        key_entities: list[KeyEntity]
        engagement_level: str  # High, Medium, Low based on likes/recasts
    
    class TrendingTopics(TypedDict):
        topics: list[Topic]
        analysis_period: str
        total_posts_analyzed: int  # MUST be accurate - set to the actual number of posts we provide (NOT 50 or 100)
    
    CRITICAL REQUIREMENTS:
    1. SPECIFIC TOPICS ONLY: No generic categories like "NFT Discussion", "Crypto News", or "Community Engagement". 
       Identify specific projects, protocols, products, features, or events (e.g., "BaseDAO Governance Proposal", "Farcaster Frames API", "Arbitrum Fraud Proofs")
    
    2. ORGANIC TRENDING CONVERSATIONS: Find topics that are naturally gaining traction. If people are discussing a specific 
       token launch, name the specific token, not just "Token Launches"
    
    3. RECENCY CRITICAL: The topic must show a clear pattern of INCREASING discussion in the most recent period
    
    4. PRECISE NAMING: Topic names should accurately capture what is being discussed without being too generic
    
    5. CONVERSATION METRICS: Pay special attention to posts that have generated active conversations. These posts have additional 
       metrics in this format: [🗨️X|👥Y|⏱️Z] where:
       - 🗨️X = Number of replies to the post
       - 👥Y = Number of unique users who replied
       - ⏱️Z = Duration of the conversation (m=minutes, h=hours, d=days)
       Posts with more replies and longer conversations often represent important trending topics
    
    6. SUBSTANCE FOCUS: Prioritize substantive discussions around technology, product updates, ecosystem developments, market events,
       or important community announcements that show organic growth in community interest.
    
    7. NO GENERIC TECH CATEGORIES: Never use generic labels like "AI Discussion" or "Blockchain Technology". Instead, identify the specific
       AI tool, blockchain project, or feature being discussed (e.g., "Scroll zkEVM Mainnet" not just "Layer 2 Rollups")
    
    8. CONVERSATION-DRIVEN TRENDS: Give higher weight to posts that have generated significant conversation (many replies, many unique 
       repliers, or long conversations). These often indicate topics that the community is actively engaging with, not just passively consuming
    
    POSTS:
    {formatted_posts}
    """
    
    # Get response with JSON formatting and increased temperature for more creativity
    try:
        response = model.generate_content(
            prompt,
            generation_config=types.GenerationConfig(
                temperature=0.4,  # Moderate temperature for balanced creativity and consistency
                response_mime_type="application/json"
            )
        )
        
        # Parse and process the response
        try:
            llm_topics = json.loads(response.text)
            # Handle case where response is a list (API format change)
            if isinstance(llm_topics, list) and len(llm_topics) > 0:
                print(f"API returned list format, extracting first item from list of {len(llm_topics)}")
                first_item = llm_topics[0]
                # Create standard format
                if isinstance(first_item, dict):
                    if 'topics' in first_item:
                        llm_topics = first_item
                    else:
                        # Handle case where the list contains topic objects directly
                        llm_topics = {
                            "topics": llm_topics,
                            "analysis_period": date_range,
                            "total_posts_analyzed": len(formatted_casts)  # Actual number of posts analyzed
                        }
                else:
                    raise ValueError("Response list doesn't contain dictionaries")
            
            # Check if the response has the expected structure
            if isinstance(llm_topics, dict):
                topics_list = llm_topics.get('topics', [])
                print(f"Successfully received structured response with {len(topics_list)} topics")
            else:
                print(f"Unexpected response type: {type(llm_topics)}")
                llm_topics = {
                    "topics": [],
                    "analysis_period": date_range,
                    "total_posts_analyzed": len(formatted_casts)  # Actual number of posts analyzed
                }
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            llm_topics = {
                "topics": [],
                "analysis_period": date_range,
                "total_posts_analyzed": len(formatted_casts)  # Actual number of posts analyzed
            }
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        # Simply propagate the error
        raise
    
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    # Save raw response for debugging/analysis
    with open('output/llm_response_raw.json', 'w') as f:
        json.dump(llm_topics, f, indent=2)
    
    # Save in the format expected by downstream processes
    with open('output/approach1_results.json', 'w') as f:
        json.dump(llm_topics, f, indent=2)
    
    # Process Results and Enrichment
    # Log topic information
    if 'topics' in llm_topics:
        for i, topic in enumerate(llm_topics['topics']):
            print(f"Topic {i+1}: {topic['name']}")
            print(f"  Estimated percentage: {topic['estimated_percentage']}")
            print(f"  Engagement level: {topic['engagement_level']}")
            print(f"  Key terms: {', '.join([term['term'] for term in topic['key_terms'][:5]])}")
            print(f"  Key entities: {', '.join([entity['name'] for entity in topic['key_entities'][:3]])}")
            print()
    
    # Find exemplar posts for each topic
    if 'topics' in llm_topics:
        topic_exemplars = {}
        for i, topic in enumerate(llm_topics['topics']):
            exemplars = find_exemplar_posts_for_topic(topic, sampled_casts)
            topic_exemplars[topic['name']] = exemplars
        
        # Save exemplars for later use
        with open('output/topic_exemplar_posts.json', 'w') as f:
            exemplar_data = {
                topic_name: [
                    {
                        'text': post['Text'],
                        'datetime': post['datetime'] if isinstance(post['datetime'], str) else str(post['datetime']),
                        'likes': 0 if pd.isna(post.get('likes_count', 0)) else int(post.get('likes_count', 0)),
                        'recasts': 0 if pd.isna(post.get('recasts_count', 0)) else int(post.get('recasts_count', 0))
                    } for post in posts
                ] for topic_name, posts in topic_exemplars.items()
            }
            json.dump(exemplar_data, f, indent=2)
    
    return llm_topics

def find_exemplar_posts_for_topic(topic, posts_df, n=3):
    """Find exemplar posts that best represent a given topic."""
    # Create a simple keyword matching function
    topic_keywords = [term['term'].lower() for term in topic['key_terms']]
    
    # Score each post based on keyword matches and engagement
    scores = []
    for _, row in posts_df.iterrows():
        text = str(row['Text']).lower()  # Ensure we have a string
        # Count keyword matches
        keyword_score = sum(1 for kw in topic_keywords if kw in text)
        # Normalize by text length (avoid favoring very long posts)
        keyword_score = keyword_score / (len(text.split()) + 1) * 10
        # Add engagement bonus
        engagement_bonus = float(row['engagement_score']) / 100
        # Combined score
        combined_score = float(keyword_score) + float(engagement_bonus)
        
        if keyword_score > 0:  # Only consider posts with at least one keyword match
            # Convert row to dict to avoid Pandas Series comparison issues
            scores.append((combined_score, row.to_dict()))
    
    # Sort by score and take top n
    scores.sort(key=lambda x: x[0], reverse=True)
    return [post for _, post in scores[:n]]

if __name__ == "__main__":
    # This module is imported and run from the main.py file
    pass