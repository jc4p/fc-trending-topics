import random
import time
import json
import re
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

def process_batch(conn, batch_df, batch_number, batch_size, date_range, total_batches, api_key):
    """
    Process a single batch of posts to identify trending topics.
    
    Args:
        conn: DuckDB connection
        batch_df: DataFrame containing posts for this batch
        batch_number: Current batch number (1-indexed)
        batch_size: Number of posts in each batch
        date_range: String representing the date range for the entire dataset
        total_batches: Total number of batches to process
        api_key: Google API key for Gemini
        
    Returns:
        dict: Structured trending topics result for this batch
    """
    batch_start_time = time.time()
    print(f"\nProcessing batch {batch_number}/{total_batches} ({len(batch_df)} posts)...")
    
    # Register the batch DataFrame as a temp table
    conn.register(f'batch_{batch_number}', batch_df)
    
    # Check if we have conversation metrics available for this batch
    has_conversation_metrics = conn.execute(f"""
    SELECT COUNT(*) > 0 
    FROM INFORMATION_SCHEMA.COLUMNS 
    WHERE TABLE_NAME = 'batch_{batch_number}' 
    AND COLUMN_NAME IN ('reply_count', 'unique_repliers')
    """).fetchone()[0]
    
    # Create batch-specific views with appropriate weights
    if has_conversation_metrics:
        print(f"Batch {batch_number}: Using enhanced engagement score with conversation metrics")
        # Calculate weights directly in SQL with conversation metrics
        conn.execute(f"""
        CREATE OR REPLACE VIEW batch_{batch_number}_weights AS
        SELECT
            *,
            -- Calculate enhanced engagement score with conversation metrics
            COALESCE(likes_count, 0) + 
            (3 * COALESCE(recasts_count, 0)) + 
            (5 * COALESCE(reply_count, 0)) + 
            (10 * COALESCE(unique_repliers, 0)) AS enhanced_engagement_score,
            
            -- Recency weight (1 to 10)
            1 + 9 * ((EXTRACT(EPOCH FROM (datetime - MIN(datetime) OVER())) / 
                   NULLIF(EXTRACT(EPOCH FROM (MAX(datetime) OVER() - MIN(datetime) OVER())), 0)))
                AS recency_weight,
            
            -- Engagement weight (1 to 10) using enhanced score
            -- Cap engagement at 90% of max to avoid outliers
            1 + 9 * (
                LEAST(
                    COALESCE(likes_count, 0) + 
                    (3 * COALESCE(recasts_count, 0)) + 
                    (5 * COALESCE(reply_count, 0)) + 
                    (10 * COALESCE(unique_repliers, 0)), 
                    0.9 * MAX(COALESCE(likes_count, 0) + 
                             (3 * COALESCE(recasts_count, 0)) + 
                             (5 * COALESCE(reply_count, 0)) + 
                             (10 * COALESCE(unique_repliers, 0))) OVER()
                ) / 
                NULLIF((CASE WHEN MAX(COALESCE(likes_count, 0) + 
                              (3 * COALESCE(recasts_count, 0)) + 
                              (5 * COALESCE(reply_count, 0)) + 
                              (10 * COALESCE(unique_repliers, 0))) OVER() > 0 
                      THEN 0.9 * MAX(COALESCE(likes_count, 0) + 
                                   (3 * COALESCE(recasts_count, 0)) + 
                                   (5 * COALESCE(reply_count, 0)) + 
                                   (10 * COALESCE(unique_repliers, 0))) OVER()
                      ELSE 1 END), 0)
            ) AS engagement_weight
        FROM batch_{batch_number}
        """)
        
        # Calculate combined weight with emphasis on conversation
        conn.execute(f"""
        CREATE OR REPLACE VIEW batch_{batch_number}_combined AS
        SELECT
            *,
            -- Combined weight (40% recency, 60% engagement)
            (4 * recency_weight + 6 * engagement_weight) / 10 AS combined_weight
        FROM batch_{batch_number}_weights
        """)
    else:
        print(f"Batch {batch_number}: Using standard engagement score")
        # Calculate weights directly in SQL with standard engagement
        conn.execute(f"""
        CREATE OR REPLACE VIEW batch_{batch_number}_weights AS
        SELECT
            *,
            -- Recency weight (1 to 10)
            1 + 9 * ((EXTRACT(EPOCH FROM (datetime - MIN(datetime) OVER())) / 
                   NULLIF(EXTRACT(EPOCH FROM (MAX(datetime) OVER() - MIN(datetime) OVER())), 0)))
                AS recency_weight,
            
            -- Engagement weight (1 to 10)
            -- Cap engagement at 90% of max to avoid outliers
            1 + 9 * (
                LEAST(
                    engagement_score, 
                    0.9 * MAX(engagement_score) OVER()
                ) / 
                NULLIF((CASE WHEN MAX(engagement_score) OVER() > 0 
                      THEN 0.9 * MAX(engagement_score) OVER()
                      ELSE 1 END), 0)
            ) AS engagement_weight
        FROM batch_{batch_number}
        """)
        
        # Standard weight distribution when no conversation metrics
        conn.execute(f"""
        CREATE OR REPLACE VIEW batch_{batch_number}_combined AS
        SELECT
            *,
            -- Combined weight (60% recency, 40% engagement)
            (6 * recency_weight + 4 * engagement_weight) / 10 AS combined_weight
        FROM batch_{batch_number}_weights
        """)
    
    # Handle top-level post filtering if available
    if conn.execute(f"""
        SELECT COUNT(*) > 0 
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_NAME = 'batch_{batch_number}' 
        AND COLUMN_NAME = 'ParentCastId'
        """).fetchone()[0]:
        
        # Focus on top-level posts for analysis
        print(f"Batch {batch_number}: Filtering to focus on top-level posts only...")
        conn.execute(f"""
        CREATE OR REPLACE VIEW batch_{batch_number}_filtered AS
        SELECT * 
        FROM batch_{batch_number}_combined
        WHERE ParentCastId IS NULL OR TRIM(ParentCastId) = ''
        """)
    else:
        # If no ParentCastId column, just use all posts
        print(f"Batch {batch_number}: ParentCastId information not available - using all posts")
        conn.execute(f"""
        CREATE OR REPLACE VIEW batch_{batch_number}_filtered AS
        SELECT * FROM batch_{batch_number}_combined
        """)
    
    # Sample posts from the batch using weighted sampling
    # Use the entire batch for analysis
    conn.execute(f"""
    CREATE OR REPLACE VIEW batch_{batch_number}_samples AS
    SELECT
        *,
        -- Generate random weight based on combined_weight
        random() * combined_weight AS sampling_key
    FROM batch_{batch_number}_filtered
    ORDER BY sampling_key DESC
    """)
    
    # Get the sampled data for this batch - limit to 1000 posts maximum per batch
    # to avoid hitting API context limits
    sampled_casts = conn.execute(f"""
    SELECT * FROM batch_{batch_number}_samples
    LIMIT 1000
    """).df()
    
    # Format texts with timestamps, engagement info, and conversation metrics if available
    formatted_casts = []
    for _, row in sampled_casts.iterrows():
        dt_str = row['datetime'].strftime('%Y-%m-%d %H:%M:%S') if hasattr(row['datetime'], 'strftime') else str(row['datetime'])
        likes = int(row['likes_count']) if pd.notna(row['likes_count']) else 0
        recasts = int(row['recasts_count']) if pd.notna(row['recasts_count']) else 0
        
        # Create base formatted text with engagement metrics
        formatted_text = f"{dt_str} [👍{likes}|↗️{recasts}]"
        
        # Check if we have conversation metrics
        has_convo_metrics = all(field in row for field in ['reply_count', 'unique_repliers'])
        
        if has_convo_metrics:
            # Handle NaN values safely using pandas isna
            reply_count = 0 if pd.isna(row.get('reply_count')) else int(row.get('reply_count', 0))
            unique_repliers = 0 if pd.isna(row.get('unique_repliers')) else int(row.get('unique_repliers', 0))
            
            # Only add conversation metrics if there are replies
            if reply_count > 0:
                # Add conversation metrics
                formatted_text += f" [🗨️{reply_count}|👥{unique_repliers}]"
        
        # Add the actual post text
        formatted_text += f": {row['Text']}"
        formatted_casts.append(formatted_text)
    
    # Get batch sample statistics
    sample_stats = conn.execute(f"""
    SELECT
        COUNT(*) AS sample_count,
        COALESCE(AVG(engagement_score), 0) AS avg_engagement,
        COALESCE(AVG(LENGTH("Text")), 0) AS avg_text_length,
        COALESCE(SUM(LENGTH("Text")), 0) AS total_chars
    FROM batch_{batch_number}_samples
    """).fetchone()
    
    # Check if we have any samples
    if sample_stats[0] == 0:
        print(f"Batch {batch_number}: No posts available for analysis after filtering")
        # Return empty result for this batch
        return {
            "topics": [],
            "analysis_period": date_range,
            "total_posts_analyzed": 0,
            "batch_number": batch_number
        }
    
    print(f"Batch {batch_number}: Sampled {sample_stats[0]:,} posts for analysis")
    print(f"Batch {batch_number}: Average engagement in sample: {sample_stats[1]:.2f}")
    print(f"Batch {batch_number}: Average text length: {sample_stats[2]:.1f} chars")
    
    # Initialize Gemini for this batch
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    batch_model = GenerativeModel('gemini-2.0-flash-lite')
    
    # Check if we included conversation metrics in the formatted posts
    includes_convo_metrics = any("🗨️" in post for post in formatted_casts)
    
    # Create batch-specific prompt
    joined_posts = "\n".join(formatted_casts)
    batch_prompt = f"""
    Analyze the following batch of Farcaster social media posts from the period: {date_range}.
    This is batch {batch_number} of {total_batches} from the dataset.
    
    CRITICALLY IMPORTANT: DO NOT INCLUDE ANY SPECIFIC TOKEN/COIN PROJECTS OR AIRDROPS. 
    You must avoid including any cryptocurrency token projects or airdrops as trending topics.
    These are typically promotional in nature, not genuine community trends.
    
    For this batch, identify the top 10 TRULY TRENDING topics of discussion with supporting evidence.
    You MUST identify 10 distinct topics to ensure comprehensive coverage.
    
    TRENDING topics have these characteristics:
    - HIGH ENGAGEMENT: Topics with many likes and recasts
    - RECENCY: Topics that are active in the most recent timeframe
    - GROWTH: Topics that show increasing activity over time
    - CONVERSATION DEPTH: Topics that generate substantive discussions (many replies, unique repliers)
    
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
        total_posts_analyzed: int
        batch_number: int
    
    CRITICAL REQUIREMENTS:
    1. SPECIFIC TOPICS ONLY: No generic categories like "NFT Discussion", "Crypto News", or "Community Engagement". 
       Identify specific projects, protocols, products, features, or events (e.g., "BaseDAO Governance Proposal", "Farcaster Frames API")
    
    2. ORGANIC TRENDING CONVERSATIONS: Find topics that are naturally gaining traction. If people are discussing a specific 
       token launch, name the specific token, not just "Token Launches"
    
    3. RECENCY CRITICAL: The topic must show a clear pattern of INCREASING discussion in the most recent period
    
    4. PRECISE NAMING: Topic names should accurately capture what is being discussed without being too generic
    
    5. PRIORITIZE SURPRISING TOPICS: Focus on identifying unexpected, novel, or intriguing trends that would make users say 
       "Huh, I didn't know about that!" - topics that would make someone want to click through to learn more
       
    6. CAPTURE EMERGING PHENOMENA: Look for emerging cultural phenomena, inside jokes, or community-specific terms/memes 
       that are gaining traction but might not be widely known yet
       
    7. FOCUS ON QUALITY: Prioritize identifying clearly trending topics with strong evidence rather than quantity
       
    8. DO NOT INCLUDE TOKEN LAUNCHES OR AIRDROPS: You MUST NOT include any specific coin/token 
       launches, airdrops, or token projects as trending topics. These are typically promotional
       in nature and not representative of genuine community interest.
       
       Instead, focus EXCLUSIVELY on:
       - Platform features and innovations
       - Community cultural trends and memes
       - Technology discussions and applications
       - Unique behavioral patterns among users
       
    9. BALANCED REPRESENTATION: Ensure you identify diverse topics across the dataset. Don't let a single viral post
       with many replies dominate your analysis. A post with hundreds of "me too" or similar low-value replies
       should not overwhelm other meaningful discussions happening in parallel. Look beyond raw engagement numbers
       to find substantive conversations on different topics.
    """ + ("""
    10. CONVERSATION METRICS: Pay special attention to posts that have generated active conversations. These posts have additional 
       metrics in this format: [🗨️X|👥Y] where:
       - 🗨️X = Number of replies to the post
       - 👥Y = Number of unique users who replied
       Posts with more replies and many unique repliers often represent important trending topics
    """ if includes_convo_metrics else "") + f"""
    
    POSTS:
    {joined_posts}
    """
    
    # Get response with JSON formatting for this batch
    batch_response = batch_model.generate_content(
        batch_prompt,
        generation_config=types.GenerationConfig(
            temperature=0,  # Zero temperature for consistent results
            response_mime_type="application/json"
        )
    )
    
    # Parse and process the batch response
    try:
        batch_topics = json.loads(batch_response.text)
        
        # Add batch information if not already present
        if isinstance(batch_topics, dict) and 'batch_number' not in batch_topics:
            batch_topics['batch_number'] = batch_number
        
        # Handle case where response is a list (API format change)
        if isinstance(batch_topics, list) and len(batch_topics) > 0:
            print(f"Batch {batch_number}: API returned list format, extracting first item from list of {len(batch_topics)}")
            first_item = batch_topics[0]
            # Create standard format
            if isinstance(first_item, dict):
                if 'topics' in first_item:
                    batch_topics = first_item
                else:
                    # Handle case where the list contains topic objects directly
                    batch_topics = {
                        "topics": batch_topics,
                        "analysis_period": date_range,
                        "total_posts_analyzed": len(formatted_casts),
                        "batch_number": batch_number
                    }
            else:
                raise ValueError("Response list doesn't contain dictionaries")
        
        # Check if the response has the expected structure
        if isinstance(batch_topics, dict):
            topics_list = batch_topics.get('topics', [])
            print(f"Batch {batch_number}: Successfully received structured response with {len(topics_list)} topics")
        else:
            print(f"Batch {batch_number}: Unexpected response type: {type(batch_topics)}")
            batch_topics = {
                "topics": [],
                "analysis_period": date_range,
                "total_posts_analyzed": len(formatted_casts),
                "batch_number": batch_number
            }
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Batch {batch_number}: Error parsing JSON response: {e}")
        batch_topics = {
            "topics": [],
            "analysis_period": date_range,
            "total_posts_analyzed": len(formatted_casts),
            "batch_number": batch_number
        }
    
    # Create output/cache directory if it doesn't exist
    os.makedirs('output/cache', exist_ok=True)
    
    # Save batch results for later consolidation
    with open(f'output/cache/batch_{batch_number}_results.json', 'w') as f:
        json.dump(batch_topics, f, indent=2)
    
    print(f"Batch {batch_number} completed in {time.time() - batch_start_time:.2f} seconds")
    return batch_topics

def consolidate_results(batch_results, date_range, total_posts, api_key):
    """
    Consolidate results from multiple batches using a final LLM call.
    
    Args:
        batch_results: List of batch result dictionaries
        date_range: String representing the date range
        total_posts: Total number of posts analyzed
        api_key: Google API key for Gemini
        
    Returns:
        dict: Final consolidated trending topics
    """
    print("\nConsolidating results from all batches...")
    start_time = time.time()
    
    # Prepare consolidated topics list from all batches
    all_topics = []
    unique_topic_names = set()
    
    # Print all topics from all batches
    print("\nALL UNIQUE TOPICS IDENTIFIED ACROSS BATCHES:")
    print("-------------------------------------------")
    
    for batch_result in batch_results:
        if 'topics' in batch_result:
            batch_num = batch_result.get('batch_number', 0)
            
            # Tag topics with their batch number for reference
            for topic in batch_result['topics']:
                topic_name = topic['name']
                unique_topic_names.add(topic_name)
                topic['source_batch'] = batch_num
                all_topics.append(topic)
    
    # Sort and display all unique topics found
    sorted_topics = sorted(list(unique_topic_names))
    for i, topic_name in enumerate(sorted_topics):
        print(f"{i+1}. {topic_name}")
    
    print(f"\nTotal topics identified across all batches: {len(all_topics)}")
    print(f"Unique topic names: {len(unique_topic_names)}")
    
    # Initialize Gemini for consolidation - using Gemini 1.5 Pro with JSON support
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    consolidation_model = GenerativeModel('gemini-1.5-pro')  # Using Gemini 1.5 Pro for consolidation
    
    # Create consolidated topics in JSON format
    consolidated_json = json.dumps(all_topics, indent=2)
    
    # Create consolidation prompt
    consolidation_prompt = f"""
    You're selecting the most compelling topics for Farcaster's Explore tab, based on {total_posts:,} posts from the period {date_range}.
    
    CORE PRINCIPLE: The Explore tab should feature GENUINELY INTERESTING and BROADLY RELEVANT trends that make users want to click.
    
    YOUR CURATION TASK:
    From the topics below, identify the 5 that are:
    1. MOST ENGAGING - would make users say "I want to see more of that!"
    2. WIDELY RELEVANT - appeal to a broad audience, not niche communities
    3. NOVEL & SURPRISING - reveal something users didn't already know
    4. CULTURALLY SIGNIFICANT - represent meaningful community patterns
    5. VISUALLY OR CONCEPTUALLY STRIKING - have strong "click appeal"
    
    EVALUATION FRAMEWORK (no single topic needs to meet all criteria):
    - VIRALITY POTENTIAL: Topics that are gaining momentum and generating excitement
    - CREATIVE EXPRESSION: Novel ways users are expressing themselves or interacting
    - CULTURAL SIGNIFICANCE: Emerging community behaviors or shared experiences
    - VISUAL APPEAL: Content with strong visual components that draw attention
    - CONVERSATION STARTERS: Topics likely to spark further discussion
    
    STRICT EXCLUSION CRITERIA:
    1. NOT PROMOTIONAL: Avoid topics that promote specific products, services, or communities
    2. NOT APP-SPECIFIC: Exclude trends tied to specific third-party tools rather than platform-wide behaviors
    3. NOT ROUTINE: Filter out predictable, everyday activities without novel elements
    4. NOT TECHNICAL: Avoid platform technical details that only appeal to power users
    5. NOT NICHE: Exclude topics that would only interest a small subset of users
   
    DETECTIVE WORK REQUIRED:
    You must analyze each topic to determine whether it represents:
    - A specific third-party app (disguised as a general trend)
    - A newly launched project or community onboarding process
    - A game requiring special frames to play
    - A trend linked to a specific commercial entity
    - A topic only relevant to a small subset of users
    
    Apply your critical judgment - the distinction between platform-wide cultural phenomena 
    and specific product promotion can be subtle. Look for clues in the names, terms, and descriptions.
    
    After reviewing all topics, select the 5 that would most compellingly showcase what makes
    Farcaster interesting and vibrant as a community.
    
    Prioritize topics that represent:
    - Widespread participation (higher percentages)
    - Unusual or surprising community activities
    - Novel cultural trends gaining momentum
    - Distinctive behavioral patterns unique to this platform
    - Visually interesting or conceptually fascinating content
    
    
    For the chosen 5 topics, a user encountering them should think "This looks interesting, 
    I want to explore this more" rather than "Oh, that's just routine platform activity."
    
    For each of your 5 selected topics, provide:
    - A compelling, specific name (max 5 words)
    - A concise explanation of why this is genuinely trending and interesting
    - An estimated percentage of posts discussing this topic
    - Key terms associated with the topic
    - Key entities (people, projects, features) involved
    - The overall engagement level (High, Medium, Low)
    
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
        total_posts_analyzed: int
    
    TOPICS FROM ALL BATCHES:
    {consolidated_json}
    """
    
    # Get consolidated response with JSON output
    consolidation_response = consolidation_model.generate_content(
        consolidation_prompt,
        generation_config=types.GenerationConfig(
            temperature=0,  # Zero temperature for consistent results
            response_mime_type="application/json"  # Request JSON output
        )
    )
    
    # Parse and process the consolidation response
    try:
        consolidated_topics = json.loads(consolidation_response.text)
        
        # Handle case where response is a list (API format change)
        if isinstance(consolidated_topics, list) and len(consolidated_topics) > 0:
            print(f"Consolidation: API returned list format, extracting first item from list of {len(consolidated_topics)}")
            first_item = consolidated_topics[0]
            # Create standard format
            if isinstance(first_item, dict):
                if 'topics' in first_item:
                    consolidated_topics = first_item
                else:
                    # Handle case where the list contains topic objects directly
                    consolidated_topics = {
                        "topics": consolidated_topics,
                        "analysis_period": date_range,
                        "total_posts_analyzed": total_posts
                    }
            else:
                raise ValueError("Response list doesn't contain dictionaries")
        
        # Check if the response has the expected structure
        if isinstance(consolidated_topics, dict):
            topics_list = consolidated_topics.get('topics', [])
            print(f"Consolidation: Successfully received structured response with {len(topics_list)} topics")
        else:
            print(f"Consolidation: Unexpected response type: {type(consolidated_topics)}")
            consolidated_topics = {
                "topics": [],
                "analysis_period": date_range,
                "total_posts_analyzed": total_posts
            }
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Consolidation: Error parsing JSON response: {e}")
        consolidated_topics = {
            "topics": [],
            "analysis_period": date_range,
            "total_posts_analyzed": total_posts
        }
    
    print(f"Consolidation completed in {time.time() - start_time:.2f} seconds")
    return consolidated_topics

def direct_llm_analysis(conn, recent_df, batch_size=15000):
    """
    Approach 1: Direct LLM Analysis
    
    Uses Gemini to analyze a sample of posts and extract trending topics directly.
    This enhanced version processes data in batches and then consolidates the results.
    
    Args:
        conn: DuckDB connection
        recent_df: DataFrame with cleaned posts
        batch_size: Number of posts per batch (default=15000)
        
    Returns:
        dict: Structured trending topics result
    """
    print("Setting up batch analysis for complete dataset...")
    start_time = time.time()
    
    # Calculate number of batches needed to cover the entire dataset
    total_records = len(recent_df)
    num_batches = (total_records + batch_size - 1) // batch_size  # Ceiling division
    
    print(f"Processing {total_records} posts in {num_batches} batches of {batch_size}...")
    
    # Load environment variables from .env file for API key
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("Warning: python-dotenv not installed. Install with 'pip install python-dotenv'")
        
    api_key = os.environ.get('GOOGLE_API_KEY')
    
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    
    # Get the date range for the entire dataset
    date_range = f"{recent_df['datetime'].min().strftime('%Y-%m-%d')} to {recent_df['datetime'].max().strftime('%Y-%m-%d')}"
    
    # Process each batch
    batch_results = []
    for batch_num in range(1, num_batches + 1):
        # Calculate batch start and end indices
        start_idx = (batch_num - 1) * batch_size
        end_idx = min(start_idx + batch_size, total_records)
        
        # Extract batch dataframe
        batch_df = recent_df.iloc[start_idx:end_idx].copy()
        
        # Process this batch
        batch_result = process_batch(conn, batch_df, batch_num, batch_size, date_range, num_batches, api_key)
        batch_results.append(batch_result)
    
    # Consolidate results from all batches
    final_results = consolidate_results(batch_results, date_range, total_records, api_key)
    
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    # Save raw response for debugging/analysis
    with open('output/llm_response_raw.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Save in the format expected by downstream processes
    with open('output/approach1_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Log topic information
    if 'topics' in final_results:
        num_topics = len(final_results['topics'])
        print(f"\nSuccessfully identified {num_topics} consolidated trending topics:")
        for i, topic in enumerate(final_results['topics']):
            print(f"Topic {i+1}: {topic['name']}")
            print(f"  Estimated percentage: {topic['estimated_percentage']}")
            print(f"  Engagement level: {topic['engagement_level']}")
            print(f"  Key terms: {', '.join([term['term'] for term in topic['key_terms'][:5]])}")
            print(f"  Key entities: {', '.join([entity['name'] for entity in topic['key_entities'][:3]])}")
            print()
            
        if num_topics > 5:
            print("NOTE: The consolidation selected more than 5 topics. The top 5 are recommended for the Explore tab.")
    
    # Create a merged DataFrame with all sampled posts from all batches for finding exemplars
    all_sampled_posts = []
    for batch_num in range(1, num_batches + 1):
        try:
            batch_samples = conn.execute(f"""
            SELECT * FROM batch_{batch_num}_samples
            """).df()
            all_sampled_posts.append(batch_samples)
        except Exception as e:
            print(f"Warning: Could not retrieve samples for batch {batch_num}: {e}")
    
    # Combine all samples if we have any
    if all_sampled_posts:
        combined_samples = pd.concat(all_sampled_posts, ignore_index=True)
        
        # Find exemplar posts for each final topic
        if 'topics' in final_results:
            topic_exemplars = {}
            for i, topic in enumerate(final_results['topics']):
                exemplars = find_exemplar_posts_for_topic(topic, combined_samples)
                topic_exemplars[topic['name']] = exemplars
            
            # Save exemplars for later use
            with open('output/topic_exemplar_posts.json', 'w') as f:
                exemplar_data = {
                    topic_name: [
                        {
                            'text': post['Text'],
                            'datetime': post['datetime'] if isinstance(post['datetime'], str) else str(post['datetime']),
                            'likes': int(post['likes_count']) if pd.notna(post['likes_count']) else 0,
                            'recasts': int(post['recasts_count']) if pd.notna(post['recasts_count']) else 0
                        } for post in posts
                    ] for topic_name, posts in topic_exemplars.items()
                }
                json.dump(exemplar_data, f, indent=2)
    
    print(f"\nTotal processing time: {time.time() - start_time:.2f} seconds")
    return final_results

def find_exemplar_posts_for_topic(topic, posts_df, n=3):
    """Find exemplar posts that best represent a given topic."""
    # Create a simple keyword matching function
    topic_keywords = [term['term'].lower() for term in topic['key_terms']]
    
    # Score each post based on keyword matches and engagement
    scores = []
    for _, row in posts_df.iterrows():
        text = str(row['Text']).lower()
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
    import duckdb
    import os
    import pandas as pd
    from datetime import datetime, timedelta
    
    # Load environment variables for API key
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("Successfully loaded environment variables from .env file")
    except ImportError:
        print("Warning: python-dotenv not installed. Install with 'pip install python-dotenv'")
    
    api_key = os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        print("WARNING: GOOGLE_API_KEY environment variable not set. The script may fail.")
    else:
        print("Found GOOGLE_API_KEY in environment variables")
    
    print("Running direct LLM analysis with Gemini...")
    
    # Setup DuckDB connection
    conn = duckdb.connect(database=':memory:')
    conn.execute("SET memory_limit='4GB'")
    
    # For testing purposes, we'll limit the number of batches
    # Setting to True will use the full dataset, False will use only a small portion
    use_full_dataset = True
    
    # Load example data for testing
    # This assumes there's a parquet file in output/interim_data directory
    # If not, it creates a minimal test DataFrame
    try:
        parquet_path = 'output/interim_data/cleaned_data.parquet'
        if os.path.exists(parquet_path):
            print(f"Loading test data from {parquet_path}...")
            # Load the data - for testing, we can load a smaller subset
            if use_full_dataset:
                test_df = pd.read_parquet(parquet_path)
                batch_size = 15000  # Use standard batch size for full dataset
                print("Using full dataset for complete analysis...")
            else:
                # Load only a small portion (e.g., 5000 rows) for testing the batch processing
                print("Using limited dataset for testing batch processing...")
                test_df = pd.read_parquet(parquet_path)
                # Ensure we have enough data for at least 2 smaller batches
                if len(test_df) > 3000:
                    test_df = test_df.iloc[:3000]
                    # Adjust batch size for testing
                    batch_size = 1500  # This will give us exactly 2 batches
                else:
                    batch_size = 1500  # Default test batch size
                
            conn.register('cleaned_casts', test_df)
        else:
            print("No test data found, creating minimal test data...")
            # Create minimal test dataframe
            test_data = [
                {"Text": "This is a test post about AI and crypto", "datetime": datetime.now(), 
                 "likes_count": 10, "recasts_count": 5, "engagement_score": 50},
                {"Text": "Farcaster is growing quickly this month", "datetime": datetime.now() - timedelta(hours=1), 
                 "likes_count": 15, "recasts_count": 8, "engagement_score": 75},
                {"Text": "New NFT collection dropping next week", "datetime": datetime.now() - timedelta(hours=2), 
                 "likes_count": 20, "recasts_count": 10, "engagement_score": 100}
            ]
            test_df = pd.DataFrame(test_data)
            test_df['datetime'] = pd.to_datetime(test_df['datetime'])
            test_df['Hash'] = [f"hash{i}" for i in range(len(test_df))]
            test_df['Fid'] = [f"user{i}" for i in range(len(test_df))]
            test_df['ParentCastId'] = None
            conn.register('cleaned_casts', test_df)
            batch_size = 2  # Very small batch size for minimal test data
        
        # Run the analysis
        print(f"Analyzing {len(test_df)} posts with Gemini in batches of {batch_size}...")
        results = direct_llm_analysis(conn, test_df, batch_size)
        
        print("\n===== ANALYSIS RESULTS =====\n")
        if 'topics' in results:
            num_topics = len(results['topics'])
            
            print(f"Identified {num_topics} trending topics for the period {results['analysis_period']}:")
            print()
            
            for i, topic in enumerate(results['topics']):
                print(f"Topic {i+1}: {topic['name']}")
                print(f"  Explanation: {topic['explanation']}")
                print(f"  Estimated percentage: {topic['estimated_percentage']}")
                print(f"  Engagement level: {topic['engagement_level']}")
                print(f"  Key terms: {', '.join([t['term'] for t in topic['key_terms'][:3]])}")
                print(f"  Key entities: {', '.join([e['name'] for e in topic['key_entities'][:3]])}")
                print()
            
            print(f"Total topics identified: {num_topics}")
            print(f"Analysis period: {results['analysis_period']}")
            print(f"Total posts analyzed: {results['total_posts_analyzed']}")
        else:
            print("No topics found in the results.")
    
    except Exception as e:
        print(f"Error running the analysis: {e}")
        import traceback
        traceback.print_exc()