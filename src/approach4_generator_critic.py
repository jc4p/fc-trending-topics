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

"""
Approach 4: Generator-Critic Model for Trending Topic Detection

This implementation uses a generator-critic architecture for trending topic detection:

1. Generator Model (Gemini 2.0 Flash Lite):
   - Processes batches of posts
   - Generates diverse candidate trending topics with creative temperature settings
   - Produces more topics than needed to allow for filtering

2. Critic Model (Gemini 2.5 Pro Preview):
   - Evaluates each candidate topic on multiple criteria
   - Assigns scores for specificity, novelty, timeliness, engagement potential, viewer appeal
   - Provides detailed reasoning and improvement suggestions
   - Identifies and filters out baseline topics

3. Consolidation Process:
   - Only high-quality topics (score >= 7) are considered
   - Final topics are selected based on overall quality, diversity, and appeal

This approach combines the creative generation capabilities of one model with the
critical evaluation powers of a more advanced model to produce high-quality trending topics.
"""

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

class TopicEvaluation(TypedDict):
    specificity: int  # 1-10
    novelty: int  # 1-10
    timeliness: int  # 1-10
    engagement_potential: int  # 1-10
    viewer_appeal: int  # 1-10
    overall: int  # 1-10
    reasoning: str
    improvement_suggestions: list[str]
    is_baseline_topic: bool

def process_batch_generator(conn, batch_df, batch_number, batch_size, date_range, total_batches, api_key, output_prefix=""):
    """
    Process a single batch of posts to generate candidate trending topics.
    
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
    print(f"\nProcessing batch {batch_number}/{total_batches} with Generator ({len(batch_df)} posts)...")
    
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
    
    # Initialize Gemini for this batch - use Gemini 2.0 Flash Lite for generation (faster for more training cycles)
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
            temperature=0.9,  # Using higher temperature for more creative topic generation
            response_mime_type="application/json"
        )
    )
    
    # Parse and process the batch response
    try:
        # Save the raw response for debugging
        debug_dir = os.path.join('output', 'debug')
        os.makedirs(debug_dir, exist_ok=True)
        with open(os.path.join(debug_dir, f'batch_{batch_number}_raw_response.txt'), 'w') as f:
            f.write(batch_response.text)
        
        # Try to parse as JSON
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
        print(f"Batch {batch_number}: Attempting regex-based parsing as fallback...")
        
        # Save the full error for debugging
        with open(os.path.join(debug_dir, f'batch_{batch_number}_json_error.txt'), 'w') as f:
            f.write(f"Error: {str(e)}\n\n")
            f.write(f"Raw response: {batch_response.text}")
        
        # Try to extract topics using regex
        import re
        topics = []
        
        # Extract each topic - look for patterns like "name": "Topic Name", etc.
        topic_matches = re.findall(r'{\s*"name":\s*"([^"]+)".*?}', batch_response.text, re.DOTALL)
        
        if topic_matches:
            print(f"Batch {batch_number}: Found {len(topic_matches)} topic matches with regex")
            
            # For each potential topic match, try to extract the key fields
            for topic_idx, topic_text in enumerate(topic_matches):
                # Find the full topic object text
                topic_start = batch_response.text.find('{', batch_response.text.find(f'"name": "{topic_text}"') - 50)
                if topic_start == -1:
                    continue
                
                # Find the matching closing brace using bracket counting
                bracket_count = 1
                topic_end = topic_start + 1
                while bracket_count > 0 and topic_end < len(batch_response.text):
                    if batch_response.text[topic_end] == '{':
                        bracket_count += 1
                    elif batch_response.text[topic_end] == '}':
                        bracket_count -= 1
                    topic_end += 1
                
                if bracket_count != 0:
                    # Didn't find a properly matched closing brace, so just look for the next closing brace
                    next_brace = batch_response.text.find('}', topic_start + 1)
                    if next_brace != -1:
                        topic_end = next_brace + 1
                    else:
                        continue
                
                # Extract the full topic text
                full_topic_text = batch_response.text[topic_start:topic_end]
                
                # Try to extract fields using regex
                try:
                    name_match = re.search(r'"name":\s*"([^"]+)"', full_topic_text)
                    explanation_match = re.search(r'"explanation":\s*"([^"]+)"', full_topic_text)
                    percentage_match = re.search(r'"estimated_percentage":\s*"([^"]+)"', full_topic_text)
                    engagement_match = re.search(r'"engagement_level":\s*"([^"]+)"', full_topic_text)
                    
                    # Create a basic topic object with extracted fields
                    topic = {
                        "name": name_match.group(1) if name_match else f"Topic {topic_idx + 1}",
                        "explanation": explanation_match.group(1) if explanation_match else "No explanation available",
                        "estimated_percentage": percentage_match.group(1) if percentage_match else "0%",
                        "engagement_level": engagement_match.group(1) if engagement_match else "Medium",
                        "key_terms": [],
                        "key_entities": []
                    }
                    
                    # Try to extract key_terms array
                    key_terms_match = re.search(r'"key_terms":\s*(\[.*?\])', full_topic_text, re.DOTALL)
                    if key_terms_match:
                        # Extract term objects from the array
                        term_matches = re.findall(r'{\s*"term":\s*"([^"]+)".*?}', key_terms_match.group(1), re.DOTALL)
                        for term in term_matches:
                            # Try to extract frequency
                            freq_match = re.search(fr'"term":\s*"{re.escape(term)}".*?"frequency":\s*(\d+)', full_topic_text, re.DOTALL)
                            frequency = int(freq_match.group(1)) if freq_match else 1
                            
                            topic["key_terms"].append({
                                "term": term,
                                "frequency": frequency
                            })
                    
                    # Try to extract key_entities array
                    key_entities_match = re.search(r'"key_entities":\s*(\[.*?\])', full_topic_text, re.DOTALL)
                    if key_entities_match:
                        # Extract entity objects from the array
                        entity_matches = re.findall(r'{\s*"name":\s*"([^"]+)".*?}', key_entities_match.group(1), re.DOTALL)
                        for entity in entity_matches:
                            # Try to extract type and relevance
                            type_match = re.search(fr'"name":\s*"{re.escape(entity)}".*?"type":\s*"([^"]+)"', full_topic_text, re.DOTALL)
                            relevance_match = re.search(fr'"name":\s*"{re.escape(entity)}".*?"relevance":\s*"([^"]+)"', full_topic_text, re.DOTALL)
                            
                            topic["key_entities"].append({
                                "name": entity,
                                "type": type_match.group(1) if type_match else "Unknown",
                                "relevance": relevance_match.group(1) if relevance_match else "Medium"
                            })
                    
                    topics.append(topic)
                except Exception as regex_error:
                    print(f"Batch {batch_number}: Error extracting topic {topic_idx + 1}: {regex_error}")
                    # Log the error and the text we were trying to parse
                    with open(os.path.join(debug_dir, f'batch_{batch_number}_topic_{topic_idx}_regex_error.txt'), 'w') as f:
                        f.write(f"Error: {str(regex_error)}\n\n")
                        f.write(f"Topic text: {full_topic_text}")
        
        # Create a topics object with the extracted topics
        batch_topics = {
            "topics": topics,
            "analysis_period": date_range,
            "total_posts_analyzed": len(formatted_casts),
            "batch_number": batch_number
        }
        
        # If regex fallback found topics, report success
        if len(topics) > 0:
            print(f"Batch {batch_number}: Successfully extracted {len(topics)} topics using regex fallback")
        else:
            print(f"Batch {batch_number}: No topics could be extracted with regex fallback")
            
        # Save the processed topics for debugging
        with open(os.path.join(debug_dir, f'batch_{batch_number}_regex_processed.json'), 'w') as f:
            json.dump(batch_topics, f, indent=2)
    
    # Create output/cache directory if it doesn't exist
    os.makedirs('output/cache', exist_ok=True)
    
    # Save batch results for later evaluation
    with open(f'output/cache/batch_{batch_number}_results_gen.json', 'w') as f:
        json.dump(batch_topics, f, indent=2)
    
    print(f"Batch {batch_number} generator completed in {time.time() - batch_start_time:.2f} seconds")
    return batch_topics


def evaluate_topics(topics, date_range, api_key):
    """
    Use Gemini 2.5 Pro Preview as the critic to evaluate the topics generated by Gemini 2.0 Flash Lite.
    Evaluates all topics in a single batch call to reduce API call overhead.
    
    Args:
        topics: List of topic dictionaries from the generator
        date_range: String representing the date range for the entire dataset
        api_key: Google API key for Gemini
        
    Returns:
        list: Topics with evaluation scores
    """
    print(f"\nEvaluating {len(topics)} topics in batch with the Critic model...")
    start_time = time.time()
    
    # Skip if no topics to evaluate
    if len(topics) == 0:
        print("No topics to evaluate.")
        return []
    
    # Initialize Gemini 2.5 Pro for evaluation
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    critic_model = GenerativeModel('gemini-2.5-pro-preview-03-25')
    
    # Format all topics into a single JSON array for batch evaluation
    topics_json = json.dumps(topics, indent=2)
    
    # Create batch evaluation prompt
    critic_prompt = f"""
    As an expert trend evaluator for Farcaster social network, critically evaluate these {len(topics)} candidate trending topics 
    from the period {date_range}.
    
    TOPIC DATA:
    {topics_json}
    
    Carefully assess each topic's quality as a genuine trending topic using these critical criteria:
    
    1. Specificity (1-10): How specific vs. generic is this topic?
       - Score 1-3: Extremely generic, basic category (e.g., general technology discussions)
       - Score 4-6: Somewhat specific but still broadly defined (e.g., activity within a category)
       - Score 7-10: Highly specific, clearly defined (e.g., particular feature or behavioral pattern)
    
    2. Novelty (1-10): How novel vs. common/baseline is this topic?
       - Score 1-3: Constant topic that's always discussed (e.g., recurring technical topics)
       - Score 4-6: Recurring topic that appears periodically (e.g., cyclical user activities)
       - Score 7-10: Genuinely new platform-wide phenomenon (e.g., first-time emergent behaviors)
    
    3. Timeliness (1-10): How time-bound vs. evergreen is this topic?
       - Score 1-3: Always relevant, no temporal specificity (e.g., general concepts)
       - Score 4-6: Generally relevant but with current uptick (e.g., recurring themes with new activity)
       - Score 7-10: Highly time-specific event or activity (e.g., response to a recent change)
    
    4. Engagement Potential (1-10): How likely to drive conversation?
       - Score 1-3: Limited engagement, few follow-ups (e.g., announcement-type content)
       - Score 4-6: Moderate engagement, some discussion (e.g., interesting but not provocative content)
       - Score 7-10: High engagement, provokes meaningful platform-wide discussion (e.g., topics that inspire action)
    
    5. Viewer Appeal (1-10): How interesting to lurkers/readers?
       - Score 1-3: Primarily interests insiders or niche users (e.g., highly technical discussions)
       - Score 4-6: Moderate appeal to regular users (e.g., content requiring platform familiarity)
       - Score 7-10: Broad appeal to general audience (e.g., visually engaging or immediately understandable content)
    
    CRITICAL EVALUATION PRINCIPLES:
    
    1. Third-Party Tool Detection: Carefully examine each topic for signs it might be about a specific third-party tool, 
       mini-app, or game rather than a platform-wide phenomenon. Look for these patterns:
       - References to specific named games or challenges where users compete or track progress
       - Tools that augment or extend the platform's functionality for specific purposes
       - Apps that provide specific functionality to a subset of users
       - Communities, channels or DAOs that primarily serve a specific niche audience
       
       Topics that appear tied to specific third-party tools should receive significantly reduced scores 
       (typically 3-5 range for novelty) and lower overall scores, as they often represent promotional 
       content rather than organic platform-wide trends.
    
    2. Ephemeral Trend Assessment: Identify topics that represent temporary viral trends or memes by looking for:
       - Rapid spread of similar profile picture styles or visual modifications
       - Inside jokes that require specific context to understand
       - Short-lived behavioral mimicry where users copy a specific action or phrase
       - Viral challenges that follow predictable patterns seen on other platforms
       
       While these topics can be engaging, they're typically short-lived and limited in lasting impact.
       They should generally score no higher than 5-6 in novelty unless they represent a truly 
       transformational shift in how users express themselves on the platform.
    
    3. Platform Evolution Priority: Identify and prioritize topics that reveal how the platform itself is evolving:
       - New core functionality that changes how all users can interact
       - Emergent behaviors that organically utilize platform features in novel ways
       - Platform-wide cultural shifts that affect the general user experience
       - Changes to interaction patterns that reflect the unique nature of this specific platform
       
       These topics deserve the highest scores (8-10) as they demonstrate genuine platform-specific 
       innovation and evolution that affects the broader user base rather than niche communities.
    
    Based on these criteria, determine for EACH topic:
    1. An overall score (1-10)
    2. Detailed reasoning for your assessment, including any mini-app or meme penalties applied
    3. Whether this is a baseline topic that should be filtered out
    4. Specific suggestions for how this topic could be improved
    
    Return your evaluation as a JSON array where each item corresponds to a topic in the same order as provided,
    with each evaluation following this TypedDict schema:
    
    class TopicEvaluation(TypedDict):
        topic_name: str  # The name of the topic being evaluated
        specificity: int  # 1-10
        novelty: int  # 1-10
        timeliness: int  # 1-10
        engagement_potential: int  # 1-10
        viewer_appeal: int  # 1-10
        overall: int  # 1-10
        reasoning: str
        improvement_suggestions: list[str]
        is_baseline_topic: bool
    
    Your response must be a valid JSON array containing one evaluation object for each topic.
    """

    print("Sending batch evaluation request to critic model...")
    
    # Get batch evaluation with JSON formatting
    critic_response = critic_model.generate_content(
        critic_prompt,
        generation_config=types.GenerationConfig(
            temperature=0,  # Zero temperature for consistent evaluation
            response_mime_type="application/json"
        )
    )
    
    # Parse and process the critic response
    try:
        # Save the raw critic response for debugging
        debug_dir = os.path.join('output', 'debug')
        os.makedirs(debug_dir, exist_ok=True)
        with open(os.path.join(debug_dir, f'critic_raw_response.txt'), 'w') as f:
            f.write(critic_response.text)
        
        # Try standard JSON parsing first
        evaluations = json.loads(critic_response.text)
        
        # Handle case where response might be wrapped in an additional object
        if not isinstance(evaluations, list) and "evaluations" in evaluations:
            evaluations = evaluations["evaluations"]
        
        if not isinstance(evaluations, list):
            print(f"Unexpected response format: {type(evaluations)}")
            print("Converting to expected format...")
            # Try to extract a list from whatever we got
            if hasattr(evaluations, "values") and callable(getattr(evaluations, "values")):
                evaluations = list(evaluations.values())
            else:
                # If still not a list, create a default list with one item
                evaluations = [evaluations]
        
        print(f"Received {len(evaluations)} evaluations from critic model")
        
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error parsing critic evaluations: {e}")
        print("Attempting regex-based parsing for critic evaluations...")
        
        # Save the json error
        with open(os.path.join(debug_dir, f'critic_json_error.txt'), 'w') as f:
            f.write(f"Error: {str(e)}\n\n")
            f.write(f"Raw response: {critic_response.text}")
        
        # Try to extract evaluations using regex
        import re
        evaluations = []
        
        # Look for evaluation objects in the response
        evaluation_blocks = re.findall(r'(?:{\s*"topic_name"|{\s*"specificity").*?}(?=,\s*{|\s*]|\s*$)', critic_response.text, re.DOTALL)
        
        if evaluation_blocks:
            print(f"Found {len(evaluation_blocks)} evaluation blocks with regex")
            
            for eval_idx, eval_text in enumerate(evaluation_blocks):
                try:
                    # Try to extract key fields with regex
                    topic_name_match = re.search(r'"topic_name":\s*"([^"]+)"', eval_text)
                    specificity_match = re.search(r'"specificity":\s*(\d+)', eval_text)
                    novelty_match = re.search(r'"novelty":\s*(\d+)', eval_text)
                    timeliness_match = re.search(r'"timeliness":\s*(\d+)', eval_text)
                    engagement_match = re.search(r'"engagement_potential":\s*(\d+)', eval_text)
                    viewer_appeal_match = re.search(r'"viewer_appeal":\s*(\d+)', eval_text)
                    overall_match = re.search(r'"overall":\s*(\d+)', eval_text)
                    reasoning_match = re.search(r'"reasoning":\s*"([^"]+)"', eval_text)
                    baseline_match = re.search(r'"is_baseline_topic":\s*(true|false)', eval_text)
                    
                    # Create an evaluation object with extracted data
                    evaluation = {
                        "topic_name": topic_name_match.group(1) if topic_name_match else f"Topic {eval_idx + 1}",
                        "specificity": int(specificity_match.group(1)) if specificity_match else 5,
                        "novelty": int(novelty_match.group(1)) if novelty_match else 5,
                        "timeliness": int(timeliness_match.group(1)) if timeliness_match else 5,
                        "engagement_potential": int(engagement_match.group(1)) if engagement_match else 5,
                        "viewer_appeal": int(viewer_appeal_match.group(1)) if viewer_appeal_match else 5,
                        "overall": int(overall_match.group(1)) if overall_match else 5,
                        "reasoning": reasoning_match.group(1) if reasoning_match else "No reasoning provided",
                        "is_baseline_topic": baseline_match.group(1).lower() == "true" if baseline_match else False,
                        "improvement_suggestions": []
                    }
                    
                    # Try to extract improvement suggestions if present
                    suggestions_match = re.search(r'"improvement_suggestions":\s*(\[.*?\])', eval_text, re.DOTALL)
                    if suggestions_match:
                        suggestions_text = suggestions_match.group(1)
                        suggestion_items = re.findall(r'"([^"]+)"', suggestions_text)
                        evaluation["improvement_suggestions"] = suggestion_items
                    
                    evaluations.append(evaluation)
                    
                except Exception as regex_error:
                    print(f"Error extracting evaluation {eval_idx + 1}: {regex_error}")
                    # Log the error and text for debugging
                    with open(os.path.join(debug_dir, f'critic_eval_{eval_idx}_regex_error.txt'), 'w') as f:
                        f.write(f"Error: {str(regex_error)}\n\n")
                        f.write(f"Evaluation text: {eval_text}")
            
            print(f"Successfully extracted {len(evaluations)} evaluations using regex")
        else:
            print("No evaluation blocks found with regex")
            # Create default evaluations for each topic
            evaluations = []
    
    # Match evaluations with topics and combine them
    evaluated_topics = []
    
    # Create a mapping of topic names to evaluations for better matching
    eval_map = {}
    for eval_item in evaluations:
        if isinstance(eval_item, dict) and "topic_name" in eval_item:
            eval_map[eval_item["topic_name"]] = eval_item
    
    # Match by topic name if possible, otherwise use index order
    for i, topic in enumerate(topics):
        topic_with_eval = topic.copy()
        topic_name = topic.get('name', '')
        
        # First try to find evaluation by matching topic name
        if topic_name in eval_map:
            evaluation = eval_map[topic_name]
            # Extract evaluation without topic_name to maintain compatibility
            eval_copy = {k: v for k, v in evaluation.items() if k != 'topic_name'}
        # Fall back to index-based matching
        elif i < len(evaluations):
            evaluation = evaluations[i]
            # Extract evaluation without topic_name to maintain compatibility
            eval_copy = {k: v for k, v in evaluation.items() if k != 'topic_name'}
        # No matching evaluation found
        else:
            print(f"Topic {i+1}: {topic_name} - No evaluation received")
            # Create a default evaluation
            eval_copy = {
                "specificity": 5,
                "novelty": 5,
                "timeliness": 5,
                "engagement_potential": 5,
                "viewer_appeal": 5,
                "overall": 5,
                "reasoning": "No evaluation available",
                "improvement_suggestions": [],
                "is_baseline_topic": False
            }
            topic_with_eval['evaluation'] = eval_copy
            evaluated_topics.append(topic_with_eval)
            continue
        
        topic_with_eval['evaluation'] = eval_copy
        
        # Log the evaluation summary
        print(f"Topic {i+1}: {topic_name}")
        print(f"  - Overall score: {eval_copy.get('overall', 'N/A')}/10")
        print(f"  - Is baseline topic: {eval_copy.get('is_baseline_topic', 'N/A')}")
        print(f"  - Specificity: {eval_copy.get('specificity', 'N/A')}/10")
        print(f"  - Novelty: {eval_copy.get('novelty', 'N/A')}/10")
        
        evaluated_topics.append(topic_with_eval)
    
    # Save the final evaluations for debugging
    with open(os.path.join(debug_dir, 'final_evaluations.json'), 'w') as f:
        json.dump(evaluated_topics, f, indent=2)
    
    # Sort topics by overall evaluation score (if available)
    sorted_topics = sorted(
        evaluated_topics,
        key=lambda t: t.get('evaluation', {}).get('overall', 0),
        reverse=True
    )
    
    print(f"Topic evaluation completed in {time.time() - start_time:.2f} seconds")
    return sorted_topics


def filter_and_consolidate_topics(evaluated_topics, date_range, total_posts, api_key):
    """
    Filter out baseline topics and consolidate the remaining high-quality topics.
    Stores evaluation data for potential reinforcement learning training.
    
    Args:
        evaluated_topics: List of topics with evaluation scores
        date_range: String representing the date range
        total_posts: Total number of posts analyzed
        api_key: Google API key for Gemini
        
    Returns:
        dict: Final consolidated trending topics
    """
    print("\nFiltering and consolidating evaluated topics...")
    start_time = time.time()
    
    # Save all evaluations for future RL training
    os.makedirs('output/training_data', exist_ok=True)
    
    # Create training data with scores bucketed into categories
    training_data = []
    
    score_buckets = {
        'excellent': (9, 10),   # 9-10: Excellent trending topics
        'good': (7, 8),         # 7-8: Good trending topics
        'mediocre': (5, 6),     # 5-6: Mediocre topics
        'poor': (3, 4),         # 3-4: Poor topics
        'baseline': (1, 2)      # 1-2: Baseline/generic topics
    }
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    for topic in evaluated_topics:
        # Get evaluation data
        eval_data = topic.get('evaluation', {})
        topic_name = topic.get('name', 'unknown')
        overall_score = eval_data.get('overall', 0)
        
        # Determine score bucket
        score_category = 'unknown'
        for category, (min_score, max_score) in score_buckets.items():
            if min_score <= overall_score <= max_score:
                score_category = category
                break
        
        # Create training example
        training_example = {
            'topic_data': topic,
            'score': overall_score,
            'score_category': score_category,
            'is_baseline': eval_data.get('is_baseline_topic', False),
            'evaluation_metrics': {
                'specificity': eval_data.get('specificity', 0),
                'novelty': eval_data.get('novelty', 0),
                'timeliness': eval_data.get('timeliness', 0),
                'engagement_potential': eval_data.get('engagement_potential', 0),
                'viewer_appeal': eval_data.get('viewer_appeal', 0),
            },
            'reasoning': eval_data.get('reasoning', ''),
            'date_evaluated': timestamp
        }
        
        training_data.append(training_example)
    
    # Save training data for future RL use
    with open(f'output/training_data/{args.output_prefix}topic_evaluations_{timestamp}.json', 'w') as f:
        json.dump(training_data, f, indent=2)
    
    print(f"Saved {len(training_data)} topic evaluations for future RL training")
    
    # Analyze score distribution for this batch
    score_distribution = {}
    for category in score_buckets:
        count = sum(1 for example in training_data if example['score_category'] == category)
        score_distribution[category] = count
    
    print("Score distribution:")
    for category, count in score_distribution.items():
        print(f"  {category}: {count} topics")
    
    # Filter out topics marked as baseline
    filtered_topics = [
        topic for topic in evaluated_topics 
        if not topic.get('evaluation', {}).get('is_baseline_topic', False)
    ]
    
    print(f"Filtered {len(evaluated_topics) - len(filtered_topics)} baseline topics, {len(filtered_topics)} remain")
    
    # Only keep topics with an overall score >= 7 (high quality)
    quality_topics = [
        topic for topic in filtered_topics
        if topic.get('evaluation', {}).get('overall', 0) >= 7
    ]
    
    print(f"Found {len(quality_topics)} high-quality topics (score >= 7)")
    
    # If we have more than 5 quality topics, use consolidation to get the final 5
    if len(quality_topics) > 5:
        # Initialize Gemini for consolidation with Gemini 2.5 Pro
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        consolidation_model = GenerativeModel('gemini-2.5-pro-preview-03-25')
        
        # Create consolidated topics in JSON format
        topics_json = json.dumps(quality_topics, indent=2)
        
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
        
        Each topic includes both its raw data and expert evaluation scores. Use both to make your selections,
        strongly prioritizing topics with higher evaluation scores.
        
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
        
        TOPIC CANDIDATES WITH EVALUATIONS:
        {topics_json}
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
    else:
        # If we have 5 or fewer quality topics, use them directly
        consolidated_topics = {
            "topics": [
                {k: v for k, v in topic.items() if k != 'evaluation'} for topic in quality_topics
            ],
            "analysis_period": date_range,
            "total_posts_analyzed": total_posts
        }
    
    print(f"Topic consolidation completed in {time.time() - start_time:.2f} seconds")
    return consolidated_topics


def generator_critic_analysis(conn, recent_df, batch_size=10000, training_iterations=1):
    """
    Approach 4: Generator-Critic Analysis
    
    Uses Gemini 2.0 Flash Lite to generate candidates and Gemini 2.5 Pro Preview to evaluate and filter topics.
    This approach processes data in batches, evaluates topics with a critic model, and consolidates.
    
    The implementation includes automated collection of evaluation data for future reinforcement learning:
    - Stores detailed evaluation metrics for all topics
    - Categorizes topics into score buckets (excellent, good, mediocre, poor, baseline)
    - Saves reasoning and improvement suggestions from the critic
    - Creates timestamped JSON files that can be used to train RL models
    
    Checkpoint capabilities enable resuming processing from a previous run:
    - Saves checkpoint data after each batch is processed
    - Can resume processing from any batch
    - Handles both generation and evaluation phases separately
    - Restores state completely when resuming
    
    Args:
        conn: DuckDB connection
        recent_df: DataFrame with cleaned posts
        batch_size: Number of posts per batch (default=15000)
        training_iterations: Number of generator-critic cycles to run (default=1)
                            Future enhancement: Implement multiple iterations where the generator
                            learns from critic feedback to improve topic quality over time
        
    Returns:
        dict: Structured trending topics result
    """
    print("Setting up Generator-Critic analysis for complete dataset...")
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
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs('output/checkpoints', exist_ok=True)
    
    # Add output prefix if specified
    prefix = f"{args.output_prefix}_" if args.output_prefix else ""
    
    # Checkpoint filenames
    generator_checkpoint_file = f'output/checkpoints/{prefix}generator_checkpoint.json'
    evaluation_checkpoint_file = f'output/checkpoints/{prefix}evaluation_checkpoint.json'
    
    # Check if we should resume from a checkpoint
    batch_topics = []
    start_batch = 1
    
    # Try to load generator checkpoint if it exists and --resume flag is set
    if args.resume and os.path.exists(generator_checkpoint_file):
        try:
            with open(generator_checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                
            batch_topics = checkpoint_data.get('batch_topics', [])
            start_batch = checkpoint_data.get('next_batch', 1)
            
            print(f"Resuming generation from batch {start_batch}/{num_batches}")
            print(f"Loaded {len(batch_topics)} topics from generator checkpoint")
        except Exception as e:
            print(f"Error loading generator checkpoint: {e}")
            print("Starting fresh from batch 1")
            start_batch = 1
            batch_topics = []
    else:
        print("Starting fresh generator process from batch 1")
    
    # Skip generation entirely if all batches are already processed
    if start_batch <= num_batches:
        print(f"Processing {num_batches - start_batch + 1} remaining batches with the generator model...")
        
        # Set up runtime limit tracking
        runtime_limit_seconds = args.max_runtime * 60 if args.max_runtime > 0 else float('inf')
        runtime_start = time.time()
        
        # Process each batch with the generator model
        for batch_num in range(start_batch, num_batches + 1):
            # Check if we've hit the runtime limit
            if args.max_runtime > 0 and (time.time() - runtime_start) >= runtime_limit_seconds:
                print(f"\nReached maximum runtime of {args.max_runtime} minutes, checkpointing and exiting...")
                
                # Save checkpoint with current progress
                checkpoint_data = {
                    'batch_topics': batch_topics,
                    'next_batch': batch_num,  # Note: not incrementing batch_num, we'll resume from this batch
                    'total_batches': num_batches,
                    'date_range': date_range,
                    'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'total_topics_so_far': len(batch_topics),
                    'runtime_limit_hit': True
                }
                
                with open(generator_checkpoint_file, 'w') as f:
                    json.dump(checkpoint_data, f, indent=2)
                    
                print(f"Checkpoint saved. Resume with --resume flag to continue from batch {batch_num}")
                print(f"Process exiting due to time limit. {num_batches - batch_num + 1} batches remaining.")
                return {
                    "topics": [],
                    "analysis_period": date_range,
                    "total_posts_analyzed": (batch_num - 1) * batch_size,
                    "status": "incomplete_runtime_limit",
                    "resume_batch": batch_num
                }
            
            batch_start_time = time.time()
            
            # Calculate batch start and end indices
            start_idx = (batch_num - 1) * batch_size
            end_idx = min(start_idx + batch_size, total_records)
            
            # Extract batch dataframe
            batch_df = recent_df.iloc[start_idx:end_idx].copy()
            
            # Process this batch with the generator model
            batch_result = process_batch_generator(conn, batch_df, batch_num, batch_size, date_range, num_batches, api_key, args.output_prefix)
            
            # Extract topics from the batch result
            if 'topics' in batch_result:
                batch_topics.extend(batch_result['topics'])
            
            # Update checkpoint after each batch
            checkpoint_data = {
                'batch_topics': batch_topics,
                'next_batch': batch_num + 1,
                'total_batches': num_batches,
                'date_range': date_range,
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_topics_so_far': len(batch_topics)
            }
            
            # Save checkpoint
            with open(generator_checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            # Calculate and display progress metrics
            batch_time = time.time() - batch_start_time
            elapsed_time = time.time() - start_time
            remaining_batches = num_batches - batch_num
            estimated_remaining_time = (remaining_batches * batch_time) if remaining_batches > 0 else 0
            
            print(f"\nCheckpoint saved after batch {batch_num}/{num_batches}")
            print(f"Total topics collected so far: {len(batch_topics)}")
            print(f"Batch processing time: {batch_time:.2f} seconds")
            print(f"Elapsed time: {elapsed_time:.2f} seconds")
            print(f"Estimated remaining time: {estimated_remaining_time:.2f} seconds ({estimated_remaining_time/60:.2f} minutes)")
    else:
        print("All batches already processed, skipping generation phase")
    
    print(f"\nGenerated {len(batch_topics)} candidate topics across all batches")
    
    # Check if we should resume from evaluation checkpoint
    evaluated_topics = []
    if args.resume and os.path.exists(evaluation_checkpoint_file) and args.skip_evaluation != True:
        try:
            with open(evaluation_checkpoint_file, 'r') as f:
                eval_checkpoint = json.load(f)
                
            evaluated_topics = eval_checkpoint.get('evaluated_topics', [])
            
            print(f"Loaded {len(evaluated_topics)} topics from evaluation checkpoint")
            print("Skipping evaluation phase, using checkpointed evaluations")
        except Exception as e:
            print(f"Error loading evaluation checkpoint: {e}")
            print("Will run evaluation phase")
            evaluated_topics = []
    
    # Only run evaluation if we don't have cached results or skip_evaluation is not set
    if not evaluated_topics and not args.skip_evaluation:
        # Apply critic threshold if specified
        if args.critic_threshold > 0:
            print(f"Applying critic threshold: Only evaluating topics with score >= {args.critic_threshold}")
            # In this version we don't have pre-scores, so this is a placeholder for future iterations
            # where we might have an initial score from the generator
            print("Note: No pre-scoring available in this version - sending all topics to critic")
        
        # Evaluate topics with the critic model
        evaluated_topics = evaluate_topics(batch_topics, date_range, api_key)
        
        # Save evaluation checkpoint
        eval_checkpoint = {
            'evaluated_topics': evaluated_topics,
            'date_range': date_range,
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_topics_evaluated': len(evaluated_topics)
        }
        
        with open(evaluation_checkpoint_file, 'w') as f:
            json.dump(eval_checkpoint, f, indent=2)
        
        print(f"Evaluation checkpoint saved with {len(evaluated_topics)} evaluated topics")
    elif args.skip_evaluation:
        print("Skipping evaluation phase as requested")
        # If we're skipping evaluation but don't have checkpoint data, create default evaluations
        if not evaluated_topics:
            print("Creating default evaluations for all topics")
            evaluated_topics = []
            for topic in batch_topics:
                # Create a default evaluation
                topic_with_eval = topic.copy()
                topic_with_eval['evaluation'] = {
                    "specificity": 7,
                    "novelty": 7,
                    "timeliness": 7,
                    "engagement_potential": 7,
                    "viewer_appeal": 7,
                    "overall": 7,
                    "reasoning": "Default evaluation (evaluation phase skipped)",
                    "improvement_suggestions": [],
                    "is_baseline_topic": False
                }
                evaluated_topics.append(topic_with_eval)
    
    # Filter and consolidate the evaluated topics
    final_results = filter_and_consolidate_topics(evaluated_topics, date_range, total_records, api_key)
    
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    # Save raw response for debugging/analysis
    with open(f'output/{prefix}generator_critic_raw.json', 'w') as f:
        # Include evaluated topics for analysis
        debug_data = {
            "evaluated_topics": evaluated_topics,
            "final_results": final_results
        }
        json.dump(debug_data, f, indent=2)
    
    # Save in the format expected by downstream processes
    with open(f'output/{prefix}approach4_results.json', 'w') as f:
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
    
    # Find exemplar posts for each final topic
    if 'topics' in final_results:
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
            topic_exemplars = {}
            for i, topic in enumerate(final_results['topics']):
                exemplars = find_exemplar_posts_for_topic(topic, combined_samples)
                topic_exemplars[topic['name']] = exemplars
            
            # Save exemplars for later use
            with open(f'output/{prefix}topic_exemplar_posts_gc.json', 'w') as f:
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
    
    # Clean up intermediate checkpoints if option specified
    if args.cleanup_checkpoints:
        try:
            if os.path.exists(generator_checkpoint_file):
                os.remove(generator_checkpoint_file)
            if os.path.exists(evaluation_checkpoint_file):
                os.remove(evaluation_checkpoint_file)
            print("Checkpoint files removed")
        except Exception as e:
            print(f"Warning: Could not remove checkpoint files: {e}")
    
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
    import argparse
    from datetime import datetime, timedelta
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Generator-Critic trending topic analysis')
    parser.add_argument('--batch-size', type=int, default=10000, help='Batch size for processing posts')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of posts to process (0 = use all)')
    parser.add_argument('--offset', type=int, default=0, help='Starting offset in the dataset')
    parser.add_argument('--debug', action='store_true', help='Enable extra debug output')
    parser.add_argument('--output-prefix', type=str, default='', help='Prefix for output files')
    parser.add_argument('--critic-threshold', type=int, default=0, 
                        help='Only send topics with generator score at or above this threshold to critic')
    
    # New checkpoint-related arguments
    parser.add_argument('--resume', action='store_true', 
                        help='Resume processing from checkpoints if they exist')
    parser.add_argument('--skip-evaluation', action='store_true', 
                        help='Skip the evaluation phase (critic model) and use default or checkpointed evaluations')
    parser.add_argument('--cleanup-checkpoints', action='store_true', 
                        help='Remove checkpoint files after successful completion')
    parser.add_argument('--max-runtime', type=int, default=0, 
                        help='Maximum runtime in minutes before graceful checkpoint and exit (0 = no limit)')
    
    args = parser.parse_args()
    
    # Set batch size from command line
    batch_size = args.batch_size
    
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
    
    print(f"Running Generator-Critic analysis with Gemini (batch size: {batch_size})...")
    
    # Setup DuckDB connection
    conn = duckdb.connect(database=':memory:')
    conn.execute("SET memory_limit='4GB'")
    
    # Load data for analysis
    # This assumes there's a parquet file in output/interim_data directory
    # If not, it creates a minimal test DataFrame
    try:
        parquet_path = 'output/interim_data/cleaned_data.parquet'
        if os.path.exists(parquet_path):
            print(f"Loading data from {parquet_path}...")
            
            # First, check the total size of the dataset without loading it all
            # This helps with determining how many posts are available
            total_dataset_size = pd.read_parquet(parquet_path, columns=['datetime']).shape[0]
            print(f"Total available posts in dataset: {total_dataset_size:,}")
            
            # Apply offset and limit if specified
            start_idx = args.offset
            
            # Load the full dataset first, then slice it
            # This is simpler and more reliable than using filters
            test_df = pd.read_parquet(parquet_path)
            
            if args.limit > 0:
                end_idx = start_idx + args.limit
                # Make sure we don't exceed the dataset size
                end_idx = min(end_idx, total_dataset_size)
                actual_limit = end_idx - start_idx
                
                print(f"Processing slice {start_idx}:{end_idx} ({actual_limit:,} posts)...")
                # Slice the dataframe to the desired range
                test_df = test_df.iloc[start_idx:end_idx].copy()
            else:
                if start_idx > 0:
                    # Load from offset to end
                    print(f"Starting from offset {start_idx} to end (processing {total_dataset_size - start_idx:,} posts)...")
                    test_df = test_df.iloc[start_idx:].copy()
                else:
                    # Using entire dataset (already loaded)
                    print(f"Using full dataset for complete analysis ({total_dataset_size:,} posts)...")
            
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
            if batch_size > len(test_df):
                print(f"Batch size ({batch_size}) larger than dataset ({len(test_df)}), adjusting...")
                batch_size = max(1, len(test_df) // 2)  # Ensure at least 2 batches for testing
        
        # Enable debug mode if requested
        if args.debug:
            print("Debug mode enabled - saving detailed output")
            os.makedirs('output/debug', exist_ok=True)
        
        # Run the analysis
        print(f"Analyzing {len(test_df)} posts with the Generator-Critic approach in batches of {batch_size}...")
        results = generator_critic_analysis(conn, test_df, batch_size)
        
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