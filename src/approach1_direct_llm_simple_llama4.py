import random
import time
import json
import pandas as pd
from enum import Enum
from typing_extensions import TypedDict
from datetime import datetime
import os
from typing import List

# Import Groq client
from groq import Groq

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
    key_terms: List[KeyTerm]
    key_entities: List[KeyEntity]
    engagement_level: str  # High, Medium, Low based on likes/recasts

class TrendingTopics(TypedDict):
    topics: List[Topic]
    analysis_period: str
    total_posts_analyzed: int

def direct_llm_analysis(conn, recent_df):
    """
    Approach 1: Direct LLM Analysis
    
    Uses Llama 4 via Groq to analyze a sample of posts and extract trending topics directly.
    
    Args:
        conn: DuckDB connection
        recent_df: DataFrame with cleaned posts
        
    Returns:
        dict: Structured trending topics result
    """
    print("Selecting optimal sample for LLM analysis...")
    start_time = time.time()
    
    # Sample size using Llama 4's large context window (10M tokens)
    # Let's use a much larger sample to take advantage of this
    # Estimate ~150 chars per cast on average, ~37.5 tokens per post (4 chars per token)
    # We can fit around 100k-200k posts theoretically, but let's be conservative
    sample_size = min(50000, len(recent_df))
    
    # Use DuckDB for weight calculation and sampling
    # Create percentile functions for weights
    conn.execute("""
    CREATE OR REPLACE FUNCTION engagement_percentile(x DOUBLE) RETURNS DOUBLE AS
    BEGIN
        RETURN PERCENTILE_CONT(x) WITHIN GROUP (ORDER BY engagement_score);
    END;
    """)
    
    # Calculate weights directly in SQL
    conn.execute("""
    CREATE OR REPLACE VIEW sample_weights AS
    SELECT
        *,
        -- Recency weight (0.1 to 1.1)
        0.1 + ((EXTRACT(EPOCH FROM (datetime - MIN(datetime) OVER())) / 
               EXTRACT(EPOCH FROM (MAX(datetime) OVER() - MIN(datetime) OVER()))))
            AS recency_weight,
        
        -- Engagement weight (0.1 to 1.1)
        -- Cap engagement at 99th percentile to avoid outliers
        0.1 + (LEAST(engagement_score, engagement_percentile(0.99)) / 
              (CASE WHEN MAX(engagement_score) OVER() > 0 
                    THEN engagement_percentile(0.99) ELSE 1 END))
            AS engagement_weight
    FROM cleaned_casts
    """)
    
    # Calculate combined weight and add to view
    conn.execute("""
    CREATE OR REPLACE VIEW combined_weights AS
    SELECT
        *,
        -- Combined weight (60% recency, 40% engagement)
        (0.6 * recency_weight) + (0.4 * engagement_weight) AS combined_weight
    FROM sample_weights
    """)
    
    # To ensure deterministic sampling with weights, use reservoir sampling
    conn.execute("""
    CREATE OR REPLACE VIEW weighted_samples AS
    SELECT
        *,
        -- Generate random weight based on combined_weight
        random() * combined_weight AS sampling_key
    FROM combined_weights
    ORDER BY sampling_key DESC
    LIMIT ?
    """, [sample_size])
    
    # Get the sampled data
    sampled_casts = conn.execute("""
    SELECT * FROM weighted_samples
    """).df()
    
    # Format texts with timestamps and engagement info
    formatted_casts = []
    for _, row in sampled_casts.iterrows():
        dt_str = row['datetime'].strftime('%Y-%m-%d %H:%M:%S') if hasattr(row['datetime'], 'strftime') else str(row['datetime'])
        formatted_text = f"{dt_str} [👍{int(row['likes_count'])}|↗️{int(row['recasts_count'])}]: {row['Text']}"
        formatted_casts.append(formatted_text)
    
    # Get sample statistics
    sample_stats = conn.execute("""
    SELECT
        COUNT(*) AS sample_count,
        AVG(engagement_score) AS avg_engagement,
        AVG(LENGTH("Text")) AS avg_text_length,
        SUM(LENGTH("Text")) AS total_chars
    FROM weighted_samples
    """).fetchone()
    
    print(f"Sampled {sample_stats[0]:,} posts for direct LLM analysis")
    print(f"Average engagement in sample: {sample_stats[1]:.2f}")
    print(f"Average text length: {sample_stats[2]:.1f} chars")
    print(f"Total characters: {sample_stats[3]:,} chars")
    print(f"Sampling completed in {time.time() - start_time:.2f} seconds")
    
    # Calculate sample representativeness
    representation = conn.execute("""
    WITH sample_metrics AS (
        SELECT
            AVG(engagement_score) AS s_avg_engagement,
            MEDIAN(engagement_score) AS s_median_engagement,
            COUNT(DISTINCT Fid) AS s_unique_users,
            SUM(CASE WHEN "ParentCastId" IS NOT NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS s_reply_percentage
        FROM weighted_samples
    ),
    population_metrics AS (
        SELECT
            AVG(engagement_score) AS p_avg_engagement,
            MEDIAN(engagement_score) AS p_median_engagement,
            COUNT(DISTINCT Fid) AS p_unique_users,
            SUM(CASE WHEN "ParentCastId" IS NOT NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS p_reply_percentage
        FROM cleaned_casts
    )
    SELECT
        s_avg_engagement, p_avg_engagement,
        s_median_engagement, p_median_engagement,
        s_unique_users, p_unique_users,
        s_reply_percentage, p_reply_percentage
    FROM sample_metrics, population_metrics
    """).fetchone()
    
    print("Sample representativeness metrics:")
    print(f"  - Engagement: sample avg {representation[0]:.2f} vs population avg {representation[1]:.2f}")
    print(f"  - Unique users: {representation[4]:,} in sample vs {representation[5]:,} in population")
    print(f"  - Reply percentage: {representation[6]:.1f}% in sample vs {representation[7]:.1f}% in population")
    
    # Initialize Groq client
    api_key = os.environ.get('GROQ_API_KEY')
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set")
    
    groq_client = Groq(api_key=api_key)
    
    # Create a prompt with time context and structured output
    date_range = f"{recent_df['datetime'].min().strftime('%Y-%m-%d')} to {recent_df['datetime'].max().strftime('%Y-%m-%d')}"
    
    # Define the schema for the output
    schema = {
        "type": "object",
        "properties": {
            "topics": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Topic name (5 words max)"},
                        "explanation": {"type": "string", "description": "Brief explanation of why trending"},
                        "estimated_percentage": {"type": "string", "description": "Percentage of posts"},
                        "key_terms": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "term": {"type": "string"},
                                    "frequency": {"type": "integer"}
                                },
                                "required": ["term", "frequency"]
                            }
                        },
                        "key_entities": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "type": {"type": "string"},
                                    "relevance": {"type": "string", "enum": ["High", "Medium", "Low"]}
                                },
                                "required": ["name", "type", "relevance"]
                            }
                        },
                        "engagement_level": {"type": "string", "enum": ["High", "Medium", "Low"]}
                    },
                    "required": ["name", "explanation", "estimated_percentage", "key_terms", "key_entities", "engagement_level"]
                }
            },
            "analysis_period": {"type": "string"},
            "total_posts_analyzed": {"type": "integer"}
        },
        "required": ["topics", "analysis_period", "total_posts_analyzed"]
    }
    
    # Create the system message
    system_message = (
        "You are a data analysis system that identifies trending topics in social media posts. "
        "You must analyze the provided posts and identify the top 5 trending topics, "
        "providing detailed information about each topic including key terms and entities. "
        f"Your response must be valid JSON following this schema: {json.dumps(schema, indent=2)}"
    )
    
    # Create the user message with the actual content
    user_message = f"""
    Analyze the following Farcaster social media posts from the period: {date_range}.
    These are samples from a dataset of {len(recent_df)} posts from this timeframe.
    
    Identify the top 5 trending topics of discussion with supporting evidence.
    
    POSTS:
    {'\n'.join(formatted_casts)}
    """
    
    print(f"Calling Llama 4 via Groq API with {len(formatted_casts)} posts...")
    llm_start_time = time.time()
    
    # Make the API call
    try:
        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
            temperature=0,
            response_format={"type": "json_object"},
            stream=False,
        )
        
        # Extract and parse the JSON response
        llm_topics = json.loads(response.choices[0].message.content)
        print(f"Successfully received structured response with {len(llm_topics.get('topics', []))} topics")
        
    except Exception as e:
        print(f"Error calling Groq API or parsing response: {e}")
        llm_topics = {
            "topics": [],
            "analysis_period": date_range,
            "total_posts_analyzed": len(formatted_casts)
        }
    
    print(f"Llama 4 processing completed in {time.time() - llm_start_time:.2f} seconds")
    
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    # Save raw response for debugging/analysis
    with open('output/llm_response_raw_llama4.json', 'w') as f:
        json.dump(llm_topics, f, indent=2)
    
    # Save in the format expected by downstream processes
    with open('output/approach1_results_llama4.json', 'w') as f:
        json.dump(llm_topics, f, indent=2)
    
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
        with open('output/topic_exemplar_posts_llama4.json', 'w') as f:
            exemplar_data = {
                topic_name: [
                    {
                        'text': post['Text'],
                        'datetime': post['datetime'].strftime('%Y-%m-%d %H:%M:%S') if hasattr(post['datetime'], 'strftime') else str(post['datetime']),
                        'likes': int(post['likes_count']),
                        'recasts': int(post['recasts_count'])
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
        text = str(row['Text']).lower()  # Convert to string to handle potential None values
        # Count keyword matches
        keyword_score = sum(1 for kw in topic_keywords if kw in text)
        # Normalize by text length (avoid favoring very long posts)
        keyword_score = keyword_score / (len(text.split()) + 1) * 10
        # Add engagement bonus
        engagement_bonus = float(row['engagement_score']) / 100
        # Combined score
        combined_score = keyword_score + engagement_bonus
        
        if keyword_score > 0:  # Only consider posts with at least one keyword match
            # Convert row to dict to avoid Series comparison issues
            scores.append((combined_score, row.to_dict()))
    
    # Sort by score and take top n
    scores.sort(key=lambda x: x[0], reverse=True)
    return [post for _, post in scores[:n]]

if __name__ == "__main__":
    # This module is imported and run from the main.py file
    pass