"""
Rolling window analysis to identify when the Ghibli trend first appeared.
This script uses the approach3 (conversation metrics) which showed the highest 
engagement for Ghibli topics.
"""

import os
import json
import pandas as pd
from datetime import datetime, timedelta
import duckdb
from tqdm import tqdm
import google.generativeai as genai
from google.generativeai import GenerativeModel, types
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.environ.get('GOOGLE_API_KEY')
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable not set")
genai.configure(api_key=api_key)

def create_rolling_windows(df, start_date, end_date, window_size=48):
    """
    Create rolling windows of data for analysis.
    
    Args:
        df: DataFrame with 'datetime' column
        start_date: Start date for analysis
        end_date: End date for analysis
        window_size: Window size in hours (default: 48)
        
    Returns:
        List of DataFrames, each representing a window
    """
    windows = []
    current_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Convert datetime to pandas datetime if it's not already
    if not isinstance(df['datetime'].iloc[0], pd.Timestamp):
        df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Sort by datetime
    df = df.sort_values('datetime')
    
    while current_date <= end_date:
        window_end = current_date + timedelta(hours=window_size)
        window_df = df[(df['datetime'] >= current_date) & (df['datetime'] < window_end)]
        
        windows.append({
            'start': current_date.strftime('%Y-%m-%d %H:%M'),
            'end': window_end.strftime('%Y-%m-%d %H:%M'),
            'data': window_df
        })
        
        # Move forward by 24 hours
        current_date += timedelta(hours=24)
    
    return windows

def analyze_window(window, window_index, conn, model_name='gemini-2.0-flash-lite'):
    """
    Analyze a single window of data to identify trending topics.
    
    Args:
        window: Dict with 'start', 'end', and 'data' keys
        window_index: Index of the window for tracking
        conn: DuckDB connection
        model_name: Name of Gemini model to use
        
    Returns:
        dict: Analysis results for this window
    """
    print(f"\nAnalyzing window {window_index}: {window['start']} to {window['end']} " +
          f"({len(window['data'])} posts)")
    
    window_df = window['data']
    
    # Check if window_df is empty
    if len(window_df) == 0:
        print("Window contains no data, skipping...")
        return {
            'window': window_index,
            'period': f"{window['start']} to {window['end']}",
            'topics': []
        }
    
    # Register the window DataFrame as a temp table
    conn.register(f'window_{window_index}', window_df)
    
    # Apply conversation metrics
    conn.execute(f"""
    CREATE OR REPLACE VIEW window_{window_index}_weights AS
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
    FROM window_{window_index}
    """)
    
    # Calculate combined weight with emphasis on conversation
    conn.execute(f"""
    CREATE OR REPLACE VIEW window_{window_index}_combined AS
    SELECT
        *,
        -- Combined weight (40% recency, 60% engagement)
        (4 * recency_weight + 6 * engagement_weight) / 10 AS combined_weight
    FROM window_{window_index}_weights
    """)
    
    # Filter to top-level posts
    conn.execute(f"""
    CREATE OR REPLACE VIEW window_{window_index}_filtered AS
    SELECT * 
    FROM window_{window_index}_combined
    WHERE ParentCastId IS NULL OR TRIM(ParentCastId) = ''
    """)
    
    # Sample posts from the window using weighted sampling
    conn.execute(f"""
    CREATE OR REPLACE VIEW window_{window_index}_samples AS
    SELECT
        *,
        -- Generate random weight based on combined_weight
        random() * combined_weight AS sampling_key
    FROM window_{window_index}_filtered
    ORDER BY sampling_key DESC
    """)
    
    # Get the sampled data for this window - limit to 6000 posts maximum per window
    # to avoid hitting API context limits
    sampled_casts = conn.execute(f"""
    SELECT * FROM window_{window_index}_samples
    LIMIT 6000
    """).df()
    
    # Format texts with timestamps, engagement info, and conversation metrics
    formatted_casts = []
    for _, row in sampled_casts.iterrows():
        dt_str = row['datetime'].strftime('%Y-%m-%d %H:%M:%S') if hasattr(row['datetime'], 'strftime') else str(row['datetime'])
        likes = int(row['likes_count']) if pd.notna(row['likes_count']) else 0
        recasts = int(row['recasts_count']) if pd.notna(row['recasts_count']) else 0
        
        # Create base formatted text with engagement metrics
        formatted_text = f"{dt_str} [👍{likes}|↗️{recasts}]"
        
        # Add conversation metrics if available
        reply_count = 0 if pd.isna(row.get('reply_count')) else int(row.get('reply_count', 0))
        unique_repliers = 0 if pd.isna(row.get('unique_repliers')) else int(row.get('unique_repliers', 0))
        
        # Only add conversation metrics if there are replies
        if reply_count > 0:
            # Add conversation metrics
            formatted_text += f" [🗨️{reply_count}|👥{unique_repliers}]"
        
        # Add the actual post text
        formatted_text += f": {row['Text']}"
        formatted_casts.append(formatted_text)
    
    # Initialize Gemini model
    model = GenerativeModel(model_name)
    
    # Create window-specific prompt
    date_range = f"{window['start']} to {window['end']}"
    joined_posts = "\n".join(formatted_casts)
    
    prompt = f"""
    Analyze the following Farcaster social media posts from the period: {date_range}.
    
    For this window, identify the top 10 TRULY TRENDING topics of discussion with supporting evidence.
    You MUST identify 10 distinct topics to ensure comprehensive coverage.
    
    TRENDING topics have these characteristics:
    - HIGH ENGAGEMENT: Topics with many likes and recasts
    - RECENCY: Topics that are active in the most recent timeframe
    - GROWTH: Topics that show increasing activity over time
    - CONVERSATION DEPTH: Topics that generate substantive discussions (many replies, unique repliers)
    
    CRITICAL: You MUST PAY SPECIAL ATTENTION to ANY mentions of "Ghibli", "Ghiblify", "Ghiblification", 
    or any topics related to Studio Ghibli art styles. These are of particular interest.
    If you find ANY posts related to Ghibli, MAKE SURE to include it as a topic even if it's not 
    one of the most trending. This is an important part of our analysis.
    
    Generate your response as a JSON object with this structure:
    {{
        "window": {window_index},  # Window number
        "period": "{date_range}",  # Analysis period
        "topics": [
            {{
                "name": "Topic Name",  # Concise name (5 words max)
                "explanation": "Brief explanation",  # Why it's trending
                "estimated_percentage": "X%",  # Approximate % of posts
                "engagement_level": "High/Medium/Low",  # Based on likes/recasts/replies
                "key_terms": [
                    {{ "term": "term1", "frequency": 10 }},
                    # More terms...
                ],
                "contains_ghibli": true/false,  # Whether this topic mentions Ghibli
                "sample_posts": [  # Up to 3 example posts that illustrate this topic
                    "Example post 1",
                    "Example post 2",
                    "Example post 3"
                ]
            }},
            # More topics...
        ],
        "has_ghibli_content": true/false,  # Whether ANY Ghibli content was found
        "ghibli_first_post_timestamp": "YYYY-MM-DD HH:MM:SS"  # Timestamp of earliest Ghibli post if found
    }}
    
    POSTS:
    {joined_posts}
    """
    
    # Get response with JSON formatting
    response = model.generate_content(
        prompt,
        generation_config=types.GenerationConfig(
            temperature=0.1,  # Very slight variation for better topic diversity
            response_mime_type="application/json"
        )
    )
    
    # Parse and process the response
    try:
        result = json.loads(response.text)
        print(f"Successfully identified {len(result.get('topics', []))} topics")
        print(f"Has Ghibli content: {result.get('has_ghibli_content', False)}")
        if result.get('has_ghibli_content', False):
            print(f"First Ghibli post: {result.get('ghibli_first_post_timestamp', 'Not specified')}")
        
        # Look for Ghibli topics
        ghibli_topics = [t for t in result.get('topics', []) if t.get('contains_ghibli', False)]
        if ghibli_topics:
            print(f"Found {len(ghibli_topics)} Ghibli-related topics:")
            for topic in ghibli_topics:
                print(f"  - {topic.get('name')}: {topic.get('estimated_percentage')}")
                
        return result
    except Exception as e:
        print(f"Error parsing response: {e}")
        # Return a minimal result object
        return {
            'window': window_index,
            'period': date_range,
            'topics': [],
            'has_ghibli_content': False,
            'error': str(e)
        }

def main():
    # Setup DuckDB connection
    conn = duckdb.connect(database=':memory:')
    conn.execute("SET memory_limit='4GB'")
    
    # Load data
    parquet_path = 'output/interim_data/cleaned_data.parquet'
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Data file not found: {parquet_path}")
    
    print(f"Loading data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} posts")
    
    # Create output directory
    os.makedirs('output/ghibli_analysis', exist_ok=True)
    
    # Create rolling windows from March 21 to March 31, 2025
    print("Creating 48-hour rolling windows...")
    windows = create_rolling_windows(
        df, 
        start_date='2025-03-21', 
        end_date='2025-03-31',
        window_size=48  # 48-hour windows
    )
    print(f"Created {len(windows)} windows")
    
    # Analyze each window
    results = []
    for i, window in enumerate(windows):
        result = analyze_window(window, i, conn)
        results.append(result)
        
        # Save individual window result
        with open(f'output/ghibli_analysis/window_{i}_results.json', 'w') as f:
            json.dump(result, f, indent=2)
    
    # Save all results
    print("\nSaving all results...")
    with open('output/ghibli_analysis/all_windows_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Summarize Ghibli findings
    print("\nGhibli trend findings summary:")
    earliest_ghibli = None
    earliest_window = None
    
    for i, result in enumerate(results):
        if result.get('has_ghibli_content', False):
            timestamp = result.get('ghibli_first_post_timestamp')
            if timestamp and (earliest_ghibli is None or timestamp < earliest_ghibli):
                earliest_ghibli = timestamp
                earliest_window = i
    
    if earliest_ghibli:
        print(f"Earliest Ghibli content found in window {earliest_window}")
        print(f"Timestamp: {earliest_ghibli}")
        print(f"Window period: {results[earliest_window]['period']}")
    else:
        print("No Ghibli content found in any window")
    
    # Create a summary report
    summary_report = {
        'total_windows': len(windows),
        'windows_with_ghibli': sum(1 for r in results if r.get('has_ghibli_content', False)),
        'earliest_ghibli_window': earliest_window,
        'earliest_ghibli_timestamp': earliest_ghibli,
        'window_details': [
            {
                'window': i,
                'period': r['period'],
                'has_ghibli': r.get('has_ghibli_content', False),
                'ghibli_topics': [t['name'] for t in r.get('topics', []) if t.get('contains_ghibli', False)]
            }
            for i, r in enumerate(results)
        ]
    }
    
    # Save summary report
    with open('output/ghibli_analysis/ghibli_trend_summary.json', 'w') as f:
        json.dump(summary_report, f, indent=2)
    
    print("\nAnalysis complete. Results saved to output/ghibli_analysis/")

if __name__ == "__main__":
    main()