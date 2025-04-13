## Approach 1: Direct LLM Analysis

1. **Optimized Sample Selection with DuckDB**
   ```python
   import random
   import time
   
   print("Selecting optimal sample for LLM analysis...")
   start_time = time.time()
   
   # Sample size considering Gemini 2.0 Flash Lite's 1M context window
   # Estimate ~150 chars per cast on average, aim for ~900K chars
   sample_size = min(6000, len(recent_df))
   
   # Use DuckDB for weight calculation and sampling
   # This is much more efficient than pandas for large datasets
   
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
   # We'll implement a weighted reservoir algorithm in DuckDB
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
   ```

2. **Gemini API Integration with Structured Response**
   ```python
   from enum import Enum
   from typing_extensions import TypedDict
   from google.generativeai import GenerativeModel
   from google.generativeai import types
   import os
   import json
   
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
   
   # Initialize Gemini with structured output
   model = GenerativeModel('gemini-2.0-flash-lite')
   
   # Create a more detailed prompt with time context and structured output
   date_range = f"{recent_df['datetime'].min().strftime('%Y-%m-%d')} to {recent_df['datetime'].max().strftime('%Y-%m-%d')}"
   
   # Create structured prompt
   prompt = f"""
   Analyze the following Farcaster social media posts from the period: {date_range}.
   These are samples from a dataset of {len(recent_df)} posts from this timeframe.
   
   Identify the top 5 trending topics of discussion with supporting evidence.
   
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
   
   POSTS:
   {'\n'.join(formatted_casts)}
   """
   
   # Get response with JSON formatting
   response = model.generate_content(
       prompt,
       config=types.GenerateContentConfig(
           temperature=0,
           response_mime_type="application/json"
       )
   )
   
   # Parse and process the response
   try:
       llm_topics = json.loads(response.text)
       print(f"Successfully received structured response with {len(llm_topics.get('topics', []))} topics")
   except json.JSONDecodeError as e:
       print(f"Error parsing JSON response: {e}")
       llm_topics = {
           "topics": [],
           "analysis_period": date_range,
           "total_posts_analyzed": len(formatted_casts)
       }
   
   # Save raw response for debugging/analysis
   with open('llm_response_raw.json', 'w') as f:
       json.dump(llm_topics, f, indent=2)
   
   # Save in the format expected by downstream processes
   with open('approach1_results.json', 'w') as f:
       json.dump(llm_topics, f, indent=2)
   ```

3. **Process Results and Enrichment**
   ```python
   # No additional parsing needed since we're using structured output
   # But we can add some analysis and enrichment of the results
   
   # Log topic information
   if 'topics' in llm_topics:
       for i, topic in enumerate(llm_topics['topics']):
           print(f"Topic {i+1}: {topic['name']}")
           print(f"  Estimated percentage: {topic['estimated_percentage']}")
           print(f"  Engagement level: {topic['engagement_level']}")
           print(f"  Key terms: {', '.join([term['term'] for term in topic['key_terms'][:5]])}")
           print(f"  Key entities: {', '.join([entity['name'] for entity in topic['key_entities'][:3]])}")
           print()
   
   # Optional enrichment: Find exemplar posts for each topic
   # This finds representative posts for each detected topic
   def find_exemplar_posts_for_topic(topic, posts_df, n=3):
       """Find exemplar posts that best represent a given topic."""
       # Create a simple keyword matching function
       topic_keywords = [term['term'].lower() for term in topic['key_terms']]
       
       # Score each post based on keyword matches and engagement
       scores = []
       for _, row in posts_df.iterrows():
           text = row['Text'].lower()
           # Count keyword matches
           keyword_score = sum(1 for kw in topic_keywords if kw in text)
           # Normalize by text length (avoid favoring very long posts)
           keyword_score = keyword_score / (len(text.split()) + 1) * 10
           # Add engagement bonus
           engagement_bonus = row['engagement_score'] / 100
           # Combined score
           combined_score = keyword_score + engagement_bonus
           
           if keyword_score > 0:  # Only consider posts with at least one keyword match
               scores.append((combined_score, row))
       
       # Sort by score and take top n
       scores.sort(reverse=True)
       return [post for _, post in scores[:n]]
   
   # Find exemplar posts for each topic
   if 'topics' in llm_topics:
       topic_exemplars = {}
       for i, topic in enumerate(llm_topics['topics']):
           exemplars = find_exemplar_posts_for_topic(topic, sampled_casts)
           topic_exemplars[topic['name']] = exemplars
   
       # Save exemplars for later use
       with open('topic_exemplar_posts.json', 'w') as f:
           exemplar_data = {
               topic_name: [
                   {
                       'text': post['Text'],
                       'datetime': post['datetime'].strftime('%Y-%m-%d %H:%M:%S'),
                       'likes': int(post['likes_count']),
                       'recasts': int(post['recasts_count'])
                   } for post in posts
               ] for topic_name, posts in topic_exemplars.items()
           }
           json.dump(exemplar_data, f, indent=2)
   ```