#!/usr/bin/env python3
import json
import sys
import os
import time
sys.path.append('/home/ubuntu/fc-trending-topics/src')
from dotenv import load_dotenv
from google.generativeai import GenerativeModel, configure
from google.generativeai import types

# Mock a cluster from approach2_lda_kmeans.py
test_cluster = {
    'cluster_id': 0,
    'size': 1000,
    'keywords': ['gm', 'bole', 'join', 'friend', 'come'],
    'sample_texts': [
        '[👍10|↗️5]: gm good morning!', 
        '[👍20|↗️10]: gm fam, have a great day'
    ],
    'total_engagement': 100.0,
    'avg_engagement': 5.0,
    'max_engagement': 20.0
}

def test_approach2_labeling():
    """Test the actual cluster labeling function from approach2_lda_kmeans.py with dummy data"""
    # Load environment variables from .env file
    load_dotenv()
    
    # Make sure API key is set
    api_key = os.environ.get('GOOGLE_API_KEY')
    if api_key:
        # Configure the Gemini API
        configure(api_key=api_key)
        print(f"Found API key: {api_key[:5]}...{api_key[-5:]}")
    else:
        print("No API key found in .env file")
        return
    
    print("\n=== TESTING APPROACH 2 CLUSTER LABELING ===")
    
    # Create a model instance (Using gemini-1.5-flash which is faster)
    model = GenerativeModel('gemini-2.0-flash-lite')
    start_time = time.time()
    
    # Create structured prompt (from approach2_lda_kmeans.py)
    prompt = f"""
    I need to identify the single most specific trending topic being discussed in a cluster of Farcaster social media posts.
    
    KEY INFORMATION ABOUT THIS CLUSTER:
    - Top keywords: {', '.join(test_cluster['keywords'])}
    - Cluster size: {test_cluster['size']} posts
    - Average engagement: {test_cluster['avg_engagement']:.2f}
    
    SAMPLE POSTS (with like and recast counts):
    {' '.join(test_cluster['sample_texts'][:10])}
    
    Generate your response based on the following Python TypedDict schema:
    
    class ClusterTopic(TypedDict):
        topic_name: str  # 5 words max
        explanation: str  # Brief explanation of why trending
        estimated_percentage: str  # Percentage of total conversation
        key_terms: list[str]  # List of strings
        engagement_level: str  # High, Medium, Low
        sentiment: str  # Positive, Neutral, Negative, Mixed
    """
    
    print(f"Calling Gemini API...")
    try:
        # Get response with JSON formatting
        response = model.generate_content(
            prompt,
            generation_config=types.GenerationConfig(
                temperature=0,
                response_mime_type="application/json"
            )
        )
        
        print(f"Response time: {time.time() - start_time:.2f} seconds")
        print(f"Raw response: {response.text}")
        
        # Parse JSON response (same as in approach2_lda_kmeans.py)
        try:
            topic_data = json.loads(response.text)
            
            # Check if topic_data is a list (API might have changed format)
            if isinstance(topic_data, list) and len(topic_data) > 0:
                print(f"Detected list response, extracting first item")
                topic_data = topic_data[0]  # Take the first item if it's a list
            
            # Safely get the topic name with a fallback
            topic_name = "Unknown"
            if isinstance(topic_data, dict) and 'topic_name' in topic_data:
                topic_name = topic_data['topic_name']
                
            print(f"Successfully labeled cluster {test_cluster['cluster_id']} as '{topic_name}'")
            print(f"Full topic data: {json.dumps(topic_data, indent=2)}")
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON for cluster {test_cluster['cluster_id']}: {e}")
            topic_data = {
                "topic_name": "Error parsing response",
                "explanation": "Could not parse JSON from Gemini response",
                "estimated_percentage": "unknown",
                "key_terms": test_cluster['keywords'][:5],
                "engagement_level": "unknown",
                "sentiment": "unknown"
            }
            print(f"Using fallback topic data: {json.dumps(topic_data, indent=2)}")
    except Exception as e:
        print(f"Error calling Gemini API: {e}")

if __name__ == "__main__":
    test_approach2_labeling()