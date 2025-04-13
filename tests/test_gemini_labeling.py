#!/usr/bin/env python3
import json
import sys
import os
sys.path.append('/home/ubuntu/fc-trending-topics/src')
from google.generativeai import GenerativeModel, configure
from google.generativeai import types
from dotenv import load_dotenv

# Test data simulating a cluster
test_cluster = {
    'cluster_id': 0,
    'size': 1000,
    'keywords': ['test', 'keyword', 'example', 'cluster', 'dummy'],
    'sample_texts': [
        '[👍10|↗️5]: This is a test post', 
        '[👍20|↗️10]: Another example of a post in this cluster'
    ],
    'total_engagement': 100.0,
    'avg_engagement': 5.0,
    'max_engagement': 20.0
}

def test_gemini_labeling():
    """Test the Gemini labeling functionality with both dict and list responses"""
    
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
    
    # For testing purposes, we'll run both mock tests and real API tests if available
    
    # First run mock tests
    print("\n=== RUNNING MOCK TESTS ===")
    mock_test()
    
    # Run real API test if key is available
    if api_key:
        print("\n=== RUNNING REAL API TEST ===")
        api_test()
    else:
        print("No API key available, skipping real API test")

def mock_test():
    """Run tests with mock data"""
    # Mock responses for testing
    mock_responses = [
        # Dictionary response (original expected format)
        {
            "topic_name": "Test Topic", 
            "explanation": "This is a test", 
            "estimated_percentage": "10%", 
            "key_terms": ["test", "example"], 
            "engagement_level": "Medium", 
            "sentiment": "Neutral"
        },
        # List response (needs to be handled)
        [
            {
                "topic_name": "List Test Topic", 
                "explanation": "This is a test in list format", 
                "estimated_percentage": "15%", 
                "key_terms": ["list", "test"], 
                "engagement_level": "High", 
                "sentiment": "Positive"
            }
        ],
        # Empty response (edge case)
        {},
        # Null response (edge case)
        None
    ]
    
    # Test all mock responses
    for i, mock_response in enumerate(mock_responses):
        print(f"\n--- Testing mock response {i+1} ---")
        
        # Convert to JSON string
        response_text = json.dumps(mock_response) if mock_response is not None else "null"
        
        # This would be the response object in the real code
        class MockResponse:
            def __init__(self, text):
                self.text = text
        
        response = MockResponse(response_text)
        
        # Parse JSON response (simulate the code in approach2_lda_kmeans.py)
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
        
        print(f"Final topic_data: {topic_data}")

def api_test():
    """Run tests with real Gemini API"""
    # Create a model instance
    model = GenerativeModel('gemini-1.5-flash')
    
    # Create structured prompt (simplified from approach2_lda_kmeans.py)
    prompt = f"""
    I need to identify the single most specific trending topic being discussed in a cluster of social media posts.
    
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
    
    print("\n--- Testing with JSON response format ---")
    try:
        # Get response with JSON formatting
        response = model.generate_content(
            prompt,
            generation_config=types.GenerationConfig(
                temperature=0,
                response_mime_type="application/json"
            )
        )
        
        print(f"Raw response: {response.text}")
        
        # Parse JSON response
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
        
        print(f"Final processed topic_data: {topic_data}")
    except Exception as e:
        print(f"Error calling Gemini API: {e}")

if __name__ == "__main__":
    test_gemini_labeling()