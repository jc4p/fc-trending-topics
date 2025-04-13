import os
import json
import glob
from datetime import datetime
from google.generativeai import GenerativeModel, types

def load_all_results():
    """
    Load all batch results from the output directory.
    
    Returns:
        dict: Mapping of batch_id to results
    """
    print("Loading all batch results...")
    results = {}
    
    # Find all batch result files (format: batch_X_approach4_results.json)
    result_files = glob.glob('output/batch_*_approach4_results.json')
    
    if not result_files:
        print("No batch result files found. Please run the batch processing first.")
        return None
    
    # Load each batch result
    for file_path in result_files:
        basename = os.path.basename(file_path)
        batch_id = basename.split('_')[1]
        
        try:
            with open(file_path, 'r') as f:
                batch_result = json.load(f)
                results[batch_id] = batch_result
                
                # Report topics count for this batch
                if 'topics' in batch_result:
                    print(f"Batch {batch_id}: Found {len(batch_result['topics'])} topics")
                else:
                    print(f"Batch {batch_id}: No topics found")
        except Exception as e:
            print(f"Error loading batch {batch_id}: {e}")
    
    print(f"Successfully loaded {len(results)} batch results")
    return results

def combine_topics(batch_results):
    """
    Combine all topics from different batches into a single list.
    
    Args:
        batch_results: Dictionary of batch results
        
    Returns:
        list: All topics from all batches
        str: Date range (merged from all batches)
        int: Total posts analyzed
    """
    # Find the combined date range
    min_date = None
    max_date = None
    total_posts = 0
    
    for batch_id, result in batch_results.items():
        if 'analysis_period' in result:
            # Parse the date range (format: "YYYY-MM-DD to YYYY-MM-DD")
            date_parts = result['analysis_period'].split(' to ')
            if len(date_parts) == 2:
                batch_min = date_parts[0]
                batch_max = date_parts[1]
                
                if min_date is None or batch_min < min_date:
                    min_date = batch_min
                    
                if max_date is None or batch_max > max_date:
                    max_date = batch_max
        
        if 'total_posts_analyzed' in result:
            total_posts += result['total_posts_analyzed']
    
    # Combine the date range
    if min_date and max_date:
        date_range = f"{min_date} to {max_date}"
    else:
        date_range = "Unknown date range"
    
    # Collect all topics
    all_topics = []
    for batch_id, result in batch_results.items():
        if 'topics' in result:
            all_topics.extend(result['topics'])
    
    print(f"Combined {len(all_topics)} topics from all batches")
    print(f"Date range: {date_range}")
    print(f"Total posts analyzed: {total_posts}")
    
    return all_topics, date_range, total_posts

def consolidate_with_critic(all_topics, date_range, total_posts):
    """
    Consolidate all topics from all batches using the Gemini 2.5 critic model.
    
    Args:
        all_topics: List of all topics
        date_range: Date range for the analysis
        total_posts: Total number of posts analyzed
        
    Returns:
        dict: Consolidated trending topics result
    """
    print(f"\nConsolidating {len(all_topics)} topics with Gemini 2.5 Pro Preview...")
    
    # Load API key
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("Warning: python-dotenv not installed. Install with 'pip install python-dotenv'")
    
    api_key = os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    
    # Initialize Gemini for consolidation
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    consolidation_model = GenerativeModel('gemini-2.5-pro-preview-03-25')
    
    # Create consolidated topics JSON
    topics_json = json.dumps(all_topics, indent=2)
    
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
    
    IMPORTANT: Prioritize PLATFORM FEATURES and WIDELY RELEVANT BEHAVIORS over mini-apps, 
    specific third-party tools, or temporary memes. Look for topics that represent significant
    platform-wide activity that would interest a broad audience.
    
    TOPIC CANDIDATES:
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
    
    return consolidated_topics

def save_consolidated_results(consolidated_topics):
    """
    Save the consolidated results to a file.
    
    Args:
        consolidated_topics: Consolidated topics result
    """
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    # Add timestamp to filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save the consolidated results
    with open(f'output/combined_approach4_results_{timestamp}.json', 'w') as f:
        json.dump(consolidated_topics, f, indent=2)
    
    # Save a copy with a standard name for easier access
    with open('output/combined_approach4_results.json', 'w') as f:
        json.dump(consolidated_topics, f, indent=2)
    
    print(f"Saved consolidated results to output/combined_approach4_results_{timestamp}.json")
    print("Also saved a copy to output/combined_approach4_results.json")

def main():
    # Load all batch results
    batch_results = load_all_results()
    
    if batch_results:
        # Combine all topics
        all_topics, date_range, total_posts = combine_topics(batch_results)
        
        if all_topics:
            # Consolidate with the critic model
            consolidated_topics = consolidate_with_critic(all_topics, date_range, total_posts)
            
            # Save the consolidated results
            save_consolidated_results(consolidated_topics)
            
            # Print a summary of the final results
            if 'topics' in consolidated_topics:
                num_topics = len(consolidated_topics['topics'])
                
                print(f"\n===== COMBINED ANALYSIS RESULTS =====\n")
                print(f"Identified {num_topics} trending topics for the period {consolidated_topics['analysis_period']}:")
                print()
                
                for i, topic in enumerate(consolidated_topics['topics']):
                    print(f"Topic {i+1}: {topic['name']}")
                    print(f"  Explanation: {topic['explanation']}")
                    print(f"  Estimated percentage: {topic['estimated_percentage']}")
                    print(f"  Engagement level: {topic['engagement_level']}")
                    
                    if 'key_terms' in topic and topic['key_terms']:
                        terms = [t['term'] for t in topic['key_terms'][:3]]
                        print(f"  Key terms: {', '.join(terms)}")
                    
                    if 'key_entities' in topic and topic['key_entities']:
                        entities = [e['name'] for e in topic['key_entities'][:3]]
                        print(f"  Key entities: {', '.join(entities)}")
                    
                    print()
                
                print(f"Total topics identified: {num_topics}")
                print(f"Analysis period: {consolidated_topics['analysis_period']}")
                print(f"Total posts analyzed: {consolidated_topics['total_posts_analyzed']}")
            else:
                print("No topics found in the final results.")
        else:
            print("No topics found in any batch. Cannot consolidate results.")
    else:
        print("No batch results found. Please run the batch processing first.")

if __name__ == "__main__":
    main()