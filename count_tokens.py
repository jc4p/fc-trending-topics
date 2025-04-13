#!/usr/bin/env python3
import json
import os
import sys
from transformers import AutoTokenizer
from dotenv import load_dotenv

def load_file(filename):
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read()
    return None

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    # Get Hugging Face token from environment
    hf_token = os.environ.get('HUGGING_FACE_HUB_TOKEN')
    if hf_token:
        print("Found Hugging Face token in environment")
    else:
        print("Warning: HUGGING_FACE_HUB_TOKEN not found in environment")
    
    # Load Gemma tokenizer which should be similar to Gemini
    try:
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b", token=hf_token)
        print("Loaded Gemma tokenizer")
    except Exception as e:
        print(f"Error loading Gemma tokenizer: {e}")
        print("Falling back to approximate word-based estimation")
        use_tokenizer = False
    else:
        use_tokenizer = True
    
    # Find the critic response file
    critic_file = "output/debug/critic_raw_response.txt"
    critic_content = load_file(critic_file)
    
    if critic_content:
        if use_tokenizer:
            critic_tokens = len(tokenizer.encode(critic_content))
            print(f"Critic response (accurate token count): {critic_tokens}")
        else:
            # Fallback to word-based approximation
            word_count = len(critic_content.split())
            token_estimate = int(word_count / 0.75)
            print(f"Critic response (approximate tokens): {token_estimate}")
    
    # Find the checkpoint file
    checkpoint_files = [f for f in os.listdir("output/checkpoints") if f.endswith("generator_checkpoint.json")]
    if checkpoint_files:
        checkpoint_file = os.path.join("output/checkpoints", sorted(checkpoint_files)[-1])
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
        
        # Get topics JSON
        topics = checkpoint_data.get('batch_topics', [])
        topics_json = json.dumps(topics, indent=2)
        
        # Create the complete critic prompt
        base_prompt = f"""
        As an expert trend evaluator for Farcaster social network, critically evaluate these {len(topics)} candidate trending topics 
        from the period 2023-01-01 to 2023-01-31.
        
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
        
        if use_tokenizer:
            topics_tokens = len(tokenizer.encode(topics_json))
            prompt_tokens = len(tokenizer.encode(base_prompt))
            print(f"Base prompt (accurate token count): {prompt_tokens - topics_tokens}")
            print(f"Topics JSON (accurate token count): {topics_tokens}")
            print(f"Total input tokens to critic: {prompt_tokens}")
        else:
            # Fallback estimates
            topics_words = len(topics_json.split())
            topics_tokens = int(topics_words / 0.75)
            base_prompt_tokens = 1500  # Estimate based on the prompt template
            print(f"Topics JSON (approximate tokens): {topics_tokens}")
            print(f"Number of topics: {len(topics)}")
            print(f"Estimated total input tokens to critic: {base_prompt_tokens + topics_tokens}")
        
        # Calculate approximate cost based on Gemini pricing
        token_count = prompt_tokens if use_tokenizer else (base_prompt_tokens + topics_tokens)
        input_cost_per_million = 2.50  # $2.50 per million tokens for input > 200K
        output_cost_per_million = 15.00  # $15.00 per million tokens for output > 200K
        
        input_cost = (token_count / 1_000_000) * input_cost_per_million
        if critic_content:
            output_tokens = len(tokenizer.encode(critic_content)) if use_tokenizer else token_estimate
            output_cost = (output_tokens / 1_000_000) * output_cost_per_million
            total_cost = input_cost + output_cost
            print(f"\nEstimated cost for this critic API call:")
            print(f"Input cost: ${input_cost:.6f}")
            print(f"Output cost: ${output_cost:.6f}")
            print(f"Total cost: ${total_cost:.6f}")
            
            # Extrapolate to 15 chunks
            print(f"\nExtrapolated cost for 15 chunks: ${total_cost * 15:.4f}")

if __name__ == "__main__":
    main()
