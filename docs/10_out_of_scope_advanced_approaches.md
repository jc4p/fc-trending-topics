## Out of Scope: Advanced Approaches

The following outlines potential future implementations for significantly improving trending topic detection. These approaches are currently out of scope due to their complexity and resource requirements.

### 1. Full Reinforcement Learning Approach

1. **Problem Formulation**
   ```
   The trending topic detection problem can be formulated as a reinforcement learning task:
   
   - State: A representation of potential topics and their features
   - Actions: Decisions about which topics to classify as "trending" and their ranking
   - Rewards: Feedback based on topic specificity, novelty, and user engagement
   ```

2. **Model Architecture**
   ```python
   import torch
   import torch.nn as nn
   import torch.nn.functional as F
   from transformers import AutoModel
   
   class TrendingTopicPolicy(nn.Module):
       """RL policy network for trending topic detection"""
       
       def __init__(self, base_model_name="EleutherAI/pythia-1b", embedding_dim=768, hidden_dim=512):
           super().__init__()
           
           # Option 1: Use a small LLM as base encoder
           self.base_model = AutoModel.from_pretrained(base_model_name)
           
           # Option 2: Use a simpler vector-based approach
           # self.base_model = SentenceTransformer('all-MiniLM-L6-v2')
           
           # Policy network layers
           self.topic_encoder = nn.Sequential(
               nn.Linear(embedding_dim, hidden_dim),
               nn.ReLU(),
               nn.Linear(hidden_dim, hidden_dim),
               nn.ReLU()
           )
           
           # Value prediction (how "trending" is a topic)
           self.value_head = nn.Linear(hidden_dim, 1)
           
           # Topic classification (trending vs non-trending)
           self.classification_head = nn.Linear(hidden_dim, 2)
           
           # Topic ranking (relative order of trending topics)
           self.ranking_head = nn.Linear(hidden_dim, 1)
       
       def forward(self, topic_data):
           """
           Forward pass through the model
           
           Args:
               topic_data: Dictionary containing topic info:
                   - name: Topic name
                   - keywords: Associated keywords
                   - text_samples: Sample posts for this topic
                   - features: Extracted features (engagement, etc.)
           """
           # Get embeddings for the topic
           topic_name_emb = self.base_model(topic_data['name']).pooler_output
           keywords_emb = self.base_model(', '.join(topic_data['keywords'])).pooler_output
           
           # For text samples, get mean embedding
           sample_embs = []
           for sample in topic_data['text_samples'][:5]:  # Limit to 5 samples
               sample_embs.append(self.base_model(sample).pooler_output)
           samples_emb = torch.mean(torch.stack(sample_embs), dim=0)
           
           # Combine embeddings
           combined_emb = torch.cat([
               topic_name_emb, 
               keywords_emb, 
               samples_emb, 
               torch.tensor(topic_data['features'])
           ], dim=-1)
           
           # Process through policy network
           topic_encoding = self.topic_encoder(combined_emb)
           
           # Get outputs
           value = self.value_head(topic_encoding)  # How trending
           classification = self.classification_head(topic_encoding)  # Binary trending/not
           ranking = self.ranking_head(topic_encoding)  # Relative rank
           
           return {
               'value': value,
               'classification': classification,
               'ranking': ranking,
               'encoding': topic_encoding
           }
   ```

3. **Training Data Generation**
   ```python
   import pandas as pd
   from google.generativeai import GenerativeModel, types
   
   def generate_training_data(num_examples=1000):
       """Generate synthetic training data for RL model"""
       
       # Initialize Gemini model for generating examples
       model = GenerativeModel('gemini-2.0-flash-lite')
       
       # 1. Generate positive examples (specific, novel trending topics)
       positive_prompt = """
       Generate {n} examples of specific, novel trending topics that might appear on a crypto/web3 social network. 
       
       These should be:
       - Specific (e.g., "Llama 3 Model Release" not just "AI")
       - Novel/interesting (not everyday topics like "Bitcoin Price")
       - Time-bound (related to events or time-limited discussions)
       
       For each topic, include:
       1. Topic name (specific and concise)
       2. Topic explanation
       3. Five keywords associated with this topic
       4. Three example posts discussing this topic
       5. Engagement metrics (estimated likes and recasts)
       
       Format as a JSON array.
       """
       
       positive_response = model.generate_content(
           positive_prompt.format(n=num_examples // 2),
           config=types.GenerateContentConfig(
               temperature=0.8,
               response_mime_type="application/json"
           )
       )
       
       # 2. Generate negative examples (generic, common topics)
       negative_prompt = """
       Generate {n} examples of common, generic topics that are always discussed on crypto/web3 social networks
       and should NOT be considered "trending".
       
       These should be:
       - General categories (e.g., "Cryptocurrency" or "NFT Art")
       - Everyday discussion topics with no temporal specificity
       - Common technical concepts or platform features
       
       For each topic, include:
       1. Topic name (generic and broad)
       2. Topic explanation
       3. Five keywords associated with this topic
       4. Three example posts discussing this topic
       5. Engagement metrics (estimated likes and recasts)
       
       Format as a JSON array.
       """
       
       negative_response = model.generate_content(
           negative_prompt.format(n=num_examples // 2),
           config=types.GenerateContentConfig(
               temperature=0.7,
               response_mime_type="application/json"
           )
       )
       
       # Combine and process examples
       positive_examples = json.loads(positive_response.text)
       for ex in positive_examples:
           ex['is_trending'] = 1
           ex['novelty_score'] = 0.8 + (0.2 * random.random())  # High novelty
           
       negative_examples = json.loads(negative_response.text)
       for ex in negative_examples:
           ex['is_trending'] = 0
           ex['novelty_score'] = 0.1 + (0.3 * random.random())  # Low novelty
           
       # Combine datasets
       training_data = positive_examples + negative_examples
       
       # Convert to DataFrame
       df = pd.DataFrame(training_data)
       
       # Add additional features
       df['specificity_score'] = df['topic_name'].apply(lambda x: min(len(x.split()), 5) / 5.0)
       df['keyword_specificity'] = df['keywords'].apply(lambda x: sum(len(k.split()) for k in x) / len(x) / 3)
       
       return df
   ```

4. **RL Training Environment**
   ```python
   import gym
   from gym import spaces
   import numpy as np
   
   class TrendingTopicEnv(gym.Env):
       """Reinforcement learning environment for trending topic detection"""
       
       def __init__(self, training_data, batch_size=16):
           super().__init__()
           
           self.training_data = training_data
           self.batch_size = batch_size
           self.current_batch = None
           self.current_step = 0
           self.max_steps = len(training_data) // batch_size
           
           # Define action space: binary classification for each topic in batch
           self.action_space = spaces.Box(
               low=0, high=1, 
               shape=(batch_size, 2),  # [p(not trending), p(trending)]
               dtype=np.float32
           )
           
           # Define observation space (topic features)
           feature_dim = 10  # Example feature dimension
           self.observation_space = spaces.Box(
               low=-float('inf'), high=float('inf'),
               shape=(batch_size, feature_dim),
               dtype=np.float32
           )
           
       def reset(self):
           """Reset environment and get new batch"""
           self.current_step = 0
           batch_indices = np.random.choice(
               len(self.training_data), 
               size=self.batch_size, 
               replace=False
           )
           self.current_batch = self.training_data.iloc[batch_indices].reset_index(drop=True)
           
           # Create observations
           observations = self._get_observations()
           return observations
       
       def _get_observations(self):
           """Convert current batch to observations"""
           features = []
           for _, row in self.current_batch.iterrows():
               # Extract features for each topic
               feature_vector = [
                   row['specificity_score'],
                   row['novelty_score'],
                   row['keyword_specificity'],
                   len(row['topic_name'].split()),
                   # Add other numerical features...
               ]
               features.append(feature_vector)
           return np.array(features, dtype=np.float32)
       
       def step(self, actions):
           """Take a step in the environment"""
           # Calculate rewards based on actions
           rewards = []
           
           for i, action in enumerate(actions):
               topic_row = self.current_batch.iloc[i]
               is_trending = topic_row['is_trending']
               
               # Get probabilities from action
               p_trending = action[1]  # Probability of being trending
               
               # Calculate reward components
               # 1. Classification reward: correct = +1, incorrect = -1
               if (p_trending >= 0.5 and is_trending == 1) or (p_trending < 0.5 and is_trending == 0):
                   classification_reward = 1.0
               else:
                   classification_reward = -1.0
                   
               # 2. Confidence reward: bonus for high confidence when correct
               confidence = abs(p_trending - 0.5) * 2  # 0 to 1
               if (p_trending >= 0.5 and is_trending == 1) or (p_trending < 0.5 and is_trending == 0):
                   confidence_reward = confidence
               else:
                   confidence_reward = -confidence
                   
               # 3. Novelty reward: bonus for detecting novel topics
               if p_trending >= 0.5:
                   novelty_reward = topic_row['novelty_score'] - 0.5
               else:
                   novelty_reward = 0
                   
               # 4. Specificity reward: bonus for specific topics
               if p_trending >= 0.5:
                   specificity_reward = topic_row['specificity_score'] - 0.5
               else:
                   specificity_reward = 0
               
               # Combine rewards with weights
               total_reward = (
                   2.0 * classification_reward +
                   1.0 * confidence_reward +
                   1.5 * novelty_reward +
                   1.5 * specificity_reward
               )
               
               rewards.append(total_reward)
           
           # Increment step counter
           self.current_step += 1
           done = (self.current_step >= self.max_steps)
           
           # Get next observations
           if not done:
               batch_indices = np.random.choice(
                   len(self.training_data), 
                   size=self.batch_size, 
                   replace=False
               )
               self.current_batch = self.training_data.iloc[batch_indices].reset_index(drop=True)
               next_observations = self._get_observations()
           else:
               next_observations = self._get_observations()  # Return current obs if done
               
           return next_observations, np.array(rewards), done, {}
   ```

5. **RL Training Loop**
   ```python
   import torch.optim as optim
   from stable_baselines3 import PPO
   
   # Create policy model
   policy_model = TrendingTopicPolicy()
   
   # Generate training data
   training_data = generate_training_data(num_examples=10000)
   
   # Create environment
   env = TrendingTopicEnv(training_data)
   
   # Initialize RL algorithm (PPO)
   model = PPO(
       policy="MlpPolicy",
       env=env,
       learning_rate=3e-4,
       n_steps=2048,
       batch_size=64,
       n_epochs=10,
       gamma=0.99,
       verbose=1
   )
   
   # Train model
   model.learn(total_timesteps=1000000)
   
   # Save trained model
   model.save("trending_topic_rl_model")
   ```

6. **Inference and Integration**
   ```python
   def filter_topics_with_rl(candidate_topics):
       """Filter and rank candidate topics using the RL model
       
       Args:
           candidate_topics: List of topic dictionaries from previous approaches
           
       Returns:
           list: Filtered and re-ranked topics
       """
       # Load trained model
       model = PPO.load("trending_topic_rl_model")
       
       # Prepare topics for inference
       topic_features = []
       for topic in candidate_topics:
           # Extract features
           topic_name = topic.get('name', topic.get('topic_data', {}).get('topic_name', ''))
           
           if 'key_terms' in topic:
               keywords = topic['key_terms']
           elif 'topic_data' in topic and 'key_terms' in topic['topic_data']:
               keywords = topic['topic_data']['key_terms']
           else:
               keywords = []
               
           # Get text samples
           text_samples = []
           # (logic to extract text samples from topic data)
           
           # Calculate manual features
           specificity_score = min(len(topic_name.split()), 5) / 5.0
           keyword_specificity = sum(len(k.split()) for k in keywords) / max(1, len(keywords)) / 3
           
           # Create feature vector
           features = [
               specificity_score,
               keyword_specificity,
               # Add other numerical features
           ]
           
           topic_features.append({
               'name': topic_name,
               'keywords': keywords,
               'text_samples': text_samples,
               'features': features
           })
       
       # Get model predictions
       predictions = []
       for features in topic_features:
           observation = np.array(features['features'], dtype=np.float32)
           action, _ = model.predict(observation)
           
           # Extract trending probability and ranking score
           trend_prob = action[1]  # Probability of being trending
           
           predictions.append({
               'topic': features['name'],
               'trending_probability': float(trend_prob),
               'is_trending': trend_prob >= 0.5
           })
       
       # Filter to only trending topics
       trending_topics = [p for p in predictions if p['is_trending']]
       
       # Sort by trending probability
       trending_topics.sort(key=lambda x: x['trending_probability'], reverse=True)
       
       # Return top trending topics with original data
       result = []
       for pred in trending_topics:
           # Find original topic data
           for topic in candidate_topics:
               topic_name = topic.get('name', topic.get('topic_data', {}).get('topic_name', ''))
               if topic_name == pred['topic']:
                   # Add prediction data to original topic
                   topic_copy = topic.copy()
                   topic_copy['trending_probability'] = pred['trending_probability']
                   result.append(topic_copy)
                   break
                   
       return result
   ```

7. **Evaluation and Feedback Loop**
   ```python
   def evaluate_trending_topics(predicted_topics, ground_truth=None):
       """Evaluate trending topic detection performance
       
       If ground truth is available, calculate precision/recall.
       Otherwise, use human feedback to create training signal.
       """
       if ground_truth is not None:
           # Calculate precision, recall, F1 score
           true_positives = 0
           false_positives = 0
           false_negatives = 0
           
           for pred in predicted_topics:
               topic_name = pred.get('name', pred.get('topic_data', {}).get('topic_name', ''))
               
               # Check if topic is in ground truth
               if topic_name in ground_truth:
                   true_positives += 1
               else:
                   false_positives += 1
                   
           for gt_topic in ground_truth:
               # Check if ground truth topic was missed
               found = False
               for pred in predicted_topics:
                   topic_name = pred.get('name', pred.get('topic_data', {}).get('topic_name', ''))
                   if topic_name == gt_topic:
                       found = True
                       break
                       
               if not found:
                   false_negatives += 1
                   
           # Calculate metrics
           precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
           recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
           f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
           
           return {
               'precision': precision,
               'recall': recall,
               'f1': f1,
               'true_positives': true_positives,
               'false_positives': false_positives,
               'false_negatives': false_negatives
           }
       else:
           # Here we'd implement human feedback collection
           # For example, a mechanism to rate topics as good/bad trending topics
           # This feedback would be used to generate additional training data
           pass
   ```

8. **Continuous Improvement Mechanism**
   ```python
   def update_model_with_feedback(model, feedback_data):
       """Use collected feedback to update the model
       
       Args:
           model: Trained RL model
           feedback_data: Dictionary mapping topic names to feedback scores
       """
       # Convert feedback to training examples
       new_training_data = []
       
       for topic_name, feedback in feedback_data.items():
           # Find topic in original data
           # This is simplified; in practice would need to store more data
           topic_data = {
               'topic_name': topic_name,
               'is_trending': 1 if feedback >= 0.5 else 0,
               'novelty_score': feedback,
               # Other features would be stored or regenerated
           }
           
           new_training_data.append(topic_data)
           
       # Convert to DataFrame
       new_df = pd.DataFrame(new_training_data)
       
       # Add to existing training data
       # In practice, would maintain a growing dataset with decay
       updated_training_data = pd.concat([model.env.training_data, new_df])
       
       # Create new environment with updated data
       new_env = TrendingTopicEnv(updated_training_data)
       
       # Update model
       model.set_env(new_env)
       model.learn(total_timesteps=100000)  # Shorter update
       
       return model
   ```

This approach would provide much more robust and adaptive trending topic detection by:

1. Learning from examples what constitutes a good trending topic
2. Automatically filtering out common baseline topics 
3. Prioritizing specific and novel discussions
4. Continuously improving through feedback loops

The primary challenges include:
- Generating sufficient high-quality training data
- Computational requirements for training and running the model
- Creating effective reward functions that capture the nuances of "trending" vs "baseline" topics
- Maintaining the system over time as conversation patterns evolve

### 2. Negative-Only Learning Approach

This approach focuses exclusively on what to avoid rather than what to find, providing a simpler but effective alternative to full reinforcement learning.

1. **Problem Reformulation**
   ```
   Instead of classifying topics as trending vs. non-trending, we reformulate as:
   - Identify and heavily penalize topics matching known baseline patterns
   - Reward topics based only on their distance from these patterns plus engagement metrics
   ```

2. **Negative Example Generation**
   ```python
   import json
   from google.generativeai import GenerativeModel, types
   
   def generate_negative_examples(num_examples=2000):
       """Generate comprehensive negative examples of baseline topics"""
       
       # Initialize Gemini model
       model = GenerativeModel('gemini-2.0-flash-lite')
       
       # Create detailed prompt for baseline topics
       negative_prompt = """
       Generate {n} examples of common, baseline topics that are ALWAYS discussed on 
       Farcaster and should NEVER be considered "trending".
       
       Focus on these categories:
       1. Cryptocurrencies and trading (Bitcoin, Ethereum, prices, market trends)
       2. Web3 infrastructure concepts (layer-2, rollups, zero-knowledge proofs)
       3. NFTs and digital collectibles (PFP projects, marketplaces)
       4. General social media topics (follows, likes, algo, engagement)
       5. General technology and AI (AI advancements, programming languages)
       
       For EACH example, include:
       - Topic name (the generic topic)
       - Description of why this is a baseline topic
       - At least 10 related subtopics or specific instances
       - 5 example keywords
       - 3 example posts
       - Estimated frequency (% of all posts)
       
       Format as a detailed JSON array.
       """
       
       # Split into multiple requests to stay within context limits
       batch_size = 100
       all_examples = []
       
       for i in range(0, num_examples, batch_size):
           batch_count = min(batch_size, num_examples - i)
           
           response = model.generate_content(
               negative_prompt.format(n=batch_count),
               config=types.GenerateContentConfig(
                   temperature=0.7,
                   response_mime_type="application/json"
               )
           )
           
           try:
               batch_examples = json.loads(response.text)
               all_examples.extend(batch_examples)
               print(f"Generated batch of {len(batch_examples)} examples")
           except json.JSONDecodeError as e:
               print(f"Error parsing batch: {e}")
       
       # Process examples
       processed_examples = []
       subtopics_set = set()
       keywords_set = set()
       
       for ex in all_examples:
           # Extract all subtopics and keywords for a comprehensive negative list
           subtopics = ex.get('subtopics', [])
           keywords = ex.get('keywords', [])
           
           subtopics_set.update([s.lower() for s in subtopics])
           keywords_set.update([k.lower() for k in keywords])
           
           # Add the main topic to both sets
           topic_name = ex.get('topic_name', '').lower()
           if topic_name:
               subtopics_set.add(topic_name)
               keywords_set.add(topic_name)
           
           processed_examples.append({
               'topic': topic_name,
               'subtopics': subtopics,
               'keywords': keywords,
               'frequency': ex.get('frequency', '0%').replace('%', ''),
               'is_baseline': True
           })
       
       # Save complete baseline topics database
       baseline_db = {
           'topics': processed_examples,
           'all_subtopics': list(subtopics_set),
           'all_keywords': list(keywords_set)
       }
       
       with open('baseline_topics_database.json', 'w') as f:
           json.dump(baseline_db, f, indent=2)
           
       print(f"Generated {len(processed_examples)} baseline topics with {len(subtopics_set)} total subtopics")
       
       return baseline_db
   ```

3. **Negative Embedding Space Construction**
   ```python
   from sentence_transformers import SentenceTransformer
   import numpy as np
   import faiss
   
   def build_negative_embedding_space(baseline_db):
       """Create embedding space of negative examples for distancing"""
       
       # Initialize sentence transformer
       model = SentenceTransformer('all-MiniLM-L6-v2')
       
       # Get all text representations of baseline topics
       baseline_texts = []
       
       # Add main topics
       baseline_texts.extend([topic['topic'] for topic in baseline_db['topics']])
       
       # Add all subtopics
       baseline_texts.extend(baseline_db['all_subtopics'])
       
       # Add all keywords
       baseline_texts.extend(baseline_db['all_keywords'])
       
       # Remove duplicates
       baseline_texts = list(set([text for text in baseline_texts if text]))
       
       # Generate embeddings
       print(f"Generating embeddings for {len(baseline_texts)} baseline terms")
       baseline_embeddings = model.encode(baseline_texts, show_progress_bar=True)
       
       # Normalize embeddings for cosine similarity
       faiss.normalize_L2(baseline_embeddings)
       
       # Build FAISS index for fast similarity search
       dim = baseline_embeddings.shape[1]
       index = faiss.IndexFlatIP(dim)  # Inner product for cosine similarity
       index.add(baseline_embeddings)
       
       # Save index
       faiss.write_index(index, 'baseline_embeddings.index')
       
       # Save mapping of index to text
       with open('baseline_embedding_texts.json', 'w') as f:
           json.dump(baseline_texts, f)
           
       return index, baseline_texts, baseline_embeddings
   ```

4. **Topic Scoring with Negative Distance**
   ```python
   def score_topic_with_negative_distance(topic_name, keywords, index, baseline_texts, model):
       """Score a topic based on distance from known baseline topics
       
       Args:
           topic_name: Name of the potential trending topic
           keywords: List of keywords for the topic
           index: FAISS index of baseline embeddings
           baseline_texts: List of baseline texts corresponding to index
           model: SentenceTransformer model used for embeddings
           
       Returns:
           float: Novelty score 0-1 (higher = further from baseline topics)
       """
       # Get embeddings for topic name and keywords
       texts_to_embed = [topic_name] + keywords
       embeddings = model.encode(texts_to_embed)
       
       # Normalize embeddings
       faiss.normalize_L2(embeddings)
       
       # Search for nearest baseline topics
       k = 5  # Get 5 closest matches
       distances, indices = index.search(embeddings, k)
       
       # Get mean distance to negative examples
       # Convert cosine similarity to distance (1 - similarity)
       distances = 1 - distances
       avg_distance = np.mean(distances)
       
       # Identify closest baseline topics (for logging)
       closest_baselines = {}
       for i, text_idx in enumerate(indices.flatten()):
           text = texts_to_embed[i // k]
           baseline = baseline_texts[text_idx]
           distance = distances.flatten()[i]
           
           if text not in closest_baselines or distance < closest_baselines[text]['distance']:
               closest_baselines[text] = {
                   'baseline': baseline,
                   'distance': float(distance)
               }
       
       # Log closest matches
       for text, match in closest_baselines.items():
           print(f"'{text}' closest to baseline '{match['baseline']}' (distance: {match['distance']:.3f})")
       
       # Calculate specificity bonus
       words = topic_name.split()
       specificity_bonus = min(len(words), 5) / 5.0 * 0.2
       
       # Calculate final score (normalized to 0-1)
       # Reward greater distances from baseline topics
       novelty_score = min(1.0, (avg_distance * 1.2) + specificity_bonus)
       
       return novelty_score, closest_baselines
   ```

5. **Integration with Topic Detection Pipeline**
   ```python
   def apply_negative_only_filtering(candidate_topics):
       """Filter candidate topics using negative-only learning approach
       
       Args:
           candidate_topics: List of topic dictionaries from any approach
           
       Returns:
           list: Filtered and re-ranked topics
       """
       # Load embeddings and model
       model = SentenceTransformer('all-MiniLM-L6-v2')
       index = faiss.read_index('baseline_embeddings.index')
       
       with open('baseline_embedding_texts.json', 'r') as f:
           baseline_texts = json.load(f)
           
       # Score each topic
       for topic in candidate_topics:
           # Extract topic name and keywords
           if 'name' in topic:
               topic_name = topic['name']
               keywords = topic.get('key_terms', [])
           elif 'topic_data' in topic:
               topic_name = topic['topic_data'].get('topic_name', '')
               keywords = topic['topic_data'].get('key_terms', [])
               if not keywords and 'keywords' in topic:
                   keywords = topic['keywords']
           else:
               continue
           
           # Apply negative distance scoring
           novelty_score, closest_baselines = score_topic_with_negative_distance(
               topic_name, keywords, index, baseline_texts, model
           )
           
           # Store scoring results
           topic['novelty_score'] = novelty_score
           topic['closest_baselines'] = closest_baselines
           
           # Calculate engagement component (normalize by largest engagement score)
           max_engagement = max([t.get('engagement_score', 0) for t in candidate_topics if 'engagement_score' in t] or [1])
           engagement_score = topic.get('engagement_score', 0) / max_engagement if max_engagement > 0 else 0
           
           # Calculate final score (70% novelty, 30% engagement)
           topic['final_score'] = (novelty_score * 0.7) + (engagement_score * 0.3)
       
       # Sort by final score
       filtered_topics = sorted(candidate_topics, key=lambda x: x.get('final_score', 0), reverse=True)
       
       # Log top topics
       print("\nTop topics after negative-only filtering:")
       for i, topic in enumerate(filtered_topics[:5]):
           name = topic.get('name', topic.get('topic_data', {}).get('topic_name', 'Unknown'))
           score = topic.get('final_score', 0)
           novelty = topic.get('novelty_score', 0)
           print(f"{i+1}. {name} (score: {score:.3f}, novelty: {novelty:.3f})")
       
       return filtered_topics
   ```

This negative-only approach has several advantages:
1. Focuses on filtering out what we know with certainty (baseline topics)
2. Requires less labeled data than full reinforcement learning
3. Simpler to implement and conceptually more straightforward
4. More conservative approach that won't aggressively promote speculative trends

The main challenge is tuning the distance threshold to avoid over-filtering while still removing common topics.

### 3. Generator-Critic LLM Approach

This approach uses two language models in a GAN-like architecture to iteratively discover and refine trending topics.

1. **Generator LLM Implementation**
   ```python
   from transformers import GPTNeoForCausalLM, GPTNeoConfig, AutoTokenizer
   import torch
   
   class TopicGeneratorLLM:
       """LLM that generates candidate trending topics"""
       
       def __init__(self, model_path="EleutherAI/gpt-neo-1.3B"):
           self.model = GPTNeoForCausalLM.from_pretrained(model_path)
           self.tokenizer = AutoTokenizer.from_pretrained(model_path)
           self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
           self.model.to(self.device)
           
       def generate_topics(self, input_data, num_topics=10, temperature=0.7):
           """Generate candidate trending topics from input data
           
           Args:
               input_data: Dictionary with topic context
               num_topics: Number of topics to generate
               temperature: Sampling temperature
               
           Returns:
               list: Generated candidate topics
           """
           # Format input data
           recent_posts = input_data.get('sample_posts', [])
           engagement_data = input_data.get('engagement_data', {})
           
           # Create prompt
           prompt = f"""
           Based on the following Farcaster posts and engagement data, generate {num_topics} 
           potential trending topics. For each topic include a name, explanation, and keywords.
           
           RECENT POSTS:
           {' '.join(recent_posts[:20])}
           
           ENGAGEMENT DATA:
           {str(engagement_data)}
           
           TRENDING TOPICS:
           """
           
           # Generate topics
           inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
           
           # Generate with sampling
           generated_ids = self.model.generate(
               inputs.input_ids,
               max_length=1024,
               do_sample=True,
               temperature=temperature,
               top_p=0.95,
               num_return_sequences=1
           )
           
           # Decode and parse topics
           generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
           
           # Parse topics from text (simplified; in production would use more robust parsing)
           topics = []
           for line in generated_text.split("\n"):
               if ":" in line:
                   parts = line.split(":", 1)
                   if len(parts) == 2:
                       topic_name = parts[0].strip()
                       explanation = parts[1].strip()
                       topics.append({
                           "topic_name": topic_name,
                           "explanation": explanation,
                           "keywords": []  # Would extract from explanation in production
                       })
                   
                   if len(topics) >= num_topics:
                       break
           
           return topics
       
       def fine_tune(self, training_data, learning_rate=5e-5, epochs=3):
           """Fine-tune the generator on successful topics
           
           Args:
               training_data: List of successfully rated topics with prompts
               learning_rate: Learning rate for training
               epochs: Number of training epochs
           """
           # Format training data
           texts = []
           for example in training_data:
               prompt = example["prompt"]
               target = example["successful_topic"]
               texts.append(f"{prompt}\n{target}")
           
           # Tokenize
           encodings = self.tokenizer("\n\n".join(texts), return_tensors="pt")
           
           # Create PyTorch dataset
           dataset = torch.utils.data.TensorDataset(
               encodings["input_ids"],
               encodings["attention_mask"]
           )
           
           # Training parameters
           training_args = {
               "per_device_train_batch_size": 4,
               "learning_rate": learning_rate,
               "num_train_epochs": epochs,
               "gradient_accumulation_steps": 4,
           }
           
           # Simplified training loop (in production would use HuggingFace Trainer)
           optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
           
           for epoch in range(epochs):
               total_loss = 0
               for batch in dataset:
                   inputs = {"input_ids": batch[0].to(self.device),
                            "attention_mask": batch[1].to(self.device),
                            "labels": batch[0].to(self.device)}
                   
                   outputs = self.model(**inputs)
                   loss = outputs.loss
                   
                   loss.backward()
                   optimizer.step()
                   optimizer.zero_grad()
                   
                   total_loss += loss.item()
                   
               print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataset):.4f}")
   ```

2. **Critic LLM Implementation**
   ```python
   from google.generativeai import GenerativeModel, types
   
   class TopicCriticLLM:
       """LLM that evaluates candidate trending topics"""
       
       def __init__(self, model_name="gemini-2.0-flash-lite"):
           self.model = GenerativeModel(model_name)
           
       def evaluate_topic(self, topic, context_data):
           """Evaluate a candidate topic's potential as a trending topic
           
           Args:
               topic: Dictionary with topic name, explanation, keywords
               context_data: Additional context (engagement data, sample posts)
               
           Returns:
               dict: Evaluation results with scores and reasoning
           """
           # Format topic data
           topic_name = topic.get("topic_name", "")
           explanation = topic.get("explanation", "")
           keywords = topic.get("keywords", [])
           
           # Get relevant sample posts
           sample_posts = context_data.get("sample_posts", [])
           
           # Create evaluation prompt as a structured request
           prompt = f"""
           Evaluate this candidate trending topic for Farcaster social network:
           
           TOPIC: {topic_name}
           EXPLANATION: {explanation}
           KEYWORDS: {', '.join(keywords)}
           
           Consider the following evaluation criteria:
           1. Specificity (1-10): How specific vs. generic is this topic?
           2. Novelty (1-10): How novel vs. common/baseline is this topic?
           3. Timeliness (1-10): How time-bound vs. evergreen is this topic?
           4. Engagement Potential (1-10): How likely to drive conversation?
           5. Viewer Appeal (1-10): How interesting to lurkers/readers?
           
           Return your evaluation as structured JSON with this schema:
           
           class Scores(TypedDict):
               specificity: int  # 1-10
               novelty: int  # 1-10
               timeliness: int  # 1-10
               engagement_potential: int  # 1-10
               viewer_appeal: int  # 1-10
               overall: int  # 1-10
               
           class Evaluation(TypedDict):
               scores: Scores
               reasoning: str
               improvement_suggestions: list[str]
               is_baseline_topic: bool
           """
           
           # Get structured evaluation
           response = self.model.generate_content(
               prompt,
               config=types.GenerateContentConfig(
                   temperature=0,
                   response_mime_type="application/json"
               )
           )
           
           # Parse response
           try:
               evaluation = json.loads(response.text)
               return evaluation
           except json.JSONDecodeError:
               # Fallback for parsing errors
               print("Error parsing critic evaluation")
               return {
                   "scores": {
                       "overall": 5,
                       "specificity": 5,
                       "novelty": 5,
                       "timeliness": 5,
                       "engagement_potential": 5,
                       "viewer_appeal": 5
                   },
                   "reasoning": "Error parsing evaluation",
                   "improvement_suggestions": [],
                   "is_baseline_topic": False
               }
   ```

3. **Generator-Critic Training Loop**
   ```python
   def train_generator_critic_system(input_data, iterations=10):
       """Train the generator-critic system through multiple iterations
       
       Args:
           input_data: Dictionary with sample posts, engagement data, etc.
           iterations: Number of training iterations
           
       Returns:
           dict: Trained models and best topics
       """
       # Initialize models
       generator = TopicGeneratorLLM()
       critic = TopicCriticLLM()
       
       # Track generated topics and scores across iterations
       all_topics = []
       successful_topics = []  # Topics with high scores
       
       for iteration in range(iterations):
           print(f"\nIteration {iteration+1}/{iterations}")
           
           # 1. Generate candidate topics
           candidates = generator.generate_topics(input_data, num_topics=20)
           
           # 2. Evaluate each candidate
           evaluated_topics = []
           for topic in candidates:
               evaluation = critic.evaluate_topic(topic, input_data)
               topic["evaluation"] = evaluation
               topic["overall_score"] = evaluation["scores"]["overall"]
               evaluated_topics.append(topic)
               
               # Log evaluation
               print(f"Topic: {topic['topic_name']}")
               print(f"Score: {topic['overall_score']}/10")
               print(f"Reasoning: {evaluation['reasoning'][:100]}...")
               
           # 3. Sort by score
           evaluated_topics.sort(key=lambda x: x["overall_score"], reverse=True)
           
           # 4. Save top topics
           all_topics.extend(evaluated_topics)
           top_topics = evaluated_topics[:5]  # Top 5 from this iteration
           
           # 5. Collect successful topics for training
           for topic in top_topics:
               if topic["overall_score"] >= 8:  # High-scoring topics
                   # Format as training example
                   successful_example = {
                       "prompt": f"Generate trending topics for Farcaster:\n\n",
                       "successful_topic": f"Topic: {topic['topic_name']}\nExplanation: {topic['explanation']}"
                   }
                   successful_topics.append(successful_example)
           
           # 6. Fine-tune generator on successful topics (if we have enough)
           if len(successful_topics) >= 5:
               print(f"Fine-tuning generator on {len(successful_topics)} successful topics")
               generator.fine_tune(successful_topics)
           
       # Return final models and all generated topics
       return {
           "generator": generator,
           "critic": critic,
           "all_topics": all_topics,
           "successful_topics": successful_topics
       }
   ```

4. **Using the Trained System**
   ```python
   def get_trending_topics_with_generator_critic(input_data, system):
       """Get trending topics using trained generator-critic system
       
       Args:
           input_data: Dictionary with recent posts and engagement data
           system: Trained generator-critic system
           
       Returns:
           list: Top trending topics
       """
       # Generate larger set of candidates
       candidates = system["generator"].generate_topics(input_data, num_topics=30)
       
       # Evaluate all candidates
       for topic in candidates:
           evaluation = system["critic"].evaluate_topic(topic, input_data)
           topic["evaluation"] = evaluation
           topic["overall_score"] = evaluation["scores"]["overall"]
           
           # Add detailed scores
           topic["specificity"] = evaluation["scores"]["specificity"]
           topic["novelty"] = evaluation["scores"]["novelty"]
           topic["timeliness"] = evaluation["scores"]["timeliness"]
           
       # Filter out baseline topics
       filtered_candidates = [
           topic for topic in candidates 
           if not topic["evaluation"].get("is_baseline_topic", False)
       ]
       
       # Sort by overall score
       filtered_candidates.sort(key=lambda x: x["overall_score"], reverse=True)
       
       # Return top topics
       return filtered_candidates[:5]
   ```

The Generator-Critic approach offers advantages:
1. Continuously adapts to changing patterns without manual intervention
2. Can discover unexpected, novel trends that rule-based systems might miss
3. Directly optimizes for user engagement potential
4. Requires no manually labeled datasets

However, it has important challenges:
1. Most complex approach requiring two LLMs working in tandem
2. Risk of reinforcing filter bubbles or optimizing for sensationalism
3. Requires careful prompt engineering to guide the critic's evaluation criteria
4. Computationally expensive and resource-intensive