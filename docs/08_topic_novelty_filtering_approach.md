## Topic Novelty Filtering Approach

1. **Generate Baseline Topics List with Gemini**
   ```python
   from google.generativeai import GenerativeModel
   from google.generativeai import types
   import json
   
   # Initialize Gemini model
   model = GenerativeModel('gemini-2.0-flash-lite')
   
   # Create a detailed prompt for generating baseline/common topics
   baseline_topics_prompt = """
   Generate a comprehensive list of baseline topics that would be commonly discussed 
   on a crypto/web3 social network like Farcaster. These shouldn't be considered 
   "trending" because they're always discussed. Focus on these categories:
   
   1. Cryptocurrencies and tokens
   2. Blockchain technology concepts
   3. Web3 infrastructure
   4. NFTs and digital collectibles 
   5. DeFi protocols and concepts
   6. Common social media topics
   7. General tech industry topics
   
   Format as a JSON object with this structure:
   {
     "topics": [
       {
         "topic": "Cryptocurrency",
         "subtopics": ["Bitcoin", "Ethereum", "Trading", "Price Discussion", "Mining"]
       },
       {
         "topic": "Web3 Infrastructure",
         "subtopics": ["Decentralization", "Protocols", "Wallets", "L2 Solutions"]
       },
       ...
     ]
   }
   
   Include at least 10 main topic categories with 5-10 subtopics each.
   """
   
   # Get baseline topics from Gemini
   response = model.generate_content(
       baseline_topics_prompt,
       config=types.GenerateContentConfig(
           temperature=0,
           response_mime_type="application/json"
       )
   )
   
   try:
       baseline_topics = json.loads(response.text)
   except json.JSONDecodeError:
       # Fallback if JSON parsing fails
       print("Error parsing JSON response, using default baseline topics")
       baseline_topics = {
           "topics": [
               {
                   "topic": "Cryptocurrency",
                   "subtopics": ["Bitcoin", "Ethereum", "Trading", "Price", "Mining"]
               },
               {
                   "topic": "Web3",
                   "subtopics": ["Blockchain", "Smart Contracts", "Tokens"]
               }
           ]
       }
   
   # Create flat list and hierarchical relationships
   flat_baseline_topics = []
   topic_hierarchy = {}
   
   print("Building baseline topics filter...")
   
   for topic_group in baseline_topics['topics']:
       main_topic = topic_group['topic'].lower()
       flat_baseline_topics.append(main_topic)
       
       subtopics = [st.lower() for st in topic_group.get('subtopics', [])]
       flat_baseline_topics.extend(subtopics)
       
       # Build hierarchy
       for subtopic in subtopics:
           topic_hierarchy[subtopic] = main_topic
   
   print(f"Generated {len(flat_baseline_topics)} baseline topics to filter")
   
   # Save baseline topics for reference and debugging
   with open('baseline_topics.json', 'w') as f:
       json.dump({
           'flat_list': flat_baseline_topics,
           'hierarchy': topic_hierarchy,
           'original': baseline_topics
       }, f, indent=2)
   ```

2. **Implement Topic Novelty Scoring**
   ```python
   def score_topic_novelty(topic_name, keywords, topic_hierarchy=topic_hierarchy, baselines=flat_baseline_topics):
       """Score a topic based on how novel/specific it is (vs baseline topics)
       
       Args:
           topic_name: The name of the potential trending topic
           keywords: List of keywords associated with the topic
           topic_hierarchy: Dictionary mapping subtopics to parent topics
           baselines: List of baseline topics to filter against
           
       Returns:
           float: Novelty score between 0-1 (higher = more novel/trending)
       """
       topic_name_lower = topic_name.lower()
       words = topic_name_lower.split()
       
       # Check for direct match with baseline topics (strongest penalty)
       if topic_name_lower in baselines:
           print(f"Topic '{topic_name}' is a baseline topic - applying strong penalty")
           return 0.3  # Heavy penalty for exact matches
       
       # Check for partial matches with baseline topics
       for baseline in baselines:
           # Skip very short baselines to avoid overfitting
           if len(baseline.split()) <= 1:
               continue
               
           # If baseline completely contains topic, apply penalty
           if baseline in topic_name_lower:
               print(f"Topic '{topic_name}' contains baseline '{baseline}' - applying penalty")
               return 0.4
       
       # Check if any words in topic name are baseline topics
       word_matches = sum(1 for word in words if word in baselines)
       word_penalty = (word_matches / len(words)) * 0.3 if words else 0
       
       # Partial match for keywords
       keyword_matches = sum(1 for kw in keywords if kw.lower() in baselines)
       keyword_penalty = (keyword_matches / len(keywords)) * 0.2 if keywords else 0
       
       # Reward specificity (longer topic names tend to be more specific)
       specificity_bonus = min(len(words), 5) / 5.0 * 0.3  # Max bonus at 5+ words
       
       # If topic is very specific and contains a single baseline word, reduce penalty
       if len(words) >= 3 and word_matches == 1:
           word_penalty = word_penalty * 0.5
       
       # Calculate final novelty score
       novelty_score = 1.0 - word_penalty - keyword_penalty + specificity_bonus
       
       # Ensure score stays in 0-1 range
       return max(0.1, min(novelty_score, 1.0))
   ```

3. **Apply Filtering to All Approaches**
   ```python
   # Function to apply novelty filtering to topic results from any approach
   def apply_novelty_filtering(topics, approach_name):
       """Apply novelty filtering to identified topics
       
       Args:
           topics: List of topic dictionaries from any approach
           approach_name: Name of the approach for logging
           
       Returns:
           list: Filtered and re-ranked topics
       """
       print(f"Applying novelty filtering to {len(topics)} topics from {approach_name}...")
       
       for topic in topics:
           # Extract topic name and keywords based on approach format
           if approach_name == "Direct LLM":
               topic_name = topic.get('name', '')
               keywords = topic.get('key_terms', [])
           elif approach_name in ["LDA + K-Means", "Embeddings"]:
               topic_name = topic['topic_data'].get('topic_name', '')
               # Keywords could be in different formats
               keywords = topic['topic_data'].get('key_terms', [])
               if not keywords and 'keywords' in topic:
                   keywords = topic['keywords']
           
           # Calculate novelty score
           novelty_score = score_topic_novelty(topic_name, keywords)
           
           # Store scores
           topic['novelty_score'] = novelty_score
           
           # Calculate final score based on approach
           if approach_name == "Direct LLM":
               # For LLM, use estimated percentage and novelty
               est_pct = float(topic.get('estimated_percentage', '5%').replace('%', '')) / 100
               topic['final_score'] = est_pct * novelty_score
           elif approach_name == "LDA + K-Means":
               # For LDA, use size and novelty
               size_score = min(topic['size'] / 1000, 1.0)  # Normalize size
               topic['final_score'] = (size_score * 0.7) + (novelty_score * 0.3)
           elif approach_name == "Embeddings":  
               # For embeddings, use coherence, size and novelty
               size_score = min(topic['size'] / 1000, 1.0)  # Normalize size
               coherence = float(topic.get('coherence_score', 0.5))
               topic['final_score'] = (size_score * 0.4) + (coherence * 0.3) + (novelty_score * 0.3)
       
       # Sort by final score
       filtered_topics = sorted(topics, key=lambda x: x.get('final_score', 0), reverse=True)
       
       # Log top and bottom topics by novelty
       top_novel = sorted(topics, key=lambda x: x.get('novelty_score', 0), reverse=True)[:3]
       bottom_novel = sorted(topics, key=lambda x: x.get('novelty_score', 0))[:3]
       
       print("Most novel topics:")
       for t in top_novel:
           name = t.get('name', t.get('topic_data', {}).get('topic_name', 'Unknown'))
           print(f"  - {name}: {t.get('novelty_score', 0):.2f}")
       
       print("Least novel topics:")
       for t in bottom_novel:
           name = t.get('name', t.get('topic_data', {}).get('topic_name', 'Unknown'))
           print(f"  - {name}: {t.get('novelty_score', 0):.2f}")
       
       return filtered_topics
   
   # Apply to LLM results
   llm_topics_filtered = apply_novelty_filtering(
       approach1_results.get('topics', []), 
       "Direct LLM"
   )
   
   # Apply to LDA+K-Means results
   lda_topics_filtered = apply_novelty_filtering(
       approach2_results, 
       "LDA + K-Means"
   )
   
   # Apply to Embedding results 
   embedding_topics_filtered = apply_novelty_filtering(
       approach3_results, 
       "Embeddings"
   )
   
   # Save filtered results
   filtered_results = {
       'direct_llm': llm_topics_filtered[:5],  # Top 5 after filtering
       'lda_kmeans': lda_topics_filtered[:5],
       'embeddings': embedding_topics_filtered[:5]
   }
   
   with open('trending_topics_filtered.json', 'w') as f:
       # Custom encoder to handle any non-serializable objects
       class CustomEncoder(json.JSONEncoder):
           def default(self, obj):
               try:
                   return float(obj)
               except:
                   return str(obj)
                   
       json.dump(filtered_results, f, indent=2, cls=CustomEncoder)
   ```

4. **Generate Combined Report with Novelty Metrics**
   ```python
   # Create HTML report with novelty information
   html_report = f"""
   <!DOCTYPE html>
   <html>
   <head>
       <title>Farcaster Trending Topics Analysis - With Novelty Filtering</title>
       <style>
           body {{ font-family: Arial, sans-serif; margin: 20px; }}
           h1, h2, h3 {{ color: #333; }}
           .topic-card {{ border: 1px solid #ccc; padding: 15px; margin: 10px 0; border-radius: 5px; }}
           .high-novelty {{ background-color: #e6fffa; border-left: 5px solid #38b2ac; }}
           .medium-novelty {{ background-color: #fefcbf; border-left: 5px solid #d69e2e; }}
           .low-novelty {{ background-color: #fed7d7; border-left: 5px solid #e53e3e; }}
           .novelty-score {{ float: right; padding: 5px 10px; border-radius: 20px; font-weight: bold; }}
           .high {{ background-color: #38b2ac; color: white; }}
           .medium {{ background-color: #d69e2e; color: white; }}
           .low {{ background-color: #e53e3e; color: white; }}
           .metrics {{ color: #666; font-size: 0.9em; }}
       </style>
   </head>
   <body>
       <h1>Farcaster Trending Topics with Novelty Filtering</h1>
       <p>Period: {time_threshold.strftime("%Y-%m-%d")} to {max_timestamp.strftime("%Y-%m-%d")}</p>
       
       <h2>Filtered Top Trending Topics</h2>
       <p>Topics below have been filtered to prioritize specific, novel conversations over general baseline topics.</p>
   """
   
   # Add section for each approach
   for approach, topics in filtered_results.items():
       approach_name = approach.replace('_', ' ').title()
       html_report += f"<h3>{approach_name} Approach</h3>"
       
       for topic in topics:
           # Get topic name based on approach
           if approach == 'direct_llm':
               name = topic.get('name', 'Unknown')
               explanation = topic.get('explanation', '')
               keywords = ', '.join(topic.get('key_terms', [])[:5])
           else:
               name = topic.get('topic_data', {}).get('topic_name', 'Unknown')
               explanation = topic.get('topic_data', {}).get('explanation', '')
               keywords = ', '.join(topic.get('topic_data', {}).get('key_terms', [])[:5])
           
           # Get novelty score and determine styling
           novelty = topic.get('novelty_score', 0.5)
           if novelty >= 0.8:
               novelty_class = "high-novelty"
               score_class = "high"
           elif novelty >= 0.5:
               novelty_class = "medium-novelty"
               score_class = "medium"
           else:
               novelty_class = "low-novelty"
               score_class = "low"
           
           # Generate topic card
           html_report += f"""
           <div class="topic-card {novelty_class}">
               <span class="novelty-score {score_class}">Novelty: {novelty:.2f}</span>
               <h3>{name}</h3>
               <p>{explanation}</p>
               <p><strong>Key terms:</strong> {keywords}</p>
               <div class="metrics">
                   <p><strong>Final score:</strong> {topic.get('final_score', 0):.3f}</p>
               </div>
           </div>
           """
   
   # Add baseline topics section
   html_report += f"""
       <h2>Baseline Topics</h2>
       <p>These are common topics that were filtered out or downranked as they're typically discussed in the community.</p>
       <ul>
   """
   
   # Add top 20 baseline topics
   for i, topic_group in enumerate(baseline_topics['topics'][:10]):
       html_report += f"<li><strong>{topic_group['topic']}</strong>: {', '.join(topic_group['subtopics'][:5])}</li>"
   
   html_report += """
       </ul>
   </body>
   </html>
   """
   
   # Save HTML report
   with open('trending_topics_novelty_report.html', 'w') as f:
       f.write(html_report)
   
   print("Generated trending topics report with novelty filtering")
   ```