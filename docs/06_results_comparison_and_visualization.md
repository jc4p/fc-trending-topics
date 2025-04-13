## Results Comparison and Visualization

1. **Compile Results**
   ```python
   import json
   import pandas as pd
   
   # Load results from each approach
   with open('approach1_results.json', 'r') as f:
       approach1_results = json.load(f)
   
   with open('approach2_results.json', 'r') as f:
       approach2_results = json.load(f)
       
   with open('approach3_results.json', 'r') as f:
       approach3_results = json.load(f)
   
   # Standardize format for comparison
   def standardize_topic(topic, approach_name, approach_method):
       """Convert topic data to standard format for comparison"""
       if approach_name == "Direct LLM":
           return {
               'approach': approach_name,
               'method': approach_method,
               'topic_name': topic.get('name', 'Unknown'),
               'explanation': topic.get('explanation', ''),
               'estimated_percentage': topic.get('estimated_percentage', 'Unknown'),
               'key_terms': topic.get('key_terms', []),
               'key_entities': topic.get('key_entities', []),
               'engagement_data': {
                   'level': 'Unknown',
                   'insight': ''
               }
           }
       elif approach_name == "LDA + K-Means":
           return {
               'approach': approach_name,
               'method': approach_method,
               'topic_name': topic['topic_data'].get('topic_name', 'Unknown'),
               'explanation': topic['topic_data'].get('explanation', ''),
               'estimated_percentage': topic['topic_data'].get('estimated_percentage', 'Unknown'),
               'key_terms': topic['topic_data'].get('key_terms', []) or topic.get('keywords', []),
               'key_entities': topic['topic_data'].get('key_terms', []),
               'cluster_size': topic['size'],
               'engagement_data': {
                   'level': topic['topic_data'].get('engagement_level', 'Unknown'),
                   'total': topic.get('total_engagement', 0),
                   'average': topic.get('avg_engagement', 0)
               }
           }
       else:  # Embeddings
           return {
               'approach': approach_name,
               'method': approach_method,
               'topic_name': topic['topic_data'].get('topic_name', 'Unknown'),
               'explanation': topic['topic_data'].get('explanation', ''),
               'estimated_percentage': 'Unknown',
               'key_terms': topic['topic_data'].get('key_terms', []),
               'key_entities': topic['topic_data'].get('key_entities', []),
               'cluster_size': topic['size'],
               'engagement_data': {
                   'insight': topic['topic_data'].get('engagement_insight', ''),
                   'avg_likes': topic.get('avg_likes', 0),
                   'avg_recasts': topic.get('avg_recasts', 0),
                   'total': topic.get('total_engagement', 0)
               }
           }
   
   # Standardize all topics
   standardized_topics = []
   
   # Approach 1: Direct LLM
   for topic in approach1_results.get('topics', []):
       standardized_topics.append(standardize_topic(topic, "Direct LLM", "Gemini 1.5 Pro"))
       
   # Approach 2: LDA + K-Means
   for topic in approach2_results:
       standardized_topics.append(standardize_topic(topic, "LDA + K-Means", "LDA with K=20, K-Means with K=5"))
       
   # Approach 3: Embeddings
   for topic in approach3_results:
       standardized_topics.append(standardize_topic(topic, "Embeddings", "all-MiniLM-L6-v2 with HDBSCAN"))
   
   # Convert to DataFrame for easier analysis
   topics_df = pd.DataFrame(standardized_topics)
   
   # Save combined results
   with open('all_trending_topics.json', 'w') as f:
       json.dump(standardized_topics, f, indent=2)
       
   print(f"Compiled {len(standardized_topics)} topics across all approaches")
   ```

2. **Generate Comparison Report and Visualizations**
   ```python
   import matplotlib.pyplot as plt
   import seaborn as sns
   from matplotlib.gridspec import GridSpec
   import re
   import numpy as np
   from datetime import datetime
   
   # Set style
   plt.style.use('ggplot')
   sns.set(font_scale=1.2)
   
   # Create topic comparison figure
   plt.figure(figsize=(20, 15))
   gs = GridSpec(3, 2, figure=plt.gcf())
   
   # 1. Topic overlaps analysis
   # Look for similar topics across approaches
   similar_topics = []
   
   # Basic fuzzy matching to find similar topics
   for i, topic1 in enumerate(standardized_topics):
       for j, topic2 in enumerate(standardized_topics[i+1:], i+1):
           if topic1['approach'] == topic2['approach']:
               continue
               
           # Compare topic names (basic approach)
           name1 = topic1['topic_name'].lower()
           name2 = topic2['topic_name'].lower()
           
           # Check for shared words
           words1 = set(re.findall(r'\w+', name1))
           words2 = set(re.findall(r'\w+', name2))
           shared_words = words1.intersection(words2)
           
           # Also check key terms for overlap
           terms1 = set([t.lower() for t in topic1['key_terms']])
           terms2 = set([t.lower() for t in topic2['key_terms']])
           shared_terms = terms1.intersection(terms2)
           
           # If sufficient overlap, consider them similar
           if len(shared_words) >= 1 or len(shared_terms) >= 2:
               similar_topics.append({
                   'topic1': topic1,
                   'topic2': topic2,
                   'shared_words': list(shared_words),
                   'shared_terms': list(shared_terms)
               })
   
   # Plot topic overlap as network
   ax1 = plt.subplot(gs[0, 0])
   
   # Create labels and summary
   topic_overlap_summary = f"Found {len(similar_topics)} potential topic overlaps between approaches"
   ax1.text(0.5, 0.5, topic_overlap_summary, ha='center', va='center', fontsize=14)
   ax1.axis('off')
   
   # 2. Topic distribution by approach
   ax2 = plt.subplot(gs[0, 1])
   
   # Count topics by approach
   approach_counts = topics_df['approach'].value_counts()
   ax2.bar(approach_counts.index, approach_counts.values, color=sns.color_palette('Set2'))
   ax2.set_title('Number of Topics by Approach')
   ax2.set_ylabel('Count')
   ax2.set_ylim(0, max(approach_counts.values) + 1)
   
   # 3. Engagement analysis
   ax3 = plt.subplot(gs[1, :])
   
   # Calculate engagement metrics per topic where available
   engagement_data = []
   for topic in standardized_topics:
       if 'engagement_data' in topic and 'total' in topic['engagement_data']:
           engagement_data.append({
               'topic': topic['topic_name'],
               'approach': topic['approach'],
               'engagement': float(topic['engagement_data']['total'])
           })
   
   engagement_df = pd.DataFrame(engagement_data)
   
   if not engagement_df.empty:
       # Plot engagement by topic and approach
       sns.barplot(x='topic', y='engagement', hue='approach', data=engagement_df, ax=ax3)
       ax3.set_title('Topic Engagement Comparison')
       ax3.set_xlabel('Topic')
       ax3.set_ylabel('Total Engagement')
       ax3.tick_params(axis='x', rotation=45)
   else:
       ax3.text(0.5, 0.5, "No engagement data available", ha='center', va='center', fontsize=14)
       ax3.axis('off')
   
   # 4. Key topics table
   ax4 = plt.subplot(gs[2, :])
   ax4.axis('off')
   
   # Create a table of the top topics
   table_data = []
   for topic in standardized_topics:
       row = [
           topic['approach'],
           topic['topic_name'],
           topic['explanation'][:50] + ('...' if len(topic['explanation']) > 50 else ''),
           topic['estimated_percentage']
       ]
       table_data.append(row)
       
   table = ax4.table(
       cellText=table_data,
       colLabels=['Approach', 'Topic', 'Explanation', 'Est. %'],
       loc='center',
       cellLoc='center'
   )
   table.auto_set_font_size(False)
   table.set_fontsize(10)
   table.scale(1, 1.5)
   ax4.set_title('Trending Topics Comparison', pad=20)
   
   # Add timestamp and summary
   plt.suptitle(f'Farcaster Trending Topics Analysis\nPeriod: {time_threshold.strftime("%Y-%m-%d")} to {max_timestamp.strftime("%Y-%m-%d")}', 
                fontsize=16, y=0.98)
   
   plt.tight_layout(rect=[0, 0, 1, 0.96])
   plt.savefig('trending_topics_comparison.png', dpi=300, bbox_inches='tight')
   
   # Generate HTML report
   html_report = f"""
   <!DOCTYPE html>
   <html>
   <head>
       <title>Farcaster Trending Topics Analysis</title>
       <style>
           body {{ font-family: Arial, sans-serif; margin: 20px; }}
           h1, h2, h3 {{ color: #333; }}
           table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
           th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
           th {{ background-color: #f2f2f2; }}
           .topic-card {{ border: 1px solid #ccc; padding: 15px; margin: 10px 0; border-radius: 5px; }}
           .approach-1 {{ background-color: #e6f7ff; }}
           .approach-2 {{ background-color: #e6ffe6; }}
           .approach-3 {{ background-color: #fff5e6; }}
       </style>
   </head>
   <body>
       <h1>Farcaster Trending Topics Analysis</h1>
       <p><strong>Period:</strong> {time_threshold.strftime("%Y-%m-%d")} to {max_timestamp.strftime("%Y-%m-%d")}</p>
       
       <h2>Top 5 Trending Topics</h2>
       
       <div id="topics-container">
   """
   
   # Group topics by approach
   for approach in ['Direct LLM', 'LDA + K-Means', 'Embeddings']:
       approach_topics = [t for t in standardized_topics if t['approach'] == approach]
       approach_class = f"approach-{1 if approach == 'Direct LLM' else 2 if approach == 'LDA + K-Means' else 3}"
       
       html_report += f"""
       <h3>{approach}</h3>
       """
       
       for topic in approach_topics:
           html_report += f"""
           <div class="topic-card {approach_class}">
               <h3>{topic['topic_name']}</h3>
               <p><strong>Explanation:</strong> {topic['explanation']}</p>
               <p><strong>Estimated Percentage:</strong> {topic['estimated_percentage']}</p>
               <p><strong>Key Terms:</strong> {', '.join(topic['key_terms'])}</p>
           """
           
           if 'key_entities' in topic and topic['key_entities']:
               html_report += f"""
               <p><strong>Key Entities:</strong> {', '.join(topic['key_entities'])}</p>
               """
               
           if 'engagement_data' in topic:
               eng_data = topic['engagement_data']
               if 'level' in eng_data and eng_data['level'] != 'Unknown':
                   html_report += f"""
                   <p><strong>Engagement Level:</strong> {eng_data['level']}</p>
                   """
               
               if 'insight' in eng_data and eng_data['insight']:
                   html_report += f"""
                   <p><strong>Engagement Insight:</strong> {eng_data['insight']}</p>
                   """
                   
               if 'total' in eng_data and eng_data['total']:
                   html_report += f"""
                   <p><strong>Total Engagement:</strong> {eng_data['total']:.2f}</p>
                   """
                   
           html_report += """
           </div>
           """
   
   # Add overlap analysis
   html_report += f"""
       <h2>Topic Overlap Analysis</h2>
       <p>Found {len(similar_topics)} potential overlaps between topics from different approaches</p>
       
       <table>
           <tr>
               <th>Approach 1</th>
               <th>Topic 1</th>
               <th>Approach 2</th>
               <th>Topic 2</th>
               <th>Shared Terms</th>
           </tr>
   """
   
   for overlap in similar_topics:
       topic1 = overlap['topic1']
       topic2 = overlap['topic2']
       shared = overlap['shared_terms'] + overlap['shared_words']
       shared_str = ', '.join(set(shared))
       
       html_report += f"""
           <tr>
               <td>{topic1['approach']}</td>
               <td>{topic1['topic_name']}</td>
               <td>{topic2['approach']}</td>
               <td>{topic2['topic_name']}</td>
               <td>{shared_str}</td>
           </tr>
       """
   
   html_report += """
       </table>
       
       <h2>Visualization</h2>
       <img src="trending_topics_comparison.png" alt="Trending Topics Visualization" style="max-width:100%;">
       
   </body>
   </html>
   """
   
   # Save HTML report
   with open('trending_topics_report.html', 'w') as f:
       f.write(html_report)
   
   print("Generated comparison report and visualizations")
   ```

3. **Interactive Dashboard (Optional)**
   ```python
   import streamlit as st
   
   # Create simple dashboard to explore topics
   # ...
   ```