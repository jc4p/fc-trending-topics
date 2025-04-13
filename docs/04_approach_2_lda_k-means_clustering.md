## Approach 2: LDA + K-Means Clustering

1. **Text Vectorization**
   ```python
   from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
   
   # Create document-term matrix
   vectorizer = CountVectorizer(
       max_features=5000, 
       stop_words='english', 
       min_df=5
   )
   X = vectorizer.fit_transform(recent_df['cleaned_text'])
   feature_names = vectorizer.get_feature_names_out()
   ```

2. **LDA Topic Modeling and Topic Similarity Analysis**
   ```python
   from sklearn.decomposition import LatentDirichletAllocation
   from sklearn.metrics.pairwise import cosine_similarity
   import numpy as np
   import matplotlib.pyplot as plt
   import seaborn as sns
   from scipy.cluster.hierarchy import linkage, fcluster
   
   # Fit LDA model
   n_topics = 30  # Start with more topics than we need
   lda = LatentDirichletAllocation(
       n_components=n_topics,
       random_state=42,
       max_iter=25
   )
   lda_output = lda.fit_transform(X)
   
   # Get topic-term distributions
   topic_keywords = []
   for topic_idx, topic in enumerate(lda.components_):
       top_keywords_idx = topic.argsort()[:-10-1:-1]
       top_keywords = [feature_names[i] for i in top_keywords_idx]
       topic_keywords.append(top_keywords)
   
   # Calculate topic coherence scores
   from sklearn.metrics import silhouette_score
   
   # Use silhouette score to measure how well-separated the topics are
   if n_topics > 1:  # Need at least 2 topics for silhouette score
       silhouette_avg = silhouette_score(X, lda_output.argmax(axis=1))
       print(f"Silhouette Score: {silhouette_avg:.3f}")
   
   # Calculate similarity between topics
   topic_similarity = cosine_similarity(lda.components_)
   
   # Visualize topic similarity with enhanced Seaborn styling
   plt.figure(figsize=(14, 12))
   sns.set(font_scale=1.1)
   mask = np.triu(np.ones_like(topic_similarity, dtype=bool))  # Create mask for upper triangle
   with sns.axes_style("white"):
       ax = sns.heatmap(
           topic_similarity, 
           annot=True, 
           cmap="YlGnBu", 
           fmt=".2f",
           linewidths=0.5,
           mask=mask,  # Only show lower triangle to reduce redundancy
           cbar_kws={'label': 'Cosine Similarity'}
       )
   plt.title('Topic Similarity Matrix', fontsize=16, fontweight='bold', pad=20)
   plt.tight_layout()
   plt.savefig('topic_similarity_matrix.png', dpi=300)
   
   # Hierarchical clustering of topics based on similarity
   # This will help identify groups of similar topics
   topic_linkage = linkage(1 - topic_similarity, method='ward')
   
   # Plot dendrogram to visualize topic clusters
   plt.figure(figsize=(14, 7))
   from scipy.cluster.hierarchy import dendrogram
   
   def fancy_dendrogram(*args, **kwargs):
       max_d = kwargs.pop('max_d', None)
       if max_d and 'color_threshold' not in kwargs:
           kwargs['color_threshold'] = max_d
       plt.figure(figsize=(12, 8))
       ddata = dendrogram(*args, **kwargs)
       if max_d:
           plt.axhline(y=max_d, c='k', ls='--', lw=1)
       return ddata
       
   fancy_dendrogram(
       topic_linkage,
       labels=[f"Topic {i+1}" for i in range(n_topics)],
       leaf_font_size=10,
       orientation='right'
   )
   plt.title('Hierarchical Clustering of Topics')
   plt.xlabel('Distance')
   plt.savefig('topic_clustering_dendrogram.png')
   
   # Determine optimal number of topic clusters
   topic_cluster_threshold = 0.7  # Similarity threshold
   topic_clusters = fcluster(topic_linkage, t=topic_cluster_threshold, criterion='distance')
   
   # Create mapping of original topics to consolidated topics
   topic_mapping = {}
   consolidated_topic_keywords = []
   
   # For each cluster, find the most representative topic
   for cluster_id in range(1, max(topic_clusters) + 1):
       # Get all topics in this cluster
       cluster_topic_indices = [i for i, c in enumerate(topic_clusters) if c == cluster_id]
       
       # If cluster has only one topic, keep it as is
       if len(cluster_topic_indices) == 1:
           topic_idx = cluster_topic_indices[0]
           topic_mapping[topic_idx] = len(consolidated_topic_keywords)
           consolidated_topic_keywords.append(topic_keywords[topic_idx])
       else:
           # Find the topic with highest topic coherence (using average document probability as proxy)
           topic_coherence = [lda_output[:, idx].mean() for idx in cluster_topic_indices]
           representative_topic_idx = cluster_topic_indices[np.argmax(topic_coherence)]
           
           # For all topics in cluster, map to the representative topic
           for topic_idx in cluster_topic_indices:
               topic_mapping[topic_idx] = len(consolidated_topic_keywords)
               
           # Merge keywords from all topics in cluster
           merged_keywords = []
           for topic_idx in cluster_topic_indices:
               merged_keywords.extend(topic_keywords[topic_idx])
           
           # Keep unique keywords, prioritizing those from the representative topic
           unique_merged_keywords = []
           seen = set()
           
           # First add keywords from representative topic
           for keyword in topic_keywords[representative_topic_idx]:
               if keyword not in seen:
                   unique_merged_keywords.append(keyword)
                   seen.add(keyword)
           
           # Then add unique keywords from other topics
           for keyword in merged_keywords:
               if keyword not in seen and len(unique_merged_keywords) < 15:
                   unique_merged_keywords.append(keyword)
                   seen.add(keyword)
           
           consolidated_topic_keywords.append(unique_merged_keywords)
   
   print(f"Reduced {n_topics} original topics to {len(consolidated_topic_keywords)} consolidated topics")
   
   # Map documents to consolidated topics
   document_consolidated_topics = np.zeros((lda_output.shape[0], len(consolidated_topic_keywords)))
   
   for doc_idx in range(lda_output.shape[0]):
       for orig_topic_idx in range(n_topics):
           consolidated_idx = topic_mapping[orig_topic_idx]
           document_consolidated_topics[doc_idx, consolidated_idx] += lda_output[doc_idx, orig_topic_idx]
   
   # Normalize to ensure probabilities sum to 1
   document_consolidated_topics = document_consolidated_topics / document_consolidated_topics.sum(axis=1, keepdims=True)
   
   # Replace original LDA output with consolidated version
   lda_output = document_consolidated_topics
   topic_keywords = consolidated_topic_keywords
   n_consolidated_topics = len(consolidated_topic_keywords)
   
   print(f"Consolidated topics and their top keywords:")
   for i, keywords in enumerate(consolidated_topic_keywords):
       print(f"Topic {i+1}: {', '.join(keywords[:10])}")
   ```

3. **K-Means Clustering on Consolidated LDA Results**
   ```python
   from sklearn.cluster import KMeans
   from sklearn.metrics import silhouette_score
   
   # Determine optimal number of clusters
   # We'll try a range of clusters and pick the one with best silhouette score
   silhouette_scores = []
   K_range = range(3, 8)  # Try between 3 and 7 clusters
   
   for k in K_range:
       kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
       clusters = kmeans.fit_predict(lda_output)
       score = silhouette_score(lda_output, clusters)
       silhouette_scores.append(score)
       print(f"K={k}, Silhouette Score: {score:.3f}")
   
   # Plot silhouette scores with Seaborn
   plt.figure(figsize=(10, 6))
   sns.set_style("whitegrid")
   sns.lineplot(x=list(K_range), y=silhouette_scores, marker='o', color='royalblue', linewidth=2.5)
   plt.xlabel('Number of Clusters (K)', fontsize=12)
   plt.ylabel('Silhouette Score', fontsize=12)
   plt.title('Optimal Number of Clusters', fontsize=14, fontweight='bold')
   plt.tight_layout()
   plt.savefig('optimal_k_clusters.png', dpi=300)
   
   # Find optimal K (highest silhouette score)
   optimal_k = K_range[np.argmax(silhouette_scores)]
   print(f"Optimal number of clusters: {optimal_k}")
   
   # Apply K-means clustering with optimal K
   kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
   clusters = kmeans.fit_predict(lda_output)
   
   # Assign cluster to each document
   recent_df['lda_cluster'] = clusters
   
   # Get cluster centers and dominant topics
   cluster_centers = kmeans.cluster_centers_
   
   # Identify the dominant LDA topics for each cluster
   dominant_topics_per_cluster = []
   for i, center in enumerate(cluster_centers):
       # Get top 3 LDA topics for this cluster
       top_topic_idx = center.argsort()[::-1][:3]
       top_topic_weights = center[top_topic_idx]
       
       cluster_topics = []
       for idx, weight in zip(top_topic_idx, top_topic_weights):
           cluster_topics.append({
               'topic_idx': int(idx),
               'weight': float(weight),
               'keywords': topic_keywords[idx][:10]
           })
           
       dominant_topics_per_cluster.append({
           'cluster_id': i,
           'dominant_topics': cluster_topics,
           'size': int((clusters == i).sum())
       })
       
       print(f"Cluster {i} (size: {(clusters == i).sum()}):")
       for topic in cluster_topics:
           print(f"  Topic {topic['topic_idx']} (weight: {topic['weight']:.3f}): {', '.join(topic['keywords'])}")
       
   # Visualize cluster distribution with Seaborn
   plt.figure(figsize=(10, 6))
   sns.set_style("whitegrid")
   ax = sns.barplot(
       x=list(range(len(cluster_sizes))), 
       y=cluster_sizes,
       palette="viridis"
   )
   # Add value labels on top of each bar
   for i, v in enumerate(cluster_sizes):
       ax.text(i, v + 0.1, str(v), ha='center', fontsize=10)
   plt.xlabel('Cluster ID', fontsize=12)
   plt.ylabel('Number of Documents', fontsize=12)
   plt.title('Documents per Cluster', fontsize=14, fontweight='bold')
   plt.xticks(range(len(cluster_sizes)))
   plt.tight_layout()
   plt.savefig('cluster_distribution.png', dpi=300)
   
   # Find exemplar documents for each cluster (closest to cluster center)
   from scipy.spatial.distance import cdist
   
   exemplars = {}
   for cluster_id in range(len(cluster_centers)):
       # Get documents in this cluster
       cluster_docs_idx = np.where(clusters == cluster_id)[0]
       
       # Calculate distance to cluster center
       cluster_docs_vectors = lda_output[cluster_docs_idx]
       distances = cdist(cluster_docs_vectors, [cluster_centers[cluster_id]], 'euclidean').flatten()
       
       # Get indices of 5 closest documents
       closest_indices = np.argsort(distances)[:5]
       exemplar_indices = cluster_docs_idx[closest_indices]
       
       # Get the actual documents
       exemplar_docs = recent_df.iloc[exemplar_indices]
       
       exemplars[cluster_id] = exemplar_docs[['Text', 'engagement_score']].to_dict('records')
       
       print(f"Exemplar documents for Cluster {cluster_id}:")
       for doc in exemplars[cluster_id][:2]:  # Just print 2 to save space
           print(f"  - {doc['Text'][:100]}..." if len(doc['Text']) > 100 else f"  - {doc['Text']}")
   ```

4. **Gemini Labeling of Clusters with Structured Response**
   ```python
   from enum import Enum
   from typing_extensions import TypedDict
   from google.generativeai import types
   
   # Define TypedDict classes for structured output
   class ClusterTopic(TypedDict):
       topic_name: str  # 5 words max
       explanation: str  # Brief explanation of why trending
       estimated_percentage: str  # Percentage of total conversation
       key_terms: list[str]
       engagement_level: str  # High, Medium, Low
       sentiment: str  # Positive, Neutral, Negative, Mixed
   
   # For each cluster, get representative texts
   cluster_data = []
   for cluster_id in range(5):
       # Get documents in this cluster
       cluster_docs = recent_df[recent_df['lda_cluster'] == cluster_id]
       
       # Get top keywords for this cluster
       top_topic_idx = np.argmax(cluster_centers[cluster_id])
       keywords = topic_keywords[top_topic_idx]
       
       # Calculate engagement metrics for this cluster
       total_engagement = cluster_docs['engagement_score'].sum()
       avg_engagement = cluster_docs['engagement_score'].mean()
       max_engagement = cluster_docs['engagement_score'].max()
       
       # Get high-engagement samples (top 50%)
       engagement_threshold = cluster_docs['engagement_score'].median()
       engaged_docs = cluster_docs[cluster_docs['engagement_score'] >= engagement_threshold]
       
       # Ensure we have some samples even if all have low engagement
       if len(engaged_docs) < 10:
           engaged_docs = cluster_docs.nlargest(min(10, len(cluster_docs)), 'engagement_score')
           
       # Get sample texts with engagement metrics
       sample_texts_with_engagement = engaged_docs.apply(
           lambda row: f"[👍{int(row['likes_count'])}|↗️{int(row['recasts_count'])}]: {row['Text']}", 
           axis=1
       ).tolist()
       
       cluster_data.append({
           'cluster_id': cluster_id,
           'size': len(cluster_docs),
           'keywords': keywords,
           'sample_texts': sample_texts_with_engagement,
           'total_engagement': total_engagement,
           'avg_engagement': avg_engagement,
           'max_engagement': max_engagement
       })
   
   # Use Gemini to label each cluster
   cluster_topics = []
   
   # Initialize Gemini
   model = GenerativeModel('gemini-2.0-flash-lite')
   
   for cluster in cluster_data:
       # Create structured prompt
       prompt = f"""
       I need to identify the single most specific trending topic being discussed in a cluster of Farcaster social media posts.
       
       KEY INFORMATION ABOUT THIS CLUSTER:
       - Top keywords: {', '.join(cluster['keywords'])}
       - Cluster size: {cluster['size']} posts
       - Average engagement: {cluster['avg_engagement']:.2f}
       
       SAMPLE POSTS (with like and recast counts):
       {' '.join(cluster['sample_texts'][:10])}
       
       Generate your response based on the following Python TypedDict schema:
       
       class ClusterTopic(TypedDict):
           topic_name: str  # 5 words max
           explanation: str  # Brief explanation of why trending
           estimated_percentage: str  # Percentage of total conversation
           key_terms: list[str]  # List of strings
           engagement_level: str  # High, Medium, Low
           sentiment: str  # Positive, Neutral, Negative, Mixed
       """
       
       # Get response with JSON formatting
       response = model.generate_content(
           prompt,
           config=types.GenerateContentConfig(
               temperature=0,
               response_mime_type="application/json"
           )
       )
       
       # Parse JSON response
       try:
           topic_data = json.loads(response.text)
           print(f"Successfully labeled cluster {cluster['cluster_id']} as '{topic_data.get('topic_name', 'Unknown')}'")
       except json.JSONDecodeError as e:
           print(f"Error parsing JSON for cluster {cluster['cluster_id']}: {e}")
           # Fallback if JSON parsing fails
           topic_data = {
               "topic_name": "Error parsing response",
               "explanation": "Could not parse JSON from Gemini response",
               "estimated_percentage": "unknown",
               "key_terms": cluster['keywords'][:5],
               "engagement_level": "unknown",
               "sentiment": "unknown"
           }
       
       # Add to results
       cluster_topics.append({
           'cluster_id': cluster['cluster_id'],
           'size': cluster['size'],
           'total_engagement': float(cluster['total_engagement']),
           'avg_engagement': float(cluster['avg_engagement']),
           'topic_data': topic_data,
           'keywords': cluster['keywords']
       })
       
   # Save intermediate results
   with open('approach2_results.json', 'w') as f:
       json.dump(cluster_topics, f, indent=2)
   ```