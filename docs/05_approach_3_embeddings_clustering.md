## Approach 3: Embeddings + Clustering

1. **Generate Embeddings**
   ```python
   from sentence_transformers import SentenceTransformer
   
   # Initialize transformer model
   embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
   
   # Sample for embeddings if dataset is too large
   embedding_sample_size = min(50000, len(recent_df))
   embedding_df = recent_df.sample(embedding_sample_size)
   
   # Generate embeddings
   embeddings = embedding_model.encode(
       embedding_df['cleaned_text'].tolist(), 
       show_progress_bar=True,
       batch_size=64
   )
   ```

2. **Dimensionality Reduction (Optional)**
   ```python
   from sklearn.decomposition import PCA
   
   # Reduce dimensions for visualization and clustering
   pca = PCA(n_components=50)
   reduced_embeddings = pca.fit_transform(embeddings)
   ```

3. **Advanced Clustering with Similarity Analysis**
   ```python
   import hdbscan
   from sklearn.metrics import pairwise_distances
   import matplotlib.pyplot as plt
   import seaborn as sns
   from scipy.cluster.hierarchy import linkage, fcluster
   import numpy as np
   
   # Step 1: Run HDBSCAN with more sensitive parameters to get more initial clusters
   clusterer = hdbscan.HDBSCAN(
       min_cluster_size=50,  # Smaller size to capture more potential topics
       min_samples=5,        # More sensitive grouping
       metric='euclidean',
       cluster_selection_method='eom'  # Excess of Mass - tends to find more clusters
   )
   embedding_df['cluster'] = clusterer.fit_predict(reduced_embeddings)
   
   # Filter out noise (-1 cluster)
   valid_clusters = embedding_df[embedding_df['cluster'] != -1]
   print(f"Initial clustering found {embedding_df['cluster'].nunique() - 1} clusters")
   print(f"Retained {len(valid_clusters)} documents in clusters ({len(valid_clusters)/len(embedding_df)*100:.1f}%)")
   
   # Visualization: Plot cluster sizes with enhanced styling
   cluster_sizes = valid_clusters['cluster'].value_counts()
   plt.figure(figsize=(14, 7))
   sns.set(style="whitegrid", font_scale=1.1)
   
   # Create a color palette that goes from light to dark based on cluster size
   palette = sns.color_palette("viridis", n_colors=len(cluster_sizes))
   # Sort palette by cluster size
   sorted_indices = np.argsort(cluster_sizes.values)
   sorted_palette = [palette[i] for i in sorted_indices]
   
   ax = sns.barplot(
       x=cluster_sizes.index,
       y=cluster_sizes.values,
       palette=sorted_palette,
       order=cluster_sizes.index[sorted_indices]  # Sort by size
   )
   
   # Add count labels on top of bars
   for i, v in enumerate(sorted(cluster_sizes.values)):
       ax.text(i, v + 5, str(v), ha='center', fontsize=9)
       
   plt.xlabel('Cluster ID', fontsize=12)
   plt.ylabel('Number of Documents', fontsize=12)
   plt.title('Documents per Initial Cluster (Sorted by Size)', fontsize=14, fontweight='bold')
   plt.xticks(rotation=45)
   plt.tight_layout()
   plt.savefig('initial_cluster_sizes.png', dpi=300)
   
   # Only proceed with similarity analysis if we have enough clusters
   if cluster_sizes.shape[0] > 5:
       # Step 2: Calculate cluster centers
       cluster_centers = {}
       for cluster_id in cluster_sizes.index:
           cluster_embeddings = reduced_embeddings[embedding_df['cluster'] == cluster_id]
           cluster_centers[cluster_id] = np.mean(cluster_embeddings, axis=0)
       
       # Step 3: Calculate similarity between clusters
       cluster_ids = list(cluster_centers.keys())
       center_vectors = np.array([cluster_centers[cid] for cid in cluster_ids])
       
       # Use cosine distance (1 - cosine similarity)
       from sklearn.metrics.pairwise import cosine_distances
       cluster_distances = cosine_distances(center_vectors)
       
       # Create a DataFrame for better visualization
       import pandas as pd
       cluster_similarity_df = pd.DataFrame(
           1 - cluster_distances,  # Convert distance to similarity
           index=cluster_ids,
           columns=cluster_ids
       )
       
       # Visualize cluster similarity with enhanced styling
       plt.figure(figsize=(16, 14))
       sns.set(font_scale=1.2)
       
       # Create a mask for the upper triangle to reduce redundancy
       mask = np.triu(np.ones_like(cluster_similarity_df), k=1)
       
       # Set up custom diverging colormap
       cmap = sns.diverging_palette(220, 20, as_cmap=True)
       
       with sns.axes_style("white"):
           ax = sns.heatmap(
               cluster_similarity_df, 
               annot=True, 
               cmap=cmap, 
               fmt=".2f",
               linewidths=0.5,
               mask=mask,
               vmin=0, vmax=1,
               cbar_kws={'label': 'Cosine Similarity', 'shrink': 0.8}
           )
           
       plt.title('Cluster Similarity Matrix', fontsize=18, fontweight='bold', pad=20)
       
       # Add a note about interpretation
       plt.figtext(0.5, 0.01, 
                   'Values closer to 1 indicate higher similarity between clusters', 
                   ha='center', fontsize=12, style='italic')
       
       plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust layout to make room for note
       plt.savefig('cluster_similarity_matrix.png', dpi=300)
       
       # Step 4: Hierarchical clustering to merge similar clusters
       # Use complete linkage to ensure all points in merged clusters are similar
       cluster_linkage = linkage(cluster_distances, method='complete')
       
       # Plot dendrogram to visualize cluster merging with enhanced styling
       plt.figure(figsize=(16, 10))
       from scipy.cluster.hierarchy import dendrogram
       
       # Create a custom color palette
       sns.set_palette("viridis", n_colors=len(cluster_ids))
       
       # Plot with more styling options
       with plt.style.context('seaborn-white'):
           dendrogram(
               cluster_linkage,
               labels=cluster_ids,
               leaf_font_size=11,
               orientation='right',
               leaf_rotation=0,  # Rotate labels for better readability
               color_threshold=0.7,  # Color threshold for visual grouping
               above_threshold_color='grey'
           )
           
       plt.title('Hierarchical Clustering of Embedding Clusters', 
                 fontsize=16, fontweight='bold', pad=20)
       plt.xlabel('Distance (1 - Cosine Similarity)', fontsize=12)
       
       # Add a grid for better readability of distances
       plt.grid(axis='x', linestyle='--', alpha=0.7)
       
       # Add an annotation explaining the dendrogram
       plt.figtext(0.5, 0.01, 
                   'Clusters that join at smaller distances are more similar to each other', 
                   ha='center', fontsize=12, style='italic')
                   
       plt.tight_layout(rect=[0, 0.03, 1, 0.97])
       plt.savefig('embedding_cluster_dendrogram.png', dpi=300)
       
       # Step 5: Determine optimal similarity threshold
       # Try different thresholds and check the resulting number of clusters
       thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
       merged_cluster_counts = []
       
       for threshold in thresholds:
           merged_clusters = fcluster(cluster_linkage, t=threshold, criterion='distance')
           merged_cluster_counts.append(len(np.unique(merged_clusters)))
       
       plt.figure(figsize=(12, 7))
       sns.set_style("whitegrid")
       sns.set_context("notebook", font_scale=1.2)
       
       # Create a more visually appealing plot
       ax = sns.lineplot(
           x=thresholds, 
           y=merged_cluster_counts, 
           marker='o', 
           markersize=10,
           linewidth=2.5,
           color='royalblue'
       )
       
       # Add value annotations
       for i, count in enumerate(merged_cluster_counts):
           ax.text(thresholds[i], count + 0.3, str(count), 
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
       
       # Highlight the selected threshold
       best_idx = np.abs(np.array(merged_cluster_counts) - target_clusters).argmin()
       plt.axvline(x=thresholds[best_idx], color='firebrick', linestyle='--', alpha=0.7,
                  label=f'Selected threshold: {thresholds[best_idx]}')
       
       # Formatting
       plt.xlabel('Similarity Threshold', fontsize=13)
       plt.ylabel('Number of Merged Clusters', fontsize=13)
       plt.title('Cluster Merging by Similarity Threshold', fontsize=15, fontweight='bold')
       plt.xticks(thresholds)
       plt.yticks(range(0, max(merged_cluster_counts) + 2))
       plt.legend(loc='best')
       
       # Add explanatory note
       plt.figtext(0.5, 0.01, 
                 'Higher threshold = stricter merging criteria = more clusters', 
                 ha='center', fontsize=11, style='italic')
       
       plt.tight_layout(rect=[0, 0.03, 1, 0.97])
       plt.savefig('cluster_merge_thresholds.png', dpi=300)
       
       # Choose threshold that gives us close to 5-8 clusters
       target_clusters = min(8, len(cluster_ids))
       best_threshold_idx = np.abs(np.array(merged_cluster_counts) - target_clusters).argmin()
       best_threshold = thresholds[best_threshold_idx]
       
       print(f"Selected similarity threshold: {best_threshold} (gives {merged_cluster_counts[best_threshold_idx]} clusters)")
       
       # Step 6: Apply the merging
       merged_cluster_labels = fcluster(cluster_linkage, t=best_threshold, criterion='distance')
       
       # Create mapping from original cluster ID to merged cluster ID
       cluster_mapping = {
           orig_id: merged_id 
           for orig_id, merged_id in zip(cluster_ids, merged_cluster_labels)
       }
       
       # Apply mapping to create new cluster labels
       embedding_df['merged_cluster'] = embedding_df['cluster'].map(
           lambda x: cluster_mapping.get(x, -1) if x != -1 else -1
       )
       
       # Get the final cluster counts
       merged_cluster_sizes = embedding_df[embedding_df['merged_cluster'] != -1]['merged_cluster'].value_counts()
       
       print(f"After merging, we have {len(merged_cluster_sizes)} clusters")
       for merged_id, size in merged_cluster_sizes.items():
           original_clusters = [
               orig_id for orig_id, merged_id_mapped in cluster_mapping.items() 
               if merged_id_mapped == merged_id
           ]
           print(f"Merged cluster {merged_id}: {size} documents, from original clusters {original_clusters}")
       
       # Visualization: Plot merged cluster sizes with enhanced styling
       plt.figure(figsize=(14, 8))
       sns.set_style("whitegrid")
       sns.set_context("notebook", font_scale=1.2)
       
       # Generate a diverging color palette based on cluster size
       palette = sns.color_palette("viridis", n_colors=len(merged_cluster_sizes))
       # Sort for visual appeal
       ordered_indices = np.argsort(merged_cluster_sizes.values)[::-1]  # Descending order
       ordered_sizes = merged_cluster_sizes.iloc[ordered_indices]
       ordered_palette = [palette[i] for i in range(len(merged_cluster_sizes))]
       
       # Create barplot with ordered data
       ax = sns.barplot(
           x=ordered_sizes.index, 
           y=ordered_sizes.values,
           palette=ordered_palette,
           edgecolor='black',
           linewidth=1.5,
           alpha=0.8
       )
       
       # Add value labels on top of bars
       for i, (idx, v) in enumerate(zip(ordered_sizes.index, ordered_sizes.values)):
           # Add the value and percentage
           percentage = (v / len(embedding_df)) * 100
           label = f"{v}\n({percentage:.1f}%)"
           ax.text(i, v + 5, label, ha='center', va='bottom', fontsize=10, fontweight='bold')
           
           # Add original cluster IDs that got merged (if multiple)
           original_clusters = [
               orig_id for orig_id, merged_id_mapped in cluster_mapping.items() 
               if merged_id_mapped == idx
           ]
           if len(original_clusters) > 1:
               orig_text = f"from: {', '.join(map(str, original_clusters))}"
               ax.text(i, v/2, orig_text, ha='center', va='center', 
                       fontsize=8, rotation=90, color='white' if v > 100 else 'black')
       
       # Better formatting
       plt.xlabel('Merged Cluster ID', fontsize=13)
       plt.ylabel('Number of Documents', fontsize=13)
       plt.title('Documents per Merged Cluster (Ordered by Size)', fontsize=15, fontweight='bold')
       plt.xticks(rotation=45)
       
       # Add explanatory annotation
       plt.figtext(0.5, 0.01, 
                 f'Total documents in clusters: {sum(merged_cluster_sizes.values)} ({(sum(merged_cluster_sizes.values)/len(embedding_df))*100:.1f}% of corpus)',
                 ha='center', fontsize=11)
       
       plt.tight_layout(rect=[0, 0.03, 1, 0.97])
       plt.savefig('merged_cluster_sizes.png', dpi=300)
       
       # Use merged clusters for analysis
       embedding_df['final_cluster'] = embedding_df['merged_cluster']
   else:
       # Not enough clusters to merge, use original clusters
       print("Using original clusters as final clusters (not enough clusters to merge)")
       embedding_df['final_cluster'] = embedding_df['cluster']
   
   # Get top clusters by size for further analysis
   final_cluster_counts = embedding_df[embedding_df['final_cluster'] != -1]['final_cluster'].value_counts()
   top_clusters = final_cluster_counts.nlargest(5).index.tolist()
   
   print(f"Selected top {len(top_clusters)} clusters for analysis, representing {final_cluster_counts[top_clusters].sum()} documents")
   ```

4. **Extract Cluster Representatives with Advanced Analysis**
   ```python
   from sklearn.metrics import pairwise_distances_argmin_min
   import numpy as np
   
   # Find representative texts and analyze cluster characteristics
   cluster_representatives = {}
   
   for cluster_id in top_clusters:
       # Get all points in this cluster using the final_cluster column
       cluster_mask = embedding_df['final_cluster'] == cluster_id
       cluster_points = reduced_embeddings[cluster_mask]
       cluster_docs = embedding_df[cluster_mask]
       
       # Calculate cluster center
       cluster_center = np.mean(cluster_points, axis=0)
       
       # Calculate intra-cluster distance statistics (measure of coherence)
       from scipy.spatial.distance import pdist
       if len(cluster_points) > 1:  # Need at least 2 points for pairwise distances
           distances = pdist(cluster_points, 'euclidean')
           avg_distance = np.mean(distances)
           max_distance = np.max(distances)
           coherence_score = 1 / (1 + avg_distance)  # Transform to 0-1 scale (higher is more coherent)
       else:
           avg_distance = 0
           max_distance = 0
           coherence_score = 1
       
       # Find top N texts closest to center (medoids)
       # These are most representative of the cluster
       n_medoids = min(5, len(cluster_points))
       distances_to_center = np.linalg.norm(cluster_points - cluster_center, axis=1)
       medoid_indices = np.argsort(distances_to_center)[:n_medoids]
       
       # Get original indices and texts
       original_medoid_indices = cluster_docs.iloc[medoid_indices].index
       medoid_texts = cluster_docs.iloc[medoid_indices]['Text'].tolist()
       
       # Get engagement stats for medoids
       medoid_engagements = cluster_docs.iloc[medoid_indices][['likes_count', 'recasts_count', 'engagement_score']]
       
       # Find high engagement texts in the cluster
       high_engagement_docs = cluster_docs.nlargest(
           min(5, len(cluster_docs)), 
           'engagement_score'
       )
       
       # Get random diverse samples (furthest from center to show cluster diversity)
       if len(cluster_points) > 10:
           # Get points that are further from center
           diverse_indices = np.argsort(distances_to_center)[-10:]
           diverse_samples = cluster_docs.iloc[diverse_indices]['Text'].tolist()
       else:
           diverse_samples = cluster_docs['Text'].tolist()
       
       # Collect temporal information about cluster
       timestamps = pd.to_datetime(cluster_docs['datetime'])
       time_stats = {
           'earliest': timestamps.min().strftime('%Y-%m-%d %H:%M'),
           'latest': timestamps.max().strftime('%Y-%m-%d %H:%M'),
           'timespan_hours': (timestamps.max() - timestamps.min()).total_seconds() / 3600,
           'median_time': timestamps.median().strftime('%Y-%m-%d %H:%M')
       }
       
       # Analyze temporal density (posts per hour)
       hours_span = max(1, (timestamps.max() - timestamps.min()).total_seconds() / 3600)
       posts_per_hour = len(cluster_docs) / hours_span
       
       # Engagement statistics
       engagement_stats = {
           'total_likes': int(cluster_docs['likes_count'].sum()),
           'total_recasts': int(cluster_docs['recasts_count'].sum()),
           'avg_likes': float(cluster_docs['likes_count'].mean()),
           'avg_recasts': float(cluster_docs['recasts_count'].mean()),
           'max_engagement': float(cluster_docs['engagement_score'].max()),
           'median_engagement': float(cluster_docs['engagement_score'].median()),
           'engagement_per_hour': float(cluster_docs['engagement_score'].sum() / hours_span)
       }
       
       # Save all cluster data
       cluster_representatives[cluster_id] = {
           'size': int(final_cluster_counts[cluster_id]),
           'coherence_score': float(coherence_score),
           'medoid_texts': medoid_texts,
           'medoid_engagements': medoid_engagements.to_dict('records'),
           'high_engagement_samples': high_engagement_docs['Text'].tolist(),
           'diverse_samples': diverse_samples,
           'time_stats': time_stats,
           'posts_per_hour': float(posts_per_hour),
           'engagement_stats': engagement_stats
       }
       
       # Print cluster summary
       print(f"\nCluster {cluster_id} Summary:")
       print(f"Size: {cluster_representatives[cluster_id]['size']} documents")
       print(f"Coherence score: {coherence_score:.3f}")
       print(f"Timespan: {time_stats['earliest']} to {time_stats['latest']} ({time_stats['timespan_hours']:.1f} hours)")
       print(f"Activity: {posts_per_hour:.1f} posts/hour")
       print(f"Engagement: {engagement_stats['avg_likes']:.1f} likes/post, {engagement_stats['avg_recasts']:.1f} recasts/post")
       print(f"Most representative text: {medoid_texts[0][:100]}...")
       
   # Save detailed cluster information for further analysis
   with open('embedding_cluster_details.json', 'w') as f:
       # Convert any non-serializable values (like numpy types) to Python native types
       import json
       class NpEncoder(json.JSONEncoder):
           def default(self, obj):
               if isinstance(obj, np.integer):
                   return int(obj)
               if isinstance(obj, np.floating):
                   return float(obj)
               if isinstance(obj, np.ndarray):
                   return obj.tolist()
               return super(NpEncoder, self).default(obj)
               
       json.dump(cluster_representatives, f, indent=2, cls=NpEncoder)
   ```

5. **Gemini Labeling of Clusters with Structured Response**
   ```python
   from enum import Enum
   from typing_extensions import TypedDict
   from google.generativeai import types
   
   # Define TypedDict for structured output
   class EmbeddingTopic(TypedDict):
       topic_name: str  # 5 words max
       explanation: str  # Brief explanation of why trending
       key_terms: list[str]
       key_entities: list[dict]  # List of entities with name and type
       engagement_insight: str
       sentiment: str
   
   # Use Gemini to label each embedding cluster
   embedding_topics = []
   
   # Initialize Gemini
   model = GenerativeModel('gemini-2.0-flash-lite')
   
   for cluster_id, data in cluster_representatives.items():
       # Get all casts in this cluster
       cluster_casts = embedding_df[embedding_df['cluster'] == cluster_id]
       
       # Calculate engagement metrics
       avg_likes = cluster_casts['likes_count'].mean()
       avg_recasts = cluster_casts['recasts_count'].mean()
       total_engagement = cluster_casts['engagement_score'].sum()
       
       # Find most engaged posts in the cluster (top 5)
       top_engaged = cluster_casts.nlargest(5, 'engagement_score')
       
       # Format engagement samples
       engagement_samples = top_engaged.apply(
           lambda row: f"[👍{int(row['likes_count'])}|↗️{int(row['recasts_count'])}]: {row['Text']}", 
           axis=1
       ).tolist()
       
       # Create structured prompt
       prompt = f"""
       I need to identify the specific trending topic being discussed in this cluster of Farcaster social media posts.
       
       REPRESENTATIVE POST:
       {data['center_text']}
       
       MOST ENGAGED POSTS (with likes and recasts shown):
       {' '.join(engagement_samples)}
       
       OTHER SAMPLE POSTS:
       {' '.join(data['samples'][:5])}
       
       ENGAGEMENT METRICS:
       - Average likes: {avg_likes:.2f}
       - Average recasts: {avg_recasts:.2f}
       - Cluster size: {data['size']} posts
       
       Generate your response based on the following Python TypedDict schema:
       
       class Entity(TypedDict):
           name: str
           type: str  # Person, Project, Company, Protocol, etc.
       
       class EmbeddingTopic(TypedDict):
           topic_name: str  # 5 words max
           explanation: str  # Brief explanation of why trending
           key_terms: list[str]  # List of strings
           key_entities: list[Entity]  # List of entities with name and type
           engagement_insight: str  # Brief insight about engagement patterns
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
           print(f"Successfully labeled embedding cluster {cluster_id} as '{topic_data.get('topic_name', 'Unknown')}'")
       except json.JSONDecodeError as e:
           print(f"Error parsing JSON for embedding cluster {cluster_id}: {e}")
           # Fallback if JSON parsing fails
           topic_data = {
               "topic_name": "Error parsing response",
               "explanation": "Could not parse JSON from Gemini response",
               "key_terms": [],
               "key_entities": [],
               "engagement_insight": "Unknown",
               "sentiment": "Unknown"
           }
       
       # Add metrics to results
       embedding_topics.append({
           'cluster_id': cluster_id,
           'size': data['size'],
           'avg_likes': float(avg_likes),
           'avg_recasts': float(avg_recasts),
           'total_engagement': float(total_engagement),
           'topic_data': topic_data
       })
   
   # Save intermediate results
   with open('approach3_results.json', 'w') as f:
       json.dump(embedding_topics, f, indent=2)
   ```