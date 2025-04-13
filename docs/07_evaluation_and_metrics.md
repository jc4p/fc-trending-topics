## Evaluation and Metrics

1. **Topic Coherence and Quality Analysis**
   ```python
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt
   import seaborn as sns
   
   # Load all approach results
   with open('approach1_results.json', 'r') as f:
       approach1_data = json.load(f)
       
   with open('approach2_results.json', 'r') as f:
       approach2_data = json.load(f)
       
   with open('approach3_results.json', 'r') as f:
       approach3_data = json.load(f)
       
   with open('embedding_cluster_details.json', 'r') as f:
       cluster_details = json.load(f)
   
   # Create a comprehensive topic quality analysis
   
   # 1. Extract coherence metrics where available
   coherence_data = []
   
   # For embedding clusters, we calculated a coherence score directly
   for cluster_id, details in cluster_details.items():
       coherence_data.append({
           'approach': 'Embeddings',
           'topic_id': f"Cluster {cluster_id}",
           'topic_name': next((t['topic_data']['topic_name'] for t in approach3_data 
                             if str(t['cluster_id']) == str(cluster_id)), "Unknown"),
           'coherence_score': details['coherence_score'],
           'size': details['size']
       })
   
   # For LDA, we need to approximate using silhouette score
   # (This is a placeholder - the actual code would use the silhouette score calculated earlier)
   if 'silhouette_avg' in locals():
       for i, topic in enumerate(approach2_data):
           coherence_data.append({
               'approach': 'LDA + K-Means',
               'topic_id': f"Cluster {topic['cluster_id']}",
               'topic_name': topic['topic_data']['topic_name'],
               'coherence_score': silhouette_avg,  # This is approximate
               'size': topic['size']
           })
   
   # For Direct LLM, we don't have a direct coherence metric
   # We could use a proxy based on the confidence or specificity of the LLM's answers
   for i, topic in enumerate(approach1_data.get('topics', [])):
       # Approximate coherence based on specificity of terms
       term_specificity = len(topic.get('key_terms', []))
       proxy_score = min(0.9, 0.5 + (term_specificity / 20))  # Scale to 0.5-0.9 range
       
       coherence_data.append({
           'approach': 'Direct LLM',
           'topic_id': f"Topic {i+1}",
           'topic_name': topic.get('name', 'Unknown'),
           'coherence_score': proxy_score,  # This is a very rough proxy
           'size': None  # Direct LLM doesn't have size data
       })
   
   # Create a DataFrame for visualization
   coherence_df = pd.DataFrame(coherence_data)
   
   # Plot coherence scores by approach
   plt.figure(figsize=(14, 8))
   sns.set_style("whitegrid")
   sns.set_context("notebook", font_scale=1.2)
   
   # Create box plot with individual points
   ax = sns.boxplot(
       x='approach', 
       y='coherence_score', 
       data=coherence_df,
       palette='viridis'
   )
   
   # Add individual points
   sns.stripplot(
       x='approach', 
       y='coherence_score', 
       data=coherence_df,
       color='black',
       size=8,
       alpha=0.6,
       jitter=True
   )
   
   # Add topic names as annotations
   for i, row in coherence_df.iterrows():
       ax.annotate(
           row['topic_name'], 
           (coherence_df['approach'].unique().tolist().index(row['approach']), row['coherence_score']),
           xytext=(5, 0),
           textcoords='offset points',
           fontsize=8,
           alpha=0.7
       )
   
   plt.xlabel('Approach', fontsize=13)
   plt.ylabel('Topic Coherence Score', fontsize=13)
   plt.title('Topic Coherence Comparison Across Approaches', fontsize=15, fontweight='bold')
   plt.tight_layout()
   plt.savefig('topic_coherence_comparison.png', dpi=300)
   
   # 2. Create a comprehensive analysis of topic quality metrics
   # Ideally we would have human evaluation metrics, but we'll use proxies:
   # - Coherence: How internally consistent the topic is
   # - Distinctiveness: How different the topic is from others
   # - Coverage: What percentage of the dataset the topic covers
   # - Engagement: How much user engagement the topic receives
   
   # Build a dataframe with all available metrics
   quality_metrics = []
   
   # Process embedding approach
   for topic in approach3_data:
       cluster_id = str(topic['cluster_id'])
       if cluster_id in cluster_details:
           details = cluster_details[cluster_id]
           engagement = details['engagement_stats']['avg_likes'] + (3 * details['engagement_stats']['avg_recasts'])
           
           quality_metrics.append({
               'approach': 'Embeddings',
               'topic_name': topic['topic_data']['topic_name'],
               'coherence': details['coherence_score'],
               'distinctiveness': 0.8,  # Placeholder - would calculate from similarity matrix
               'coverage': details['size'] / len(embedding_df) * 100,
               'engagement': engagement,
               'posts_per_hour': details['posts_per_hour'] 
           })
   
   # Process LDA approach (with less detailed metrics)
   for topic in approach2_data:
       quality_metrics.append({
           'approach': 'LDA + K-Means',
           'topic_name': topic['topic_data']['topic_name'],
           'coherence': 0.7,  # Placeholder 
           'distinctiveness': 0.75,  # Placeholder
           'coverage': topic['size'] / len(recent_df) * 100,
           'engagement': topic['avg_engagement']
       })
       
   # Process Direct LLM approach (with estimated metrics)
   for topic in approach1_data.get('topics', []):
       # Extract percentage from string like "15%"
       coverage = float(topic.get('estimated_percentage', '0%').replace('%', '')) 
       
       quality_metrics.append({
           'approach': 'Direct LLM',
           'topic_name': topic.get('name', 'Unknown'),
           'coherence': 0.65,  # Placeholder
           'distinctiveness': 0.7,  # Placeholder
           'coverage': coverage,
           'engagement': None  # No direct engagement data
       })
   
   # Create radar chart for comparing topic quality across approaches
   from matplotlib.patches import Circle, RegularPolygon
   from matplotlib.path import Path
   from matplotlib.projections.polar import PolarAxes
   from matplotlib.projections import register_projection
   from matplotlib.spines import Spine
   from matplotlib.transforms import Affine2D
   
   def radar_factory(num_vars, frame='circle'):
       """Create a radar chart with `num_vars` axes."""
       # Calculate evenly-spaced axis angles
       theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
       
       class RadarAxes(PolarAxes):
           name = 'radar'
           
           def __init__(self, *args, **kwargs):
               super().__init__(*args, **kwargs)
               self.set_theta_zero_location('N')
               
           def fill(self, *args, closed=True, **kwargs):
               """Override fill so that line is closed by default"""
               return super().fill(closed=closed, *args, **kwargs)
               
           def plot(self, *args, **kwargs):
               """Override plot so that line is closed by default"""
               lines = super().plot(*args, **kwargs)
               for line in lines:
                   self._close_line(line)
                   
           def _close_line(self, line):
               x, y = line.get_data()
               # FIXME: markers at x[0], y[0] get doubled-up
               if x[0] != x[-1]:
                   x = np.concatenate((x, [x[0]]))
                   y = np.concatenate((y, [y[0]]))
                   line.set_data(x, y)
                   
           def set_varlabels(self, labels):
               self.set_thetagrids(np.degrees(theta), labels)
               
           def _gen_axes_patch(self):
               if frame == 'circle':
                   return Circle((0.5, 0.5), 0.5)
               elif frame == 'polygon':
                   return RegularPolygon((0.5, 0.5), num_vars,
                                         radius=.5, edgecolor="k")
               else:
                   raise ValueError("unknown value for 'frame': %s" % frame)
                   
           def draw(self, renderer):
               """ Draw. If frame is polygon, make gridlines polygon-shaped """
               if frame == 'polygon':
                   gridlines = self.yaxis.get_gridlines()
                   for gl in gridlines:
                       gl.get_path()._interpolation_steps = num_vars
               super().draw(renderer)
               
           def _gen_axes_spines(self):
               if frame == 'circle':
                   return super()._gen_axes_spines()
               elif frame == 'polygon':
                   # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                   spine = Spine(axes=self,
                                 spine_type='circle',
                                 path=Path.unit_regular_polygon(num_vars))
                   # unit_regular_polygon returns a polygon of radius 1 centered at
                   # (0,0) but we want a polygon of radius 0.5 centered at (0.5,0.5)
                   # in axes coordinates.
                   spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                       + self.transAxes)
                   return {'polar': spine}
               else:
                   raise ValueError("unknown value for 'frame': %s" % frame)
                   
       register_projection(RadarAxes)
       return theta
   
   # Group metrics by approach for radar chart
   approach_metrics = {}
   metric_names = ['coherence', 'distinctiveness', 'coverage', 'engagement']
   
   quality_df = pd.DataFrame(quality_metrics)
   
   # Normalize each metric to 0-1 scale
   for metric in metric_names:
       if metric in quality_df.columns:
           # Skip metrics with missing values
           if quality_df[metric].isnull().any():
               continue
               
           min_val = quality_df[metric].min()
           max_val = quality_df[metric].max()
           
           if max_val > min_val:
               quality_df[f'{metric}_norm'] = (quality_df[metric] - min_val) / (max_val - min_val)
           else:
               quality_df[f'{metric}_norm'] = 0.5  # Default if all values are the same
   
   # Average by approach
   for approach in quality_df['approach'].unique():
       approach_data = quality_df[quality_df['approach'] == approach]
       approach_metrics[approach] = []
       
       for metric in metric_names:
           norm_metric = f'{metric}_norm'
           if norm_metric in approach_data.columns and not approach_data[norm_metric].isnull().all():
               avg_val = approach_data[norm_metric].mean()
               approach_metrics[approach].append(avg_val)
           else:
               approach_metrics[approach].append(0)  # Default for missing metrics
   
   # Create the radar chart
   plt.figure(figsize=(12, 8))
   theta = radar_factory(len(metric_names), frame='polygon')
   
   ax = plt.subplot(111, projection='radar')
   
   colors = sns.color_palette('viridis', n_colors=len(approach_metrics))
   
   for i, (approach, metrics) in enumerate(approach_metrics.items()):
       ax.plot(theta, metrics, color=colors[i], label=approach)
       ax.fill(theta, metrics, alpha=0.1, color=colors[i])
   
   ax.set_varlabels(metric_names)
   plt.legend(loc='upper right')
   plt.title('Topic Quality Metrics by Approach', fontsize=15, fontweight='bold')
   
   plt.tight_layout()
   plt.savefig('topic_quality_radar.png', dpi=300)
   ```

2. **Performance Metrics**
   ```python
   # Document runtime, memory usage, etc.
   ```

3. **Result Overlaps**
   ```python
   # Analyze how much the different approaches agree on trending topics
   ```