# Import sentence-transformers
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
import hdbscan
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist, cdist
import numpy as np
import pandas as pd
import json
import os
import time
import difflib
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai import GenerativeModel
from google.generativeai import types
from typing_extensions import TypedDict

# Load from .env file in current directory 
# Get the absolute path to find it regardless of where the script is run from
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
env_path = os.path.join(project_dir, '.env')

print(f"Looking for .env file at: {env_path}")
if os.path.exists(env_path):
    print(f".env file found, loading environment variables")
    load_dotenv(env_path)
else:
    print(f".env file not found at {env_path}")
    # Try root directory as fallback
    load_dotenv()

# Configure Gemini API
api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please check your .env file.")
genai.configure(api_key=api_key)

class EmbeddingTopic(TypedDict):
    topic_name: str  # 5 words max
    explanation: str  # Brief explanation of why trending
    key_terms: list[str]  # List of 5-10 most important terms
    key_entities: list[dict]  # List of entities with name and type
    engagement_insight: str  # Brief insight about engagement patterns
    sentiment: str  # Positive, Neutral, Negative, Mixed
    language: str  # Primary language of the cluster (English, Vietnamese, etc.)

class Entity(TypedDict):
    name: str
    type: str

def filter_semantic_duplicates(df, embedding_matrix, similarity_threshold=0.85, sampled_indices=None, embedding_index_mapping=None):
    """
    Filter out texts that are semantically too similar using embedding similarity
    
    Args:
        df: DataFrame containing texts to filter
        embedding_matrix: Matrix of embeddings
        similarity_threshold: Threshold for similarity (higher = more filtering)
        sampled_indices: Optional set of indices that were sampled for embeddings
        embedding_index_mapping: Optional mapping from original indices to embedding indices
        
    Returns:
        DataFrame with semantically unique texts
    """
    if len(df) <= 1:
        return df
    
    # Use simple text similarity if we don't have embeddings for all indices
    if sampled_indices is not None:
        # Check if any indices are not in the sample
        missing_indices = [idx for idx in df.index if idx not in sampled_indices]
        if missing_indices:
            # Fall back to text-based similarity
            print(f"Using text-based similarity as {len(missing_indices)} indices don't have embeddings")
            
            # Get all texts
            texts = df['Text'].tolist()
            text_indices = df.index.tolist()
            
            # Track which indices to keep (start with the first one)
            kept_indices = [text_indices[0]]
            kept_texts = [texts[0]]
            
            # Compare each text against already kept texts
            for i in range(1, len(texts)):
                current_text = texts[i]
                
                # Calculate text similarity with kept texts
                # Use SequenceMatcher for string similarity
                similarities = []
                for kept_text in kept_texts:
                    sim = difflib.SequenceMatcher(None, current_text, kept_text).ratio()
                    similarities.append(sim)
                
                # If not too similar to any kept text, keep this one too
                if not similarities or max(similarities) < similarity_threshold:
                    kept_indices.append(text_indices[i])
                    kept_texts.append(current_text)
            
            # Return filtered dataframe with diverse texts
            return df.loc[kept_indices]
    
    # If we have embeddings for all indices, use embedding similarity
    try:
        # Get indices from the dataframe
        indices = df.index.tolist()
        
        # Extract the embeddings for these texts
        # If we have an index mapping, use it to get the correct embedding indices
        if embedding_index_mapping:
            # Use the mapping to get the correct indices for the embedding matrix
            embedding_indices = [embedding_index_mapping.get(idx, 0) for idx in indices]
            text_embeddings = embedding_matrix[embedding_indices]
        else:
            # Direct indexing (may cause errors if indices are out of bounds)
            try:
                text_embeddings = embedding_matrix[indices]
            except IndexError:
                print("IndexError: Falling back to text similarity")
                # Fall back to text similarity
                # Since embedding_matrix is None, we'll use text similarity
                texts = df['Text'].tolist()
                text_indices = df.index.tolist()
            
                # Track which indices to keep (start with the first one)
                kept_indices = [text_indices[0]]
                kept_texts = [texts[0]]
                
                # Compare each text against already kept texts
                for i in range(1, len(texts)):
                    current_text = texts[i]
                    
                    # Calculate text similarity with kept texts
                    similarities = []
                    for kept_text in kept_texts:
                        sim = difflib.SequenceMatcher(None, current_text, kept_text).ratio()
                        similarities.append(sim)
                    
                    # If not too similar to any kept text, keep this one too
                    if not similarities or max(similarities) < similarity_threshold:
                        kept_indices.append(text_indices[i])
                        kept_texts.append(current_text)
                
                # Return filtered dataframe with diverse texts
                return df.loc[kept_indices]
        
        # Track which indices to keep (start with the highest engagement)
        kept_indices = [indices[0]]
        kept_embeddings = [text_embeddings[0]]
        
        # Compare each text against already kept texts
        for i in range(1, len(indices)):
            current_embedding = text_embeddings[i]
            
            # Calculate cosine similarity with kept embeddings
            similarities = []
            for kept_embedding in kept_embeddings:
                # Normalize for cosine similarity
                sim = np.dot(current_embedding, kept_embedding) / (
                    np.linalg.norm(current_embedding) * np.linalg.norm(kept_embedding)
                )
                similarities.append(sim)
            
            # If not too similar to any kept text, keep this one too
            if not similarities or max(similarities) < similarity_threshold:
                kept_indices.append(indices[i])
                kept_embeddings.append(current_embedding)
        
        # Return filtered dataframe with diverse texts
        return df.loc[kept_indices]
    except (IndexError, ValueError) as e:
        # If embedding access fails, fall back to text-based similarity
        print(f"Warning: Embedding-based filtering failed ({e}), falling back to text similarity")
        
        # Get all texts
        texts = df['Text'].tolist()
        text_indices = df.index.tolist()
        
        # Track which indices to keep (start with the first one)
        kept_indices = [text_indices[0]]
        kept_texts = [texts[0]]
        
        # Compare each text against already kept texts
        for i in range(1, len(texts)):
            current_text = texts[i]
            
            # Calculate text similarity with kept texts
            # Use SequenceMatcher for string similarity
            similarities = []
            for kept_text in kept_texts:
                sim = difflib.SequenceMatcher(None, current_text, kept_text).ratio()
                similarities.append(sim)
            
            # If not too similar to any kept text, keep this one too
            if not similarities or max(similarities) < similarity_threshold:
                kept_indices.append(text_indices[i])
                kept_texts.append(current_text)
        
        # Return filtered dataframe with diverse texts
        return df.loc[kept_indices]

def embeddings_clustering(recent_df):
    """
    Approach 3: Embeddings + Clustering
    
    This approach uses text embeddings with advanced clustering 
    to identify trending topics in the dataset.
    
    Args:
        recent_df: DataFrame with cleaned posts
        
    Returns:
        dict: Structured trending topics result
    """
    print("Starting Embeddings + Clustering approach...")
    start_time = time.time()
    
    # Filter out posts with 0 likes and 0 recasts since we're focusing on viral content
    initial_count = len(recent_df)
    
    # Handle potential NaN values in likes_count and recasts_count columns
    recent_df['likes_count'] = recent_df['likes_count'].fillna(0).astype(int)
    recent_df['recasts_count'] = recent_df['recasts_count'].fillna(0).astype(int)
    
    # Keep only posts with at least some engagement (likes or recasts > 0)
    recent_df = recent_df[(recent_df['likes_count'] > 0) | (recent_df['recasts_count'] > 0)]
    
    # Enhance with conversation metrics if ParentCastId is available
    # This will help prioritize posts that generate conversation in clustering
    has_parent_info = 'ParentCastId' in recent_df.columns
    
    if has_parent_info:
        print("Adding conversation metrics to enhance embedding clustering...")
        
        # Identify top-level posts vs replies
        recent_df['is_reply'] = ~(recent_df['ParentCastId'].isnull() | (recent_df['ParentCastId'] == ''))
        top_level_posts = recent_df[~recent_df['is_reply']]
        reply_posts = recent_df[recent_df['is_reply']]
        
        # Calculate conversation metrics
        # For each top level post, count how many replies it received
        reply_counts = {}
        unique_repliers = {}
        
        # Track parent hashes for replies
        for _, row in reply_posts.iterrows():
            # Extract parent hash - it should be in the format "FID:HASH"
            parent_id = row['ParentCastId']
            if ':' in parent_id:
                parent_hash = parent_id.split(':', 1)[1]
            else:
                parent_hash = parent_id
                
            # Update counts
            if parent_hash in reply_counts:
                reply_counts[parent_hash] += 1
                unique_repliers[parent_hash].add(row['Fid'])
            else:
                reply_counts[parent_hash] = 1
                unique_repliers[parent_hash] = {row['Fid']}
        
        # Create a copy of the dataframe to prevent SettingWithCopyWarning
        recent_df = recent_df.copy()
        
        # Add conversation metrics to top-level posts using .loc to avoid SettingWithCopyWarning
        recent_df.loc[:, 'reply_count'] = recent_df['Hash'].map(lambda h: reply_counts.get(h, 0))
        recent_df.loc[:, 'unique_repliers'] = recent_df['Hash'].map(lambda h: len(unique_repliers.get(h, set())))
        
        # Create an enhanced engagement score that includes conversation metrics
        # Original engagement: likes + 3*recasts
        # Add conversation component: reply_count*5 + unique_repliers*10
        recent_df.loc[:, 'enhanced_engagement'] = (
            recent_df['likes_count'] + 
            3 * recent_df['recasts_count'] + 
            5 * recent_df['reply_count'] + 
            10 * recent_df['unique_repliers']
        )
        
        # Use the enhanced engagement for clustering
        recent_df.loc[:, 'engagement_score'] = recent_df['enhanced_engagement']
        
        print(f"Added conversation metrics to {len(top_level_posts)} top-level posts")
        print(f"Average replies per post with replies: {sum(reply_counts.values()) / len(reply_counts) if reply_counts else 0:.1f}")
        print(f"Posts with at least one reply: {len(reply_counts)}")
        print(f"Posts with 5+ replies: {sum(1 for c in reply_counts.values() if c >= 5)}")
        print(f"Posts with 10+ unique repliers: {sum(1 for r in unique_repliers.values() if len(r) >= 10)}")
    
    filtered_count = len(recent_df)
    print(f"Filtered out {initial_count - filtered_count} posts with 0 likes and 0 recasts.")
    print(f"Proceeding with {filtered_count} posts that have at least some engagement.")
    
    # CUDA setup and diagnostics
    if torch.cuda.is_available():
        # Force initialization
        torch.cuda.init()
        torch.cuda.empty_cache()
        
        # Log GPU information
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        
        print(f"CUDA is available: {device_count} device(s)")
        print(f"Using device: {current_device} - {device_name}")
        print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"Memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
    else:
        print("CUDA is not available, using CPU")
    
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    os.makedirs('output/figures', exist_ok=True)
    
    # Step 1: Generate Embeddings
    print("Generating embeddings...")
    
    # Initialize transformer model with CUDA if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device} for embeddings generation")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    
    # Significantly larger sample for embeddings to ensure better representation
    # We'll use almost the entire dataset if possible to minimize sampling bias
    embedding_sample_size = min(100000, len(recent_df))  # Doubled from 50000 to 100000
    # Important: Use seed to ensure consistent sampling
    embedding_df = recent_df.sample(embedding_sample_size, random_state=42)
    
    # Store the indices that were sampled for later reference
    sampled_indices = set(embedding_df.index.tolist())
    
    # Create a mapping from original indices to the reduced embeddings array indices
    # This is crucial for correctly accessing embeddings for arbitrary dataframes
    embedding_index_mapping = {idx: i for i, idx in enumerate(embedding_df.index)}
    
    # Generate embeddings with appropriate batch size for the device
    batch_size = 128 if device == 'cuda' else 32  # Larger batch size for GPU
    embeddings = embedding_model.encode(
        embedding_df['cleaned_text'].tolist(), 
        show_progress_bar=True,
        batch_size=batch_size,
        convert_to_numpy=True
    )
    
    print(f"Generated {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")
    
    # Step 2: Dimensionality Reduction (Optional)
    print("Reducing embedding dimensions...")
    
    # Try to use GPU-accelerated PCA if available
    try:
        if torch.cuda.is_available():
            from cuml.decomposition import PCA as cuPCA
            print("Using GPU-accelerated PCA from RAPIDS cuML")
            pca = cuPCA(n_components=50)
            reduced_embeddings = pca.fit_transform(embeddings)
            # Convert from cuDF series to numpy if needed
            if hasattr(reduced_embeddings, 'to_numpy'):
                reduced_embeddings = reduced_embeddings.to_numpy()
        else:
            raise ImportError("CUDA not available for cuML")
    except ImportError:
        # Fall back to scikit-learn's PCA
        print("Using scikit-learn's PCA on CPU")
        pca = PCA(n_components=50)
        reduced_embeddings = pca.fit_transform(embeddings)
    
    print(f"Reduced dimensions from {embeddings.shape[1]} to {reduced_embeddings.shape[1]}")
    print(f"Explained variance ratio: {sum(pca.explained_variance_ratio_):.3f}")
    
    # Step 3: Advanced Clustering with Similarity Analysis
    print("Performing clustering with GPU acceleration if available...")
    
    # Try cuML KMeans first (GPU-accelerated)
    try:
        if torch.cuda.is_available():
            from cuml.cluster import KMeans as cuKMeans
            
            print("Using RAPIDS cuML KMeans on GPU (fastest option)")
            
            # Convert embeddings to correct format (float32)
            embeddings_for_clustering = reduced_embeddings.astype(np.float32)
            
            # Determine number of clusters based on dataset size
            # Use a higher minimum and smaller divisor to create more balanced clusters
            # Increase the divisor significantly to create more clusters initially
            n_clusters = max(40, min(100, len(embeddings_for_clustering) // 400))
            
            # Create and configure the KMeans object
            kmeans = cuKMeans(
                n_clusters=n_clusters,
                max_iter=300,
                tol=1e-4,
                verbose=2,
                random_state=42,
                n_init=10
            )
            
            # Train KMeans
            print(f"Running k-means on GPU with {n_clusters} clusters...")
            cluster_labels = kmeans.fit_predict(embeddings_for_clustering)
            
            # Set the clusters
            embedding_df['cluster'] = cluster_labels
            
            # Convert from cuDF series to numpy if needed
            if hasattr(embedding_df['cluster'], 'to_numpy'):
                embedding_df['cluster'] = embedding_df['cluster'].to_numpy()
            
            # Print some stats about GPU usage after clustering
            print(f"GPU memory after cuML KMeans: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f"GPU max memory: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
            
            # Make sure we have clusters numbered from 0 to n_clusters-1 (no -1 noise points)
            # Renumber any -1 values to new cluster numbers
            if -1 in np.unique(embedding_df['cluster']):
                print("Renumbering -1 noise points to new clusters")
                embedding_df.loc[embedding_df['cluster'] == -1, 'cluster'] = np.arange(
                    np.max(embedding_df['cluster']) + 1,
                    np.max(embedding_df['cluster']) + 1 + np.sum(embedding_df['cluster'] == -1)
                )
            
            raise Exception("Bypassing other methods - cuML KMeans completed successfully")
    except Exception as e:
        print(f"cuML KMeans clustering not available or failed: {e}")
        
        # Try to use RAPIDS cuML for GPU-accelerated clustering
        try:
            # Import RAPIDS libraries
            from cuml.cluster import DBSCAN as cuDBSCAN
            
            # Check if CUDA is available
            if torch.cuda.is_available():
                print("Using GPU-accelerated DBSCAN from RAPIDS cuML")
                # DBSCAN parameters need to be tuned differently than HDBSCAN
                # eps is the most critical parameter - distance threshold for forming clusters
                clusterer = cuDBSCAN(
                    eps=0.5,  # Adjust based on your embedding distances
                    min_samples=5,
                    metric='euclidean'
                )
                embedding_df['cluster'] = clusterer.fit_predict(reduced_embeddings)
                # Convert from cuDF series to numpy if needed
                if hasattr(embedding_df['cluster'], 'to_numpy'):
                    embedding_df['cluster'] = embedding_df['cluster'].to_numpy()
            else:
                raise ImportError("CUDA not available for cuML")
        except (ImportError, Exception) as e:
            print(f"RAPIDS cuML not available or CUDA not detected: {e}")
            
            # Try Scikit-learn's KMeans with multiple cores
            try:
                print("Trying scikit-learn's KMeans with multiple cores")
                from sklearn.cluster import KMeans
                
                # Determine n_clusters based on dataset size
                # Use a higher minimum and smaller divisor to create more balanced clusters
                # Increase the divisor significantly to create more clusters initially
                n_clusters = max(40, min(100, len(reduced_embeddings) // 400))
                
                kmeans = KMeans(
                    n_clusters=n_clusters,
                    n_init=10,
                    max_iter=300,
                    tol=1e-4,
                    verbose=1,
                    random_state=42,
                    n_jobs=-1  # Use all cores
                )
                
                # Perform clustering
                embedding_df['cluster'] = kmeans.fit_predict(reduced_embeddings)
                
                raise Exception("KMeans completed successfully")
            except Exception as e:
                print(f"KMeans failed: {e}")
                print("Falling back to CPU version of HDBSCAN")
                
                # Try to use HDBSCAN with faster backend
                try:
                    # Try using FAISS as backend for nearest neighbors
                    print("Attempting to use HDBSCAN with optimized backend")
                    clusterer = hdbscan.HDBSCAN(
                        min_cluster_size=50,
                        min_samples=5,
                        metric='euclidean',
                        cluster_selection_method='eom',
                        algorithm='best',  # Let it choose best algorithm
                        core_dist_n_jobs=-1,  # Use all CPU cores
                        approx_min_span_tree=True  # Use approximate MST for speed
                    )
                    embedding_df['cluster'] = clusterer.fit_predict(reduced_embeddings)
                except Exception as e:
                    print(f"Optimized HDBSCAN failed: {e}")
                    print("Using standard HDBSCAN configuration")
                    # Fall back to standard HDBSCAN
                    clusterer = hdbscan.HDBSCAN(
                        min_cluster_size=50,
                        min_samples=5,
                        metric='euclidean',
                        cluster_selection_method='eom'
                    )
                    embedding_df['cluster'] = clusterer.fit_predict(reduced_embeddings)
    
    # Filter out noise (-1 cluster)
    valid_clusters = embedding_df[embedding_df['cluster'] != -1]
    print(f"Initial clustering found {embedding_df['cluster'].nunique() - 1} clusters")
    print(f"Retained {len(valid_clusters)} documents in clusters ({len(valid_clusters)/len(embedding_df)*100:.1f}%)")
    
    # Visualization: Plot cluster sizes with enhanced styling
    cluster_sizes = valid_clusters['cluster'].value_counts()
    plt.figure(figsize=(14, 7))
    sns.set_theme(style="whitegrid", font_scale=1.1)
    
    # Create a color palette that goes from light to dark based on cluster size
    palette = sns.color_palette("viridis", n_colors=len(cluster_sizes))
    # Sort palette by cluster size
    sorted_indices = np.argsort(cluster_sizes.values)
    sorted_palette = [palette[i] for i in sorted_indices]
    
    ax = sns.barplot(
        x=cluster_sizes.index,
        y=cluster_sizes.values,
        hue=cluster_sizes.index,  # Fix FutureWarning by using hue
        palette=sorted_palette,
        order=cluster_sizes.index[sorted_indices],  # Sort by size
        legend=False  # Don't show legend for hue
    )
    
    # Add count labels on top of bars
    for i, v in enumerate(sorted(cluster_sizes.values)):
        ax.text(i, v + 5, str(v), ha='center', fontsize=9)
        
    plt.xlabel('Cluster ID', fontsize=12)
    plt.ylabel('Number of Documents', fontsize=12)
    plt.title('Documents per Initial Cluster (Sorted by Size)', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('output/figures/initial_cluster_sizes.png', dpi=300)
    
    # Only proceed with similarity analysis if we have enough clusters
    if cluster_sizes.shape[0] > 5:
        print("Analyzing cluster similarity...")
        # Step 2: Calculate cluster centers
        cluster_centers = {}
        for cluster_id in cluster_sizes.index:
            cluster_embeddings = reduced_embeddings[embedding_df['cluster'] == cluster_id]
            cluster_centers[cluster_id] = np.mean(cluster_embeddings, axis=0)
        
        # Step 3: Calculate similarity between clusters
        cluster_ids = list(cluster_centers.keys())
        center_vectors = np.array([cluster_centers[cid] for cid in cluster_ids])
        
        # Try to use GPU for distance calculation
        try:
            if torch.cuda.is_available():
                print("Using GPU-accelerated distance calculation with PyTorch")
                
                # Convert to torch tensor and move to GPU
                centers_gpu = torch.tensor(center_vectors, dtype=torch.float32, device='cuda')
                
                # Normalize vectors for cosine similarity
                centers_normalized = torch.nn.functional.normalize(centers_gpu, p=2, dim=1)
                
                # Compute cosine similarity matrix: sim(A,B) = A·B/(|A|·|B|)
                # For normalized vectors, this is just the dot product
                similarity_matrix = torch.mm(centers_normalized, centers_normalized.t())
                
                # Convert similarity to distance: dist = 1 - sim
                cluster_distances = 1.0 - similarity_matrix.cpu().numpy()
                
                # Release GPU memory
                del centers_gpu, centers_normalized, similarity_matrix
                torch.cuda.empty_cache()
                
                print(f"GPU memory after similarity calculation: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            else:
                raise ImportError("CUDA not available")
        except (ImportError, Exception) as e:
            # Try with cuML's implementation
            try:
                print(f"PyTorch method failed, trying cuML: {e}")
                from cuml.metrics import pairwise_distances as cu_distances
                
                # Calculate distances on GPU
                cluster_distances = cu_distances(
                    center_vectors.astype(np.float32),
                    metric='cosine'
                )
                
                # Move back to CPU if needed
                if hasattr(cluster_distances, 'get'):
                    # For CuPy arrays
                    cluster_distances = cluster_distances.get()
                elif hasattr(cluster_distances, 'to_numpy'):
                    # For cuDF/cuML data types
                    cluster_distances = cluster_distances.to_numpy()
            except Exception as e2:
                print(f"Using scikit-learn's cosine_distances (CPU): {e2}")
                from sklearn.metrics.pairwise import cosine_distances
                cluster_distances = cosine_distances(center_vectors)
        
        # Create a DataFrame for better visualization
        cluster_similarity_df = pd.DataFrame(
            1 - cluster_distances,  # Convert distance to similarity
            index=cluster_ids,
            columns=cluster_ids
        )
        
        # Visualize cluster similarity with enhanced styling
        plt.figure(figsize=(16, 14))
        sns.set_theme(style="whitegrid", font_scale=1.2)
        
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
        plt.savefig('output/figures/cluster_similarity_matrix.png', dpi=300)
        
        # Step 4: Hierarchical clustering to merge similar clusters
        # Use complete linkage to ensure all points in merged clusters are similar
        cluster_linkage = linkage(cluster_distances, method='complete')
        
        # Plot dendrogram to visualize cluster merging with enhanced styling
        plt.figure(figsize=(16, 10))
        
        # Create a custom color palette
        sns.set_palette("viridis", n_colors=len(cluster_ids))
        
        # Plot with more styling options (updated for newer matplotlib)
        with plt.style.context('seaborn-v0_8-white'):
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
        plt.savefig('output/figures/embedding_cluster_dendrogram.png', dpi=300)
        
        # Step 5: Determine optimal similarity threshold
        # Try higher thresholds to avoid excessive merging and create more balanced clusters
        # Higher thresholds = less merging = more clusters
        thresholds = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
        merged_cluster_counts = []
        
        # Target number of clusters - increase significantly to avoid having just a few large clusters
        # Aim for more clusters (15-20) to avoid one dominant cluster
        target_clusters = min(20, max(15, len(cluster_ids) // 3))
        
        for threshold in thresholds:
            merged_clusters = fcluster(cluster_linkage, t=threshold, criterion='distance')
            merged_cluster_counts.append(len(np.unique(merged_clusters)))
        
        plt.figure(figsize=(12, 7))
        sns.set_theme(style="whitegrid", context="notebook", font_scale=1.2)
        
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
        plt.savefig('output/figures/cluster_merge_thresholds.png', dpi=300)
        
        # Choose threshold that gives us close to 5-8 clusters
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
        sns.set_theme(style="whitegrid", context="notebook", font_scale=1.2)
        
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
            hue=ordered_sizes.index,  # Fix FutureWarning by using hue
            palette=ordered_palette,
            edgecolor='black',
            linewidth=1.5,
            alpha=0.8,
            legend=False  # Don't show legend for hue
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
        plt.savefig('output/figures/merged_cluster_sizes.png', dpi=300)
        
        # Use merged clusters for analysis
        embedding_df['final_cluster'] = embedding_df['merged_cluster']
    else:
        # Not enough clusters to merge, use original clusters
        print("Using original clusters as final clusters (not enough clusters to merge)")
        embedding_df['final_cluster'] = embedding_df['cluster']
    
    # Get counts for all final clusters
    final_cluster_counts = embedding_df[embedding_df['final_cluster'] != -1]['final_cluster'].value_counts()
    
    # Calculate engagement metrics for each cluster
    print("Calculating engagement metrics for cluster selection...")
    cluster_engagement = {}
    
    for cluster_id in final_cluster_counts.index:
        # Get documents in this cluster
        cluster_docs = embedding_df[embedding_df['final_cluster'] == cluster_id]
        
        # Calculate engagement metrics
        avg_likes = cluster_docs['likes_count'].mean()
        avg_recasts = cluster_docs['recasts_count'].mean()
        total_engagement = cluster_docs['engagement_score'].sum()
        size = len(cluster_docs)
        
        # Add conversation metrics if available
        if has_parent_info and 'reply_count' in cluster_docs.columns:
            avg_replies = cluster_docs['reply_count'].mean()
            avg_unique_repliers = cluster_docs['unique_repliers'].mean()
            posts_with_replies = (cluster_docs['reply_count'] > 0).sum()
            reply_percentage = (posts_with_replies / size) * 100 if size > 0 else 0
            
            # Add these to our metrics - use .loc accessor to avoid SettingWithCopyWarning
            # Calculate conversation score without modifying the dataframe
            conversation_scores = cluster_docs['reply_count'] * 10 + cluster_docs['unique_repliers'] * 20
            conversation_score = conversation_scores.mean()
            
            # Add to total engagement with a weight
            conversation_weight = 0.3  # 30% weight to conversation
            total_engagement = (1 - conversation_weight) * total_engagement + conversation_weight * conversation_score * size
        
        # Calculate time-based metrics to determine if the cluster is trending
        if 'datetime' in cluster_docs.columns:
            # Get the timestamps and sort them
            timestamps = pd.to_datetime(cluster_docs['datetime']).sort_values()
            
            # If we have enough data points
            if len(timestamps) >= 3:
                # Calculate posts per hour in the first third vs last third of the time period
                time_range = (timestamps.max() - timestamps.min()).total_seconds() / 3600  # in hours
                if time_range > 0:
                    # Split into three equal time periods
                    time_split1 = timestamps.min() + (timestamps.max() - timestamps.min()) / 3
                    time_split2 = timestamps.min() + 2 * (timestamps.max() - timestamps.min()) / 3
                    
                    # Count posts in each period
                    early_posts = sum(timestamps < time_split1)
                    middle_posts = sum((timestamps >= time_split1) & (timestamps < time_split2))
                    recent_posts = sum(timestamps >= time_split2)
                    
                    # Get first and last third durations
                    early_period = (time_split1 - timestamps.min()).total_seconds() / 3600
                    recent_period = (timestamps.max() - time_split2).total_seconds() / 3600
                    
                    # Calculate posts per hour for each period (avoid division by zero)
                    early_rate = early_posts / max(0.1, early_period)
                    recent_rate = recent_posts / max(0.1, recent_period)
                    
                    # Trending score: ratio of recent to early post rate
                    trending_score = recent_rate / max(0.1, early_rate) if early_rate > 0 else 1.0
                else:
                    trending_score = 1.0
            else:
                trending_score = 1.0
        else:
            trending_score = 1.0
            
        # Calculate recency factor - how recent is this cluster's activity?
        recency_score = 1.0
        if 'datetime' in cluster_docs.columns:
            # Get latest activity time and max possible time
            latest_time = pd.to_datetime(cluster_docs['datetime']).max()
            max_possible_time = pd.to_datetime(embedding_df['datetime']).max()
            
            # Calculate hours from latest activity to the most recent activity in the whole dataset
            hours_ago = (max_possible_time - latest_time).total_seconds() / 3600
            
            # Convert to a recency score (higher is better/more recent)
            # Score decreases exponentially as you go back in time
            # Within last 12 hours = very high score
            # 12-24 hours ago = high score
            # 24-48 hours ago = moderate score
            # >48 hours ago = low score
            recency_score = np.exp(-hours_ago / 24.0)  # Decay by e^(-hours/24)
        
        # Score is a combination of size, engagement, trending factor, and recency
        # We want clusters that:
        # 1. Have significant engagement
        # 2. Are trending upward in activity
        # 3. Have some minimum size to be meaningful
        # 4. Have recent activity
        engagement_weight = 0.4
        trending_weight = 0.25
        recency_weight = 0.25
        size_weight = 0.1
        
        # Normalize size logarithmically to prevent the largest cluster from dominating
        size_factor = 1 + np.log10(max(1, size))
        
        # Calculate the weighted score
        engagement_score = (
            engagement_weight * total_engagement + 
            trending_weight * trending_score * total_engagement + 
            recency_weight * recency_score * total_engagement +
            size_weight * size_factor * total_engagement
        )
        
        # Create base engagement metrics
        cluster_metrics = {
            'size': size,
            'avg_likes': avg_likes,
            'avg_recasts': avg_recasts,
            'total_engagement': total_engagement,
            'trending_score': trending_score,
            'recency_score': recency_score,
            'engagement_score': engagement_score
        }
        
        # Add conversation metrics if available
        if has_parent_info and 'reply_count' in cluster_docs.columns:
            cluster_metrics.update({
                'avg_replies': avg_replies,
                'avg_unique_repliers': avg_unique_repliers,
                'posts_with_replies': posts_with_replies,
                'reply_percentage': reply_percentage,
                'conversation_score': conversation_score
            })
            
            # Boost engagement score for clusters with high conversation
            if conversation_score > 0:
                # Add up to 20% boost based on conversation score
                conversation_boost = 1 + min(0.2, conversation_score / 1000)
                cluster_metrics['engagement_score'] *= conversation_boost
        
        # Save metrics for this cluster
        cluster_engagement[cluster_id] = cluster_metrics
        
        # Format hours ago for display if available
        hours_ago_str = ""
        if 'datetime' in cluster_docs.columns:
            latest_time = pd.to_datetime(cluster_docs['datetime']).max()
            max_possible_time = pd.to_datetime(embedding_df['datetime']).max()
            hours_ago = (max_possible_time - latest_time).total_seconds() / 3600
            hours_ago_str = f"{hours_ago:.1f}h ago"
            
        print(f"Cluster {cluster_id}: {size} posts, {avg_likes:.1f} likes, {avg_recasts:.1f} recasts, {trending_score:.2f} trend, {recency_score:.2f} recency ({hours_ago_str}), {engagement_score:.1f} score")
    
    # Sort clusters by engagement score and select top 5
    sorted_clusters = sorted(cluster_engagement.items(), key=lambda x: x[1]['engagement_score'], reverse=True)
    top_clusters = [cluster_id for cluster_id, _ in sorted_clusters[:5]]
    
    # Filter out extremely large clusters (likely catch-all clusters)
    largest_cluster = final_cluster_counts.idxmax()
    largest_cluster_percent = final_cluster_counts[largest_cluster] / sum(final_cluster_counts)
    
    # If the largest cluster contains more than 50% of all documents
    # It's probably a generic catch-all cluster - penalize its score significantly
    if largest_cluster_percent > 0.5:
        print(f"Penalizing largest cluster {largest_cluster} ({largest_cluster_percent:.1%} of all docs) to avoid catch-all clusters")
        # Reduce the score for extremely large clusters to deprioritize them
        if largest_cluster in cluster_engagement:
            # Reduce score by 90% for very large clusters
            cluster_engagement[largest_cluster]['engagement_score'] *= 0.1
            
    # Sort clusters again after potential score adjustments
    sorted_clusters = sorted(cluster_engagement.items(), key=lambda x: x[1]['engagement_score'], reverse=True)
    top_clusters = [cluster_id for cluster_id, _ in sorted_clusters[:8]]  # Select more clusters initially
    
    # Create a more balanced selection by limiting how many documents can come from one cluster
    balanced_clusters = []
    total_docs = 0
    max_docs_per_cluster = sum(final_cluster_counts) * 0.3  # No more than 30% from one cluster
    
    for cluster_id in top_clusters:
        cluster_size = cluster_engagement[cluster_id]['size']
        
        # If this would make the cluster too dominant, skip it
        if cluster_size > max_docs_per_cluster and len(balanced_clusters) > 0:
            print(f"Skipping cluster {cluster_id} with {cluster_size} docs (exceeds max {max_docs_per_cluster:.0f} docs per cluster)")
            continue
            
        balanced_clusters.append(cluster_id)
        total_docs += cluster_size
        
        # Stop once we have 5 clusters
        if len(balanced_clusters) >= 5:
            break
            
    # Use the balanced selection
    top_clusters = balanced_clusters
    
    top_docs_count = sum(cluster_engagement[c]['size'] for c in top_clusters)
    print(f"Selected top {len(top_clusters)} clusters for analysis by engagement score, representing {top_docs_count} documents")
    
    # Step 4: Extract Cluster Representatives with Advanced Analysis
    print("Extracting cluster representatives...")
    
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
        if len(cluster_points) > 1:  # Need at least 2 points for pairwise distances
            try:
                if torch.cuda.is_available():
                    # Try to use GPU for distance calculations with batch processing
                    print(f"Using GPU for distance calculations in cluster {cluster_id}")
                    
                    # Convert to torch tensor and move to GPU
                    points_tensor = torch.tensor(cluster_points, dtype=torch.float32, device='cuda')
                    
                    # Use PyTorch's built-in distance calculation (more memory efficient)
                    # Process in smaller batches if needed for large clusters
                    batch_size = 1000  # Adjust based on memory availability
                    n_points = points_tensor.shape[0]
                    
                    if n_points <= batch_size:
                        # For smaller clusters, compute all at once
                        # This computes sqrt(sum((x_i - x_j)^2)) for all i,j pairs
                        distances_matrix = torch.cdist(points_tensor, points_tensor)
                        
                        # Get upper triangle values (excluding diagonal)
                        indices = torch.triu_indices(n_points, n_points, offset=1)
                        distances = distances_matrix[indices[0], indices[1]].cpu().numpy()
                    else:
                        # For larger clusters, compute in batches
                        print(f"Processing large cluster in batches ({n_points} points)")
                        distances_list = []
                        
                        for i in range(0, n_points, batch_size):
                            end_i = min(i + batch_size, n_points)
                            batch_i = points_tensor[i:end_i]
                            
                            for j in range(i, n_points, batch_size):
                                end_j = min(j + batch_size, n_points)
                                batch_j = points_tensor[j:end_j]
                                
                                # Calculate distances between batches
                                batch_dists = torch.cdist(batch_i, batch_j)
                                
                                # For the diagonal batches, only keep upper triangle
                                if i == j:
                                    # Get upper triangle excluding diagonal
                                    batch_indices = torch.triu_indices(batch_dists.shape[0], batch_dists.shape[1], offset=1)
                                    distances_list.append(
                                        batch_dists[batch_indices[0], batch_indices[1]].cpu()
                                    )
                                else:
                                    # For off-diagonal batches, keep all distances
                                    distances_list.append(batch_dists.reshape(-1).cpu())
                        
                        # Combine all batch results
                        distances = torch.cat(distances_list).numpy()
                else:
                    raise ImportError("CUDA not available")
            except (ImportError, Exception) as e:
                print(f"Using CPU for distance calculations in cluster {cluster_id}: {e}")
                # Fall back to CPU implementation with optimized scikit-learn
                from sklearn.metrics import pairwise_distances_chunked
                
                # Use chunked calculation to save memory
                def reduce_func(D_chunk, start):
                    return D_chunk[np.triu_indices_from(D_chunk, k=1)]
                
                # Process in chunks to avoid memory issues
                chunks = pairwise_distances_chunked(
                    cluster_points, 
                    metric='euclidean',
                    working_memory=None,  # Use default (no limit)
                    reduce_func=reduce_func
                )
                
                # Combine all chunks
                distances = np.concatenate(list(chunks))
                
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
        
        # Try GPU for distance-to-center calculations if available
        try:
            if torch.cuda.is_available():
                print(f"Using GPU for distance-to-center calculations in cluster {cluster_id}")
                
                # Process in batches for large clusters
                batch_size = 5000  # Adjust based on memory availability
                n_points = len(cluster_points)
                
                # Create tensor for center
                center_tensor = torch.tensor(cluster_center, dtype=torch.float32, device='cuda')
                
                if n_points <= batch_size:
                    # Small enough to process at once
                    points_tensor = torch.tensor(cluster_points, dtype=torch.float32, device='cuda')
                    distances_to_center = torch.norm(points_tensor - center_tensor, dim=1).cpu().numpy()
                else:
                    # Process in batches for large clusters
                    print(f"Processing distances to center in batches ({n_points} points)")
                    distances_to_center = np.zeros(n_points)
                    
                    for i in range(0, n_points, batch_size):
                        end_i = min(i + batch_size, n_points)
                        # Move batch to GPU
                        batch_points = torch.tensor(cluster_points[i:end_i], dtype=torch.float32, device='cuda')
                        # Calculate distances for this batch
                        batch_distances = torch.norm(batch_points - center_tensor, dim=1).cpu().numpy()
                        # Store in result array
                        distances_to_center[i:end_i] = batch_distances
            else:
                raise ImportError("CUDA not available")
        except (ImportError, Exception) as e:
            print(f"Using CPU for distance-to-center calculations in cluster {cluster_id}: {e}")
            # Fall back to CPU implementation with optimized broadcasting
            from scipy.spatial.distance import cdist
            
            # Reshape center for broadcasting with cdist
            center_reshaped = cluster_center.reshape(1, -1)
            distances_to_center = cdist(cluster_points, center_reshaped, metric='euclidean').flatten()
            
        medoid_indices = np.argsort(distances_to_center)[:min(n_medoids * 3, len(distances_to_center))]
        
        # Get original indices and texts
        original_medoid_indices = cluster_docs.iloc[medoid_indices].index
        all_candidate_texts = cluster_docs.iloc[medoid_indices]['Text'].tolist()
        
        # Filter out low-quality texts first (too short or containing 'undefined')
        filtered_texts = []
        filtered_indices = []
        for i, text in enumerate(all_candidate_texts):
            # Skip empty or None texts
            if not text or pd.isna(text):
                continue
                
            # Skip texts that are too short or contain 'undefined'
            if len(text.split()) < 2 or 'undefined' in text.lower() or text.strip() == 'on':
                continue
                
            filtered_texts.append(text)
            filtered_indices.append(medoid_indices[i])
            
        # If we have more than n_medoids, apply semantic filtering
        if len(filtered_texts) > n_medoids:
            # Get the dataframe with the filtered candidates
            filtered_candidates = cluster_docs.iloc[filtered_indices]
            
            # For simplicity, let's manually filter to avoid index mapping issues
            # Just keep the first n_medoids unique texts (simpler approach)
            seen_texts = set()
            unique_medoid_texts = []
            
            for text in filtered_texts:
                # Skip if we've seen a very similar text already (simple text similarity)
                # This doesn't use embeddings but will catch exact or near exact duplicates
                if text in seen_texts or any(difflib.SequenceMatcher(None, text, seen).ratio() > 0.8 for seen in seen_texts):
                    continue
                
                seen_texts.add(text)
                unique_medoid_texts.append(text)
                
                if len(unique_medoid_texts) >= n_medoids:
                    break
            
            # Use the unique texts
            medoid_texts = unique_medoid_texts[:n_medoids]
        elif filtered_texts:
            # If we have fewer than needed but some valid texts
            medoid_texts = filtered_texts
        else:
            # If we couldn't find any quality texts, fall back to the original
            medoid_texts = all_candidate_texts[:n_medoids] if all_candidate_texts else []
        
        # Get engagement stats for medoids
        medoid_engagements = cluster_docs.iloc[medoid_indices][['likes_count', 'recasts_count', 'engagement_score']]
        
        # Find high engagement texts while excluding semantic duplicates
        # First, sort by engagement to prioritize high engagement posts
        sorted_by_engagement = cluster_docs.sort_values('engagement_score', ascending=False)
        
        # Using the filter_semantic_duplicates function defined at the module level
        
        # Get more candidates than we need (to allow for filtering)
        candidate_high_engagement = sorted_by_engagement.head(min(20, len(sorted_by_engagement)))
        
        # Filter out semantically similar posts
        high_engagement_docs = filter_semantic_duplicates(
            candidate_high_engagement, 
            reduced_embeddings,
            similarity_threshold=0.85,  # Adjust threshold - higher means more filtering
            sampled_indices=sampled_indices,
            embedding_index_mapping=embedding_index_mapping
        ).head(min(5, len(candidate_high_engagement)))
        
        # Get diverse samples (furthest from center) using semantic deduplication
        if len(cluster_points) > 10:
            # Get points that are further from center (get more candidates than needed)
            diverse_indices = np.argsort(distances_to_center)[-50:]  # Get many more candidates
            diverse_candidates = cluster_docs.iloc[diverse_indices]
            
            # Apply the semantic deduplication (also excluding texts similar to high engagement ones)
            # First combine high engagement docs with diverse candidates to ensure diversity
            already_selected_texts = high_engagement_docs.copy()
            combined_candidates = pd.concat([already_selected_texts, diverse_candidates])
            
            # Apply semantic filtering with a slightly lower threshold for more diversity
            semantically_diverse_df = filter_semantic_duplicates(
                combined_candidates, 
                reduced_embeddings,
                similarity_threshold=0.80,  # Lower threshold for more diversity
                sampled_indices=sampled_indices,
                embedding_index_mapping=embedding_index_mapping
            )
            
            # Exclude the high engagement docs we already have
            diverse_only = semantically_diverse_df.loc[~semantically_diverse_df.index.isin(already_selected_texts.index)]
            
            # Take up to 200 diverse samples (massively increased from 50)
            # This will give Gemini a much more comprehensive view of the cluster content
            diverse_samples = diverse_only['Text'].tolist()[:200]
        else:
            # For small clusters, use all unique texts but still filter semantically
            filtered_df = filter_semantic_duplicates(
                cluster_docs, 
                reduced_embeddings,
                similarity_threshold=0.80,
                sampled_indices=sampled_indices,
                embedding_index_mapping=embedding_index_mapping
            )
            diverse_samples = filtered_df['Text'].tolist()
        
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
        print(f"Most representative text: {medoid_texts[0][:100]}..." if len(medoid_texts) > 0 and len(medoid_texts[0]) > 100 else f"Most representative text: {medoid_texts[0]}" if len(medoid_texts) > 0 else "No representative text found")
    
    # Save detailed cluster information for further analysis
    with open('output/embedding_cluster_details.json', 'w') as f:
        # Convert any non-serializable values (like numpy types) to Python native types
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NpEncoder, self).default(obj)
        
        # Convert numpy int64 keys to Python integers
        serializable_data = {int(k): v for k, v in cluster_representatives.items()}
        
        json.dump(serializable_data, f, indent=2, cls=NpEncoder)
    
    # Step 5: Gemini Labeling of Clusters with Structured Response
    print("Using Gemini to label clusters...")
    
    # Use Gemini to label each embedding cluster
    embedding_topics = []
    
    # Initialize Gemini
    model = GenerativeModel('gemini-2.0-flash')
    
    for cluster_id, data in cluster_representatives.items():
        # Get center text
        center_text = data['medoid_texts'][0] if data['medoid_texts'] else ""
        
        # Get engagement samples
        engagement_samples = data['high_engagement_samples']
        
        # Format engagement samples with engagement metrics
        formatted_engagement_samples = []
        for i, text in enumerate(engagement_samples):
            if i < len(data['medoid_engagements']):
                engagement = data['medoid_engagements'][i]
                formatted_text = f"[👍{int(engagement['likes_count'])}|↗️{int(engagement['recasts_count'])}]: {text}"
                formatted_engagement_samples.append(formatted_text)
            else:
                formatted_engagement_samples.append(text)
        
        # Get more samples while staying within Gemini's 1M token limit
        # Using a reasonable number of samples for comprehensive cluster understanding
        # Need to leave room for the other parts of the prompt (medoid texts, engagement samples, etc.)
        all_diverse_samples = data['diverse_samples'][:100]  # Reduced from 200 to 100 to stay within token limits
        
        # Create a collection of all medoid texts for context
        all_medoid_texts = data['medoid_texts']
        
        # Estimate token count and adjust sample sizes if needed
        # Average token-to-character ratio is ~4 characters per token
        medoid_chars = sum(len(text) for text in all_medoid_texts)
        engagement_chars = sum(len(text) for text in formatted_engagement_samples)
        diverse_chars = sum(len(text) for text in all_diverse_samples)
        instruction_chars = 3000  # Approximate char count for instructions
        
        total_chars = medoid_chars + engagement_chars + diverse_chars + instruction_chars
        estimated_tokens = total_chars / 4
        
        # Target 250K tokens max per cluster prompt to stay well below limits
        if estimated_tokens > 250000:
            print(f"WARNING: Estimated token count ({estimated_tokens:.0f}) exceeds 250K, adjusting sample sizes...")
            
            # Calculate reduction factor
            reduction_factor = 250000 / estimated_tokens
            
            # Proportionally reduce each sample type
            medoid_limit = max(3, int(len(all_medoid_texts) * reduction_factor))
            engagement_limit = max(5, int(len(formatted_engagement_samples) * reduction_factor))
            diverse_limit = max(10, int(len(all_diverse_samples) * reduction_factor))
            
            # Apply limits
            all_medoid_texts = all_medoid_texts[:medoid_limit]
            formatted_engagement_samples = formatted_engagement_samples[:engagement_limit]
            all_diverse_samples = all_diverse_samples[:diverse_limit]
            
            print(f"Adjusted samples: {medoid_limit} medoids, {engagement_limit} engagement, {diverse_limit} diverse")
        
        # Create structured prompt with adjusted samples
        prompt = f"""
        I need to identify the specific trending topic being discussed in this cluster of Farcaster social media posts.
        
        REPRESENTATIVE POSTS (at cluster center):
        {chr(10).join(all_medoid_texts)}
        
        MOST ENGAGED POSTS (with likes and recasts shown):
        {chr(10).join(formatted_engagement_samples)}
        
        OTHER SAMPLE POSTS (showing diversity of cluster):
        {chr(10).join(all_diverse_samples)}
        
        ENGAGEMENT METRICS:
        - Average likes: {data['engagement_stats']['avg_likes']:.2f}
        - Average recasts: {data['engagement_stats']['avg_recasts']:.2f}
        - Total engagement score: {data['engagement_stats']['total_likes'] + 3*data['engagement_stats']['total_recasts']}
        - Cluster size: {data['size']} posts
        - Posts per hour: {data['posts_per_hour']:.2f}
        - Time range: {data['time_stats']['earliest']} to {data['time_stats']['latest']}
        {
        f'''
        CONVERSATION METRICS:
        - Total replies: {data['engagement_stats']['total_replies']}
        - Average replies per post: {data['engagement_stats']['avg_replies']:.2f}
        - Posts with replies: {data['engagement_stats']['posts_with_replies']} ({data['engagement_stats']['conversation_ratio']*100:.1f}%)
        - Average unique users replying: {data['engagement_stats']['avg_unique_repliers']:.2f}
        '''
        if has_parent_info and 'total_replies' in data['engagement_stats']
        else ''
        }
        
        Generate your response based on the following Python TypedDict schema:
        
        class Entity(TypedDict):
            name: str
            type: str  # Person, Project, Company, Protocol, etc.
        
        class EmbeddingTopic(TypedDict):
            topic_name: str  # 5 words max
            explanation: str  # Brief explanation of why trending
            key_terms: list[str]  # List of 5-10 most important terms
            key_entities: list[Entity]  # List of entities with name and type
            engagement_insight: str  # Brief insight about engagement patterns
            sentiment: str  # Positive, Neutral, Negative, Mixed
            language: str  # Primary language of the cluster (English, Vietnamese, etc.)
        
        CRITICAL REQUIREMENTS:
        1. DO NOT create topics about Farcaster itself - avoid topic names like "Farcaster Social Media Platform", "Farcaster Community Discussion" or anything that just describes users talking about the platform itself. These are too generic.
        
        2. AVOID broad platform terms in topic name - don't use "Warpcast", "Farcaster", "frames", "casts" in the topic name unless discussing a very specific feature or update to these platforms.
        
        3. SPECIFIC TOPICS ONLY - Focus on identifying the ACTUAL content topic, not the platform it appears on. For example, if users are discussing LinkedIn verification issues, the topic should be "LinkedIn Verification Error" not "Farcaster LinkedIn Discussion".
        
        4. NAME THE ACTUAL SERVICE/PRODUCT - If users are discussing a specific service like "WrapAI" or a specific game like "Mystery Location", use that exact name rather than a generic description.
        
        5. BE PRECISE IN NAMING - Use the actual topic name/term as used by the community. For example, if the discussion is about a specific token like "$DEGEN", use that name rather than "Cryptocurrency Discussion".
        
        6. FOCUS ON THE SUBJECT - Name the specific thing being discussed, not the fact that it's being discussed. For example "NFT Market Trends" not "NFT Discussion" or "Airdrop Offers" not "Airdrop Discussions".
        
        7. CONVERSATION MATTERS - If conversation metrics are provided, they are extremely important. Topics that generate more replies and have more unique repliers are often more significant community discussions rather than passive content consumption. Pay special attention to these metrics when determining the importance of a topic.
        """
        
        # Get response with JSON formatting and moderate temperature for creativity
        response = model.generate_content(
            prompt,
            generation_config=types.GenerationConfig(
                temperature=0.4,  # Added moderate temperature for more diverse topic identification
                response_mime_type="application/json"
            )
        )
        
        # Parse JSON response
        try:
            # Print raw response for debugging
            print(f"Raw response for cluster {cluster_id}: {response.text[:100]}...")
            
            # Parse the JSON response
            topic_data = json.loads(response.text)
            
            # Check if response is a dictionary or list and handle accordingly
            if isinstance(topic_data, list) and len(topic_data) > 0:
                print(f"Response is a list, converting first item to dictionary")
                topic_data = topic_data[0] if isinstance(topic_data[0], dict) else {
                    "topic_name": str(topic_data[0]) if len(topic_data) > 0 else "Unknown",
                    "explanation": ' '.join(str(x) for x in topic_data[1:3]) if len(topic_data) > 1 else "No explanation provided",
                    "key_terms": [str(x) for x in topic_data[3:8]] if len(topic_data) > 3 else [],
                    "key_entities": [],
                    "engagement_insight": "Unknown",
                    "sentiment": "Unknown"
                }
            
            # Ensure topic_data is a dictionary with required fields
            if not isinstance(topic_data, dict):
                raise TypeError(f"Expected dict, got {type(topic_data)}")
                
            # Ensure all required fields exist
            for field in ["topic_name", "explanation", "key_terms", "key_entities", "engagement_insight", "sentiment", "language"]:
                if field not in topic_data:
                    topic_data[field] = "Unknown" if field not in ["key_terms", "key_entities"] else []
            
            print(f"Successfully labeled embedding cluster {cluster_id} as '{topic_data['topic_name']}'")
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            print(f"Error processing JSON for embedding cluster {cluster_id}: {e}")
            # Fallback if JSON parsing fails
            topic_data = {
                "topic_name": f"Cluster {cluster_id} - {len(data['medoid_texts'])} posts",
                "explanation": f"Cluster with {data['engagement_stats']['avg_likes']:.1f} avg likes, {data['engagement_stats']['avg_recasts']:.1f} avg recasts",
                "key_terms": [text.split()[0] for text in data['medoid_texts'][:5] if text and len(text.split()) > 0],
                "key_entities": [],
                "engagement_insight": f"Average likes: {data['engagement_stats']['avg_likes']:.1f}, Average recasts: {data['engagement_stats']['avg_recasts']:.1f}",
                "sentiment": "Unknown",
                "language": "Unknown"
            }
        
        # Add metrics to results
        embedding_topics.append({
            'cluster_id': cluster_id,
            'size': data['size'],
            'avg_likes': float(data['engagement_stats']['avg_likes']),
            'avg_recasts': float(data['engagement_stats']['avg_recasts']),
            'total_engagement': float(data['size'] * data['engagement_stats']['avg_likes'] + 3 * data['size'] * data['engagement_stats']['avg_recasts']),
            'topic_data': topic_data
        })
    
    # Save intermediate results
    with open('output/approach3_results.json', 'w') as f:
        json.dump(embedding_topics, f, indent=2)
    
    print(f"Embeddings + Clustering approach completed in {time.time() - start_time:.2f} seconds")
    
    return embedding_topics

if __name__ == "__main__":
    # This module is imported and run from the main.py file
    pass
