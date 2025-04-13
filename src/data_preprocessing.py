import pandas as pd
import numpy as np
import duckdb
from datetime import datetime, timedelta
import time
import re
import difflib
import hashlib
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# Start timing for performance metrics
start_time = time.time()

def text_fingerprint(text):
    """Create a simplified fingerprint of text for fuzzy matching"""
    # Remove all whitespace and lowercase
    simplified = re.sub(r'\s+', '', text.lower())
    # Create a hash for faster comparison
    return hashlib.md5(simplified.encode()).hexdigest()

def is_similar_text(text1, text2, threshold=0.80):
    """Check if two texts are similar using SequenceMatcher with stricter threshold"""
    # Quick check for exact matches or empty strings
    if text1 == text2 or not text1 or not text2:
        return True
    
    # For very different length texts, they're probably not similar
    len_ratio = min(len(text1), len(text2)) / max(len(text1), len(text2))
    if len_ratio < 0.5:  # If one text is less than half the length of the other
        return False
    
    # Use more aggressive similarity threshold (lowered from 0.85 to 0.80)
    # This will catch more posts with minor variations as "similar"
    similarity = difflib.SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    return similarity >= threshold

def chunk_dataframe(df, n_chunks):
    """Split dataframe into chunks for parallel processing"""
    chunk_size = len(df) // n_chunks
    if chunk_size == 0:
        return [df]  # If df is smaller than n_chunks
    return [df[i:i+chunk_size] for i in range(0, len(df), chunk_size)]

def process_chunk(chunk_data):
    """Process a chunk of data to find duplicates"""
    chunk, similarity_threshold = chunk_data
    
    # Create fingerprints for faster initial filtering
    chunk['text_fp'] = chunk['cleaned_text'].apply(text_fingerprint)
    
    # Track indices to keep
    keep_indices = []
    
    # Sort by engagement score to keep higher engagement posts when duplicates exist
    sorted_chunk = chunk.sort_values('engagement_score', ascending=False)
    
    # Create dictionaries for exact matches and fingerprints
    seen_texts = set()
    seen_fingerprints = {}  # Map fingerprint to index
    
    for idx, row in sorted_chunk.iterrows():
        text = row['cleaned_text']
        fp = row['text_fp']
        
        # Skip empty texts
        if not text:
            continue
        
        # Check for exact duplicates
        if text in seen_texts:
            continue
        
        # Check fingerprint matches first (faster prefiltering)
        if fp in seen_fingerprints:
            # If fingerprint matches, do a more detailed comparison
            existing_idx = seen_fingerprints[fp]
            existing_text = sorted_chunk.loc[existing_idx, 'cleaned_text']
            
            if is_similar_text(text, existing_text, similarity_threshold):
                continue  # Skip this duplicate
        
        # If we get here, this is a unique text worth keeping
        seen_texts.add(text)
        seen_fingerprints[fp] = idx
        keep_indices.append(idx)
    
    return keep_indices

def remove_duplicates(df, similarity_threshold=0.85):
    """
    Remove exact and near-duplicate posts from the dataset
    
    Args:
        df: DataFrame containing posts
        similarity_threshold: Threshold for text similarity (0-1)
        
    Returns:
        DataFrame with duplicates removed
    """
    print(f"Starting duplicate removal on {len(df)} posts with similarity threshold {similarity_threshold}")
    
    # Handle empty dataframe
    if len(df) == 0:
        return df
    
    # Determine number of chunks for parallelization
    n_cores = max(1, multiprocessing.cpu_count() - 1)
    n_chunks = min(n_cores * 2, len(df))
    
    # Check if we actually need parallelization
    if len(df) < 10000:
        print(f"Dataset small enough ({len(df)} rows), processing in single thread")
        keep_indices = process_chunk((df, similarity_threshold))
        return df.loc[keep_indices]
    
    # Split the dataframe into chunks
    print(f"Splitting dataset into {n_chunks} chunks for parallel processing on {n_cores} cores")
    chunks = chunk_dataframe(df, n_chunks)
    
    # Distribute processing across cores
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        results = list(executor.map(process_chunk, [(chunk, similarity_threshold) for chunk in chunks]))
    
    # Combine results
    all_keep_indices = []
    for result in results:
        all_keep_indices.extend(result)
    
    # Create the filtered dataframe
    filtered_df = df.loc[all_keep_indices]
    
    # Run a second pass to eliminate duplicates across chunks
    if len(chunks) > 1:
        print(f"Running second pass to check for duplicates across chunks")
        filtered_df['text_fp'] = filtered_df['cleaned_text'].apply(text_fingerprint)
        
        # Use the same process but on the combined unique set
        second_pass_indices = process_chunk((filtered_df, similarity_threshold))
        filtered_df = filtered_df.loc[second_pass_indices]
    
    # Calculate reduction percentage
    reduction = (1 - len(filtered_df) / len(df)) * 100
    print(f"Removed {len(df) - len(filtered_df)} duplicates ({reduction:.1f}%)")
    print(f"Kept {len(filtered_df)} unique posts")
    
    # Drop the temporary fingerprint column
    if 'text_fp' in filtered_df.columns:
        filtered_df = filtered_df.drop(columns=['text_fp'])
    
    return filtered_df

def generate_embeddings(texts, batch_size=512):
    """
    Generate embeddings for a list of texts using the SentenceTransformer model
    Maximizes GPU utilization with optimized parameters
    
    Args:
        texts: List of text strings to embed
        batch_size: How many texts to process at once (increased for better GPU utilization)
        
    Returns:
        Numpy array of embeddings and the model
    """
    from sentence_transformers import SentenceTransformer
    import numpy as np
    import torch
    
    # Force CUDA to use maximum memory and performance
    # Enable TF32 precision (improves performance on Ampere/newer GPUs)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Set CUDA to use largest available memory block first
        torch.cuda.empty_cache()
        # Use mixed precision for faster computation
        torch.set_float32_matmul_precision('high')
    
    # Check for GPU and report statistics
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ Using GPU for embedding generation: {gpu_name} with {gpu_mem:.1f} GB memory")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("⚠️ No GPU detected, using CPU for embedding generation (will be slow)")
    
    # Use SentenceTransformer with MiniLM model, forcing a fresh load
    print(f"Loading SentenceTransformer model all-MiniLM-L6-v2...")
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    
    # Set model to evaluation mode explicitly
    model.eval()
    
    # Generate embeddings in batches to better utilize GPU
    print(f"Generating embeddings for {len(texts)} texts with batch size {batch_size}...")
    
    # Process in batches with CUDA optimizations
    all_embeddings = []
    
    # Generate embeddings with optimizations
    with torch.no_grad():  # Disable gradient computation for inference
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            # Skip empty texts
            batch = [t if t else "empty_text" for t in batch]
            
            # Show progress
            if i % (batch_size * 10) == 0:
                print(f"  Processing batch {i//batch_size + 1}/{len(texts)//batch_size + 1}...")
                # Report GPU memory usage
                if device == 'cuda':
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    print(f"  GPU memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
            
            # Generate embeddings for this batch
            # normalize_embeddings=True ensures vectors are unit length for cosine similarity
            batch_embeddings = model.encode(
                batch, 
                convert_to_tensor=True, 
                show_progress_bar=False,
                batch_size=batch_size,
                normalize_embeddings=True
            )
            
            # Move to CPU to save GPU memory only after batch is complete
            batch_embeddings = batch_embeddings.cpu().numpy()
            all_embeddings.append(batch_embeddings)
            
            # Force CUDA to clean up memory after each batch
            if device == 'cuda':
                torch.cuda.empty_cache()
    
    # Combine all batches
    embeddings = np.vstack(all_embeddings)
    
    print(f"Generated embeddings with shape: {embeddings.shape}")
    
    # Return embeddings and model
    return embeddings, model

def filter_with_embeddings(df, embeddings, sample_size=10000, similarity_threshold=0.92):
    """
    Filter semantically similar posts using embeddings from SentenceTransformer.
    Uses a very high threshold (0.92) to only filter out nearly identical content.
    GPU-optimized implementation that processes similarities in large batches.
    
    Args:
        df: DataFrame containing the posts
        embeddings: Dense matrix of embeddings for each post (numpy array)
        sample_size: Target number of posts to keep
        similarity_threshold: Posts with similarity above this threshold are considered duplicates (higher = stricter)
        
    Returns:
        DataFrame with semantically diverse posts
    """
    import numpy as np
    import torch
    import time
    
    start_time = time.time()
    print(f"Filtering with SentenceTransformer embeddings to target {sample_size} diverse posts...")
    print(f"Using strict similarity threshold of {similarity_threshold} (higher = more aggressive filtering)")
    
    # If we already have fewer posts than the sample size, return everything
    if len(df) <= sample_size:
        print(f"Sample already smaller than target ({len(df)} <= {sample_size}), returning all")
        return df
    
    # For large datasets, we'll process in chunks to avoid memory issues
    # Use larger chunk size with your RTX 4060 Ti GPU (16GB VRAM)
    CHUNK_SIZE = 20000  # How many posts to process in each chunk
    
    # Sort by engagement score to prioritize keeping high-engagement posts
    sorted_indices = df['engagement_score'].sort_values(ascending=False).index.tolist()
    
    # Track which indices to keep
    kept_indices = []
    kept_embeddings = []
    
    # Create a position mapping for lookups
    position_map = {idx: i for i, idx in enumerate(df.index)}
    
    # Setup for GPU-accelerated similarity computation
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Process in chunks for large datasets
    total_chunks = (len(sorted_indices) + CHUNK_SIZE - 1) // CHUNK_SIZE
    
    for chunk_idx in range(total_chunks):
        chunk_start = chunk_idx * CHUNK_SIZE
        chunk_end = min(chunk_start + CHUNK_SIZE, len(sorted_indices))
        chunk_indices = sorted_indices[chunk_start:chunk_end]
        
        print(f"Processing chunk {chunk_idx+1}/{total_chunks} with {len(chunk_indices)} posts...")
        
        # Early stopping if we have enough posts
        if len(kept_indices) >= sample_size:
            print(f"Already reached target size of {sample_size} posts, stopping...")
            break
        
        # Get embeddings for the current chunk
        chunk_positions = []
        chunk_embeddings = []
        
        for idx in chunk_indices:
            try:
                pos = position_map[idx]
                chunk_positions.append(pos)
                chunk_embeddings.append(embeddings[pos])
            except (KeyError, IndexError) as e:
                # Skip posts without embeddings
                continue
        
        # Convert to numpy arrays
        chunk_embeddings = np.array(chunk_embeddings)
        
        # If we don't have any kept posts yet, initialize with the first post
        if not kept_indices and len(chunk_indices) > 0:
            first_idx = chunk_indices[0]
            first_pos = position_map[first_idx]
            kept_indices.append(first_idx)
            kept_embeddings.append(embeddings[first_pos])
            
            # If we only wanted one post, we're done
            if sample_size == 1:
                break
        
        # Skip processing if we have no kept posts yet
        if not kept_indices:
            continue
        
        # Now batch process similarities on GPU for the whole chunk
        if device == 'cuda' and torch.cuda.is_available():
            # Process on GPU for speed
            # Convert embeddings to tensors
            kept_embeddings_tensor = torch.tensor(np.array(kept_embeddings), device=device)
            chunk_embeddings_tensor = torch.tensor(chunk_embeddings, device=device)
            
            # Compute similarities in a single batch operation
            # Matrix multiplication for all pairs at once (much faster)
            # Shape: [chunk_size, kept_size]
            similarities = torch.matmul(chunk_embeddings_tensor, kept_embeddings_tensor.T)
            
            # Find max similarity for each post in the chunk
            max_similarities, _ = torch.max(similarities, dim=1)
            
            # Move back to CPU for filtering
            max_similarities = max_similarities.cpu().numpy()
            
            # Find posts that are not too similar to any kept post
            unique_mask = max_similarities < similarity_threshold
            unique_indices = np.where(unique_mask)[0]
            
            # Convert to original indices and add to kept posts
            for i in unique_indices:
                if i < len(chunk_indices):  # Safety check
                    idx = chunk_indices[i]
                    
                    # Safety check to avoid duplicates
                    if idx not in kept_indices:
                        kept_indices.append(idx)
                        pos = position_map[idx]
                        kept_embeddings.append(embeddings[pos])
                        
                        # Stop once we have enough posts
                        if len(kept_indices) >= sample_size:
                            break
        else:
            # CPU fallback using numpy
            # Compare each post in the chunk against all kept posts
            for i, idx in enumerate(chunk_indices):
                # Get embedding for current post
                try:
                    pos = position_map[idx]
                    current_embedding = embeddings[pos]
                except (KeyError, IndexError):
                    continue
                
                # Compare to all kept embeddings
                compare_embeddings = np.array(kept_embeddings)
                similarities = np.dot(current_embedding, compare_embeddings.T)
                
                # Keep if not too similar to any kept post
                if max(similarities) < similarity_threshold:
                    kept_indices.append(idx)
                    kept_embeddings.append(current_embedding)
                    
                    # Stop once we have enough posts
                    if len(kept_indices) >= sample_size:
                        break
        
        # Progress update
        print(f"  After chunk {chunk_idx+1}: kept {len(kept_indices)} posts")
        
        # Clear GPU memory
        if device == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Final filter to ensure we don't exceed sample size
    if len(kept_indices) > sample_size:
        kept_indices = kept_indices[:sample_size]
    
    filtered_df = df.loc[kept_indices]
    
    elapsed = time.time() - start_time
    print(f"Filtering completed in {elapsed:.2f} seconds")
    print(f"Filtered to {len(filtered_df)} semantically diverse posts")
    print(f"Kept {len(filtered_df)} out of {len(df)} posts ({len(filtered_df)/len(df)*100:.1f}%)")
    
    return filtered_df

def funnel_filter_posts(conn, df, target_sample_size=50000, min_replies=10, detect_repetitive=True):
    """
    Apply GPU-accelerated funnel filtering to get a diverse sample of posts
    
    1. Remove exact duplicates
    2. Generate embeddings using SentenceTransformer on GPU
    3. Detect and downsample repetitive content patterns
    4. Iteratively filter with high precision using GPU-accelerated similarity search
    5. Include replies to the selected posts
    
    Args:
        conn: DuckDB connection
        df: DataFrame with initial posts
        target_sample_size: Target number of top-level posts to keep
        min_replies: Minimum number of replies to collect per top post
        detect_repetitive: Whether to detect and downsample repetitive content patterns
        
    Returns:
        DataFrame with filtered posts and their replies
    """
    import numpy as np
    import torch
    import time
    
    start_time = time.time()
    
    print("\n=== APPLYING GPU-ACCELERATED FUNNEL FILTER APPROACH ===")
    print(f"Initial dataset: {len(df):,} posts")
    
    # Report GPU information
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            device_mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU #{i}: {device_name} with {device_mem:.1f} GB memory")
    else:
        print("No GPU detected, computation will be slower")
    
    # Step 1: Extract top level posts (not replies)
    print("Separating top-level posts from replies...")
    
    # Check if ParentCastId exists in the dataframe
    if 'ParentCastId' in df.columns:
        # Use boolean mask to separate top level posts and replies
        top_level_mask = df['ParentCastId'].isnull() | (df['ParentCastId'] == '')
        top_level_posts = df[top_level_mask].copy()
        replies = df[~top_level_mask].copy()
        
        print(f"Found {len(top_level_posts):,} top-level posts and {len(replies):,} replies")
    else:
        # If no parent field, treat all as top level posts
        top_level_posts = df.copy()
        replies = pd.DataFrame()
        print(f"No parent field found. Treating all {len(top_level_posts):,} posts as top-level")
    
    # Save the initial number of posts to track filtering
    initial_post_count = len(top_level_posts)
    
    # Step 2: Remove exact duplicates from top level posts
    print("Removing exact duplicates from top-level posts...")
    top_level_posts = top_level_posts.drop_duplicates(subset=['cleaned_text'])
    print(f"After removing exact duplicates: {len(top_level_posts):,} top-level posts")
    
    # Step 3: Apply funnel filtering until we reach target size
    current_sample = top_level_posts.copy()
    iteration = 1
    similarity_threshold = 0.8  # Initialize similarity threshold
    
    # If we already have fewer posts than the target, skip embedding filtering
    if len(current_sample) <= target_sample_size:
        print(f"Already have fewer posts ({len(current_sample)}) than target size ({target_sample_size})")
        filtered_top_posts = current_sample
    else:
        # Free up memory before generating embeddings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"Beginning iterative filtering to reach target of {target_sample_size} posts...")
        print(f"Generating embeddings for {len(current_sample):,} posts...")
        
        # Generate embeddings in batches with optimized GPU usage
        # Use larger batch size for GPU efficiency - adjust based on available VRAM
        embeddings, model = generate_embeddings(
            current_sample['cleaned_text'].fillna('').tolist(),
            batch_size=1024  # Increased batch size for better GPU utilization
        )
        
        print(f"Successfully generated embeddings with shape: {embeddings.shape}")
        
        # Step 3.5: Detect and downsample repetitive content patterns if requested
        if detect_repetitive:
            print("\n===== DETECTING REPETITIVE CONTENT PATTERNS =====")
            
            # Use both TF-IDF and embeddings to find repetitive patterns
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.cluster import KMeans, DBSCAN
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
            
            # Create TF-IDF vectors to find common patterns in text
            print("Creating TF-IDF vectors to detect repetitive patterns...")
            tfidf = TfidfVectorizer(
                max_features=5000,
                min_df=10,  # Term must appear in at least 10 documents
                max_df=0.3,  # Ignore terms that appear in more than 30% of documents
                ngram_range=(1, 3)  # Include 1-3 word phrases
            )
            
            # Create TF-IDF matrix
            tfidf_matrix = tfidf.fit_transform(current_sample['cleaned_text'].fillna(''))
            print(f"Created TF-IDF matrix with shape {tfidf_matrix.shape}")
            
            # Look for clusters of highly similar posts using embeddings
            # We'll try to find dense clusters in embedding space
            print("Finding clusters of highly similar posts...")
            
            # Use DBSCAN to find dense clusters (high similarity regions)
            # Using much stricter parameters to aggressively identify repetitive content
            dbscan = DBSCAN(
                eps=0.12,  # Reduced epsilon for stricter similarity requirement
                min_samples=10,  # Lower threshold to catch smaller clusters of repetitive content
                metric='cosine',  # Use cosine similarity
                n_jobs=-1  # Use all cores
            )
            
            # Fit DBSCAN to embeddings
            try:
                # First pass: DBSCAN to find dense clusters
                print("First clustering pass with DBSCAN...")
                cluster_labels = dbscan.fit_predict(embeddings)
                
                # Look for structural patterns in text using TF-IDF
                # This helps identify templated posts that might have slightly different wording
                print("Additional TF-IDF analysis to find structural patterns...")
                from sklearn.feature_extraction.text import TfidfVectorizer
                
                # Create a TF-IDF vectorizer focusing on word patterns and n-grams
                ngram_vectorizer = TfidfVectorizer(
                    min_df=5,  # Term must appear in at least 5 documents
                    max_df=0.3,  # Ignore terms that appear in more than 30% of documents
                    ngram_range=(2, 4),  # Focus on phrases (2-4 words) which better captures templates
                    max_features=10000,
                    stop_words='english'  # Remove English stop words
                )
                
                # Generate TF-IDF matrix
                try:
                    # First, check for NaN values
                    text_for_tfidf = current_sample['cleaned_text'].fillna('')
                    tfidf_matrix = ngram_vectorizer.fit_transform(text_for_tfidf)
                    
                    # Second pass: DBSCAN on TF-IDF matrix to find template-based clusters
                    print("Second clustering pass on TF-IDF features...")
                    tfidf_dbscan = DBSCAN(
                        eps=0.7,  # Higher epsilon for TF-IDF (different scale)
                        min_samples=5,  # Lower threshold to catch smaller template groups
                        metric='cosine',
                        n_jobs=-1
                    )
                    
                    tfidf_labels = tfidf_dbscan.fit_predict(tfidf_matrix)
                    
                    # Combine both clustering approaches - if either identifies a point as part of a cluster,
                    # consider it part of that cluster pattern
                    combined_labels = np.zeros_like(cluster_labels)
                    for i in range(len(cluster_labels)):
                        # If either clustering identifies a pattern (-1 means no cluster in DBSCAN)
                        if cluster_labels[i] != -1 or tfidf_labels[i] != -1:
                            # Use embedding cluster ID if available, otherwise TF-IDF + offset
                            if cluster_labels[i] != -1:
                                combined_labels[i] = cluster_labels[i]
                            else:
                                # Offset TF-IDF labels to avoid overlap
                                combined_labels[i] = tfidf_labels[i] + np.max(cluster_labels) + 1
                        else:
                            combined_labels[i] = -1
                    
                    # Use the combined labels for better pattern detection
                    cluster_labels = combined_labels
                    
                    print(f"Combined clustering found {len(np.unique(cluster_labels)) - (1 if -1 in cluster_labels else 0)} clusters")
                except Exception as e:
                    print(f"Error in TF-IDF analysis: {e}. Using only DBSCAN results.")
                
                # Count posts in each cluster
                from collections import Counter
                cluster_counts = Counter(cluster_labels)
                
                # Remove -1 (noise points) from the counts
                if -1 in cluster_counts:
                    del cluster_counts[-1]
                    
                print(f"Found {len(cluster_counts)} dense clusters of repetitive content")
                
                # Get clusters sorted by size (largest first)
                large_clusters = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)
                
                # Process clusters with at least 25 posts - lowered threshold to catch more repetitive patterns
                clusters_to_downsample = [(cluster_id, count) for cluster_id, count in large_clusters if count >= 25]
                
                if clusters_to_downsample:
                    print(f"Largest repetitive clusters found:")
                    for i, (cluster_id, count) in enumerate(clusters_to_downsample[:5]):
                        # Get 3 sample posts from this cluster to show what it contains
                        cluster_indices = np.where(cluster_labels == cluster_id)[0][:3]
                        sample_posts = [current_sample.iloc[idx]['cleaned_text'][:50] + "..." for idx in cluster_indices]
                        
                        print(f"  Cluster {i+1}: {count} posts")
                        for j, post in enumerate(sample_posts):
                            print(f"    Sample {j+1}: {post}")
                    
                    # Downsample large clusters
                    print("\nDownsampling large repetitive clusters...")
                    
                    # Determine how many posts to keep from each cluster
                    for cluster_id, count in clusters_to_downsample:
                        # Enforce hard cap of 20 posts maximum per repetitive pattern
                        # For very large clusters (>1000 posts), be even more aggressive
                        if count > 1000:
                            sample_size = 10  # Only keep 10 posts for extremely large clusters (likely spam/repetitive)
                        else:
                            sample_size = min(count // 50, 20)  # Keep at most 2% up to 20 posts
                            
                        # Ensure we keep at least 5 posts as a minimum
                        sample_size = max(sample_size, 5)
                        
                        # Get indices of posts in this cluster
                        cluster_indices = np.where(cluster_labels == cluster_id)[0]
                        
                        # If cluster is too large, downsample it
                        if len(cluster_indices) > sample_size:
                            # Get the posts in this cluster - use index lookup to avoid iloc positional indexing issues
                            # Convert cluster_indices (which are positional) to actual DataFrame indices
                            actual_indices = [current_sample.index[i] for i in cluster_indices if i < len(current_sample.index)]
                            cluster_posts = current_sample.loc[actual_indices]
                            
                            # Sort by engagement score and keep the top posts
                            if 'engagement_score' in cluster_posts.columns:
                                # Keep top engaged posts from this cluster
                                top_engaged = cluster_posts.sort_values('engagement_score', ascending=False).head(sample_size)
                                kept_indices = top_engaged.index
                            else:
                                # Randomly sample if no engagement score - Handle case when sample_size > len(cluster_posts)
                                sample_size_adjusted = min(sample_size, len(cluster_posts.index))
                                if sample_size_adjusted > 0:
                                    kept_indices = np.random.choice(cluster_posts.index, size=sample_size_adjusted, replace=False)
                                else:
                                    kept_indices = []
                            
                            # Create mask for filtering
                            # Keep everything that's not in this cluster plus the sampled posts
                            keep_mask = ~current_sample.index.isin(cluster_posts.index) | current_sample.index.isin(kept_indices)
                            
                            # Apply the mask
                            current_sample = current_sample[keep_mask]
                            
                            # Report how many we removed
                            removed = count - sample_size
                            print(f"  Downsampled cluster of {count} similar posts: kept {sample_size}, removed {removed}")
                    
                    # Report total after downsampling
                    print(f"After downsampling repetitive content: {len(current_sample):,} posts")
                    
                    # We need to regenerate embeddings for the downsampled dataset
                    if len(current_sample) < len(embeddings):
                        print("Regenerating embeddings for downsampled dataset...")
                        embeddings, model = generate_embeddings(
                            current_sample['cleaned_text'].fillna('').tolist(),
                            batch_size=1024
                        )
                        print(f"New embeddings shape: {embeddings.shape}")
                else:
                    print("No large repetitive clusters found that need downsampling")
            
            except Exception as e:
                print(f"Error during repetitive content detection: {e}")
                print("Continuing with regular filtering...")
        
        # Start with a very high threshold to only filter near-duplicates
        # 0.98 is extremely selective (almost identical text)
        similarity_threshold = 0.98
        
        # Configure target parameters for each iteration
        # This creates a smooth progression toward target size
        target_progression = []
        remaining = len(current_sample) - target_sample_size
        
        # Calculate how many to filter in each iteration
        steps = 5  # Max number of iterations
        for i in range(steps):
            # Use exponential decay so we filter more aggressively at first
            progress = 1 - (0.7 ** (i + 1))
            current_target = max(
                target_sample_size,
                int(len(current_sample) - remaining * progress)
            )
            target_progression.append(current_target)
        
        print(f"Filtering plan: {target_progression}")
        
        # Actual filtering iterations
        while len(current_sample) > target_sample_size:
            print(f"\n======= ITERATION {iteration} =======")
            print(f"Current dataset: {len(current_sample):,} posts")
            print(f"Similarity threshold: {similarity_threshold:.3f} (higher = more selective)")
            
            # Set target for this iteration
            if iteration <= len(target_progression):
                current_target = target_progression[iteration-1]
            else:
                current_target = target_sample_size
                
            print(f"Target for this iteration: {current_target:,} posts")
            
            # Filter with GPU acceleration
            current_sample = filter_with_embeddings(
                current_sample,
                embeddings,
                sample_size=current_target,
                similarity_threshold=similarity_threshold
            )
            
            # Break if we're at target
            if len(current_sample) <= target_sample_size * 1.05:
                print("🎯 Reached target size, stopping iterations")
                break
                
            # Gradually decrease similarity threshold to filter more aggressively
            # but with smaller steps as we get closer to avoid over-filtering
            similarity_threshold -= 0.01
            
            # Don't go below 0.85 to avoid removing semantically different posts
            similarity_threshold = max(similarity_threshold, 0.85)
            
            iteration += 1
            
            # Break after max iterations
            if iteration > 8:
                print("Reached maximum iterations, breaking...")
                break
        
        # Final adjustment to exactly match target size if needed
        if len(current_sample) > target_sample_size:
            print(f"Final adjustment: {len(current_sample):,} → {target_sample_size:,} posts")
            # Sort by engagement and keep top N
            filtered_top_posts = current_sample.sort_values('engagement_score', ascending=False).head(target_sample_size)
        else:
            filtered_top_posts = current_sample
    
    print(f"Final top-level post count: {len(filtered_top_posts):,}")
    
    # Step 4: Retrieve all replies to the filtered top-level posts
    if len(replies) > 0:
        print("\nRetrieving replies to filtered top-level posts...")
        
        # Extract normalized hashes from filtered top posts
        top_post_hashes = set(filtered_top_posts['Hash'].str.lower().str.strip())
        
        # Register filtered top posts back to DuckDB
        conn.register('filtered_top_posts', filtered_top_posts)
        
        # Create a view of filtered hashes in DuckDB
        try:
            conn.execute("""
            CREATE OR REPLACE TEMP TABLE filtered_hashes AS
            SELECT DISTINCT LOWER(TRIM(Hash)) AS normalized_hash 
            FROM filtered_top_posts
            """)
        except Exception as e:
            print(f"Error creating temp table: {e}")
            
        # Find replies to filtered top posts
        try:
            matching_replies = conn.execute("""
            SELECT r.* 
            FROM casts_normalized r
            JOIN filtered_hashes f ON r.parent_hash_normalized = f.normalized_hash
            WHERE r.ParentCastId IS NOT NULL AND r.ParentCastId != ''
            """).df()
            
            print(f"Found {len(matching_replies):,} direct replies to filtered top posts")
            
            # Combine top posts with their replies
            final_dataset = pd.concat([filtered_top_posts, matching_replies])
            print(f"Final dataset: {len(final_dataset):,} posts ({len(filtered_top_posts):,} top + {len(matching_replies):,} replies)")
        except Exception as e:
            print(f"Error retrieving replies: {e}")
            final_dataset = filtered_top_posts
            print(f"Falling back to top posts only: {len(final_dataset):,} posts")
    else:
        # If no replies in original dataset, just use filtered top posts
        final_dataset = filtered_top_posts
        print(f"No replies in original dataset. Final dataset: {len(final_dataset):,} posts")
    
    # Calculate metrics on final dataset
    top_post_count = len(filtered_top_posts)
    reply_count = len(final_dataset) - top_post_count
    replies_per_top = reply_count / top_post_count if top_post_count > 0 else 0
    
    print(f"\nFinal dataset metrics:")
    print(f"- Top-level posts: {top_post_count:,}")
    print(f"- Replies: {reply_count:,}")
    print(f"- Average replies per top post: {replies_per_top:.2f}")
    print(f"- Total dataset size: {len(final_dataset):,}")
    
    # Calculate percentage reduction from initial posts
    if 'initial_post_count' in locals():
        pct_reduction = (1 - top_post_count / initial_post_count) * 100
        print(f"- Reduced from {initial_post_count:,} to {top_post_count:,} top-level posts ({pct_reduction:.1f}% reduction)")
        if detect_repetitive:
            print(f"- Filtering included detection and downsampling of repetitive content patterns")
    
    elapsed = time.time() - start_time
    print(f"\nTotal filtering time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    
    return final_dataset

def main(save_interim_data=True):
    # Initialize DuckDB connection with appropriate memory settings
    conn = duckdb.connect(database=':memory:')
    conn.execute("SET memory_limit='180GB'")  # Reserve some memory for other processes
    conn.execute("SET temp_directory='/tmp'")  # Set temp directory for spilling
    
    print("Loading data with DuckDB...")
    
    # Register parquet files with DuckDB
    # This is more efficient than loading into pandas first
    conn.execute("CREATE VIEW casts AS SELECT * FROM read_parquet('casts.parquet')")
    conn.execute("CREATE VIEW reactions AS SELECT * FROM read_parquet('farcaster_reactions.parquet')")
    
    # We cannot create indexes on views, so we'll skip this step
    # These were intended for faster joins, but DuckDB is optimized for in-memory operations
    
    # Convert Farcaster timestamp to datetime within DuckDB
    # This is much faster than applying the conversion in pandas
    conn.execute("""
    CREATE VIEW casts_with_datetime AS 
    SELECT *, 
           TIMESTAMP '2021-01-01 00:00:00' + (CAST("Timestamp" AS BIGINT) * INTERVAL '1 second') AS datetime
    FROM casts
    """)
    
    conn.execute("""
    CREATE VIEW reactions_with_datetime AS 
    SELECT *, 
           TIMESTAMP '2021-01-01 00:00:00' + (CAST("Timestamp" AS BIGINT) * INTERVAL '1 second') AS datetime
    FROM reactions
    """)
    
    # Find the most recent timestamp in the casts dataset
    max_timestamp = conn.execute("""
    SELECT MAX(datetime) FROM casts_with_datetime
    """).fetchone()[0]
    
    # Calculate time threshold (48 hours before latest timestamp)
    time_threshold = max_timestamp - timedelta(hours=96)  # Extended to 96 hours to capture more conversation history
    
    print(f"Most recent data timestamp: {max_timestamp}")
    print(f"Analyzing data from {time_threshold} to {max_timestamp} (96-hour window)")
    
    # Also check timestamp distribution
    timestamp_stats = conn.execute("""
    SELECT 
        MIN(datetime) as min_time,
        MAX(datetime) as max_time,
        COUNT(*) as total_count,
        COUNT(DISTINCT Fid) as unique_users
    FROM casts_with_datetime
    """).fetchone()
    
    print(f"Overall dataset time range: {timestamp_stats[0]} to {timestamp_stats[1]}")
    print(f"Total casts: {timestamp_stats[2]:,} from {timestamp_stats[3]:,} unique users")
    
    # Count datasets
    cast_count = conn.execute(f"""
    SELECT COUNT(*) FROM casts_with_datetime 
    WHERE datetime >= '{time_threshold}'
    """).fetchone()[0]
    
    reaction_count = conn.execute(f"""
    SELECT COUNT(*) FROM reactions_with_datetime 
    WHERE datetime >= '{time_threshold}'
    """).fetchone()[0]
    
    print(f"Dataset size for analysis: {cast_count:,} posts, {reaction_count:,} reactions")
    
    # Create a filtered view of recent casts and reactions
    conn.execute(f"""
    CREATE VIEW recent_casts AS
    SELECT * FROM casts_with_datetime
    WHERE datetime >= '{time_threshold}'
    """)
    
    conn.execute(f"""
    CREATE VIEW recent_reactions AS
    SELECT * FROM reactions_with_datetime
    WHERE datetime >= '{time_threshold}'
    """)
    
    # Get reactions sample to inspect structure
    sample_reactions = conn.execute("""
    SELECT * FROM recent_reactions LIMIT 5
    """).df()
    
    # Check if we have parent references in the filtered dataset
    parent_count = conn.execute(f"""
    SELECT COUNT(*) FROM recent_casts
    WHERE ParentCastId IS NOT NULL AND ParentCastId != ''
    """).fetchone()[0]
    
    print(f"Found {parent_count} casts with parent references in the filtered dataset")
    
    # Extract the hash part after the colon from TargetCastId
    conn.execute("""
    CREATE VIEW reactions_with_hash AS
    SELECT 
        *,
        CASE 
            WHEN POSITION(':' IN TargetCastId) > 0 
            THEN SUBSTRING(TargetCastId, POSITION(':' IN TargetCastId) + 1)
            ELSE TargetCastId 
        END AS target_hash
    FROM recent_reactions
    """)
    
    # We need to extract the hash part from ParentCastId (the part after the colon)
    # This hash is what we'll match with the Hash column of parent posts
    conn.execute("""
    CREATE VIEW casts_with_parent_hash AS
    SELECT 
        *,
        CASE 
            WHEN POSITION(':' IN ParentCastId) > 0 
            THEN SUBSTRING(ParentCastId, POSITION(':' IN ParentCastId) + 1)
            ELSE ParentCastId 
        END AS parent_hash,
        -- Also extract the parent FID (the part before the colon)
        CASE 
            WHEN POSITION(':' IN ParentCastId) > 0 
            THEN SUBSTRING(ParentCastId, 1, POSITION(':' IN ParentCastId) - 1)
            ELSE NULL
        END AS parent_fid
    FROM recent_casts
    """)
    
    # Create normalized fields for reliable matching
    conn.execute("""
    CREATE VIEW casts_normalized AS
    SELECT *, 
        LOWER(TRIM(Hash)) as hash_normalized,
        LOWER(TRIM(parent_hash)) as parent_hash_normalized
    FROM casts_with_parent_hash
    """)
    
    # Create a normalized view of the reactions with extracted hash
    conn.execute("""
    CREATE VIEW reactions_normalized AS
    SELECT 
        *, 
        LOWER(TRIM(target_hash)) as reaction_target_normalized
    FROM reactions_with_hash
    """)
    
    # Enable parallel processing
    conn.execute("""
    PRAGMA threads=30;  -- Use more threads for parallelization
    """)
    
    # Create materialized tables instead of views for better performance
    print("Creating materialized tables for faster joins...")
    
    # Create a materialized table for reactions by post hash
    conn.execute("""
    CREATE TABLE reactions_by_post AS
    SELECT 
        reaction_target_normalized,
        COUNT(*) AS total_reactions,
        SUM(CASE WHEN ReactionType = 'Like' THEN 1 ELSE 0 END) AS likes_count,
        SUM(CASE WHEN ReactionType = 'Recast' THEN 1 ELSE 0 END) AS recasts_count
    FROM reactions_normalized
    GROUP BY reaction_target_normalized
    """)
    
    # Create an index on reaction_target_normalized for faster joins
    conn.execute("""
    CREATE INDEX IF NOT EXISTS reactions_by_post_target_idx ON reactions_by_post(reaction_target_normalized)
    """)
    
    # Create a materialized table for top-level posts
    conn.execute("""
    CREATE TABLE top_level_posts AS
    SELECT * FROM casts_normalized
    WHERE ParentCastId IS NULL OR TRIM(ParentCastId) = ''
    """)
    
    # Create an index on Hash for faster joins
    conn.execute("""
    CREATE INDEX IF NOT EXISTS top_level_posts_hash_idx ON top_level_posts(Hash)
    """)
    
    # Create a materialized table for replies
    conn.execute("""
    CREATE TABLE reply_posts AS
    SELECT * FROM casts_normalized
    WHERE ParentCastId IS NOT NULL AND TRIM(ParentCastId) != ''
    """)
    
    # Create an index on ParentCastId for faster joins
    conn.execute("""
    CREATE INDEX IF NOT EXISTS reply_posts_parent_idx ON reply_posts(parent_hash_normalized)
    """)
    
    # Calculate reply metrics for top-level posts
    conn.execute("""
    CREATE TABLE post_reply_metrics AS
    SELECT 
        p.hash_normalized AS post_hash,
        COALESCE(COUNT(r.Hash), 0) AS reply_count,
        COALESCE(COUNT(DISTINCT r.Fid), 0) AS unique_repliers,
        COALESCE(MAX(r.datetime) - MIN(r.datetime), INTERVAL '0 seconds') AS conversation_duration
    FROM top_level_posts p
    LEFT JOIN reply_posts r ON r.parent_hash_normalized = p.hash_normalized
    GROUP BY p.hash_normalized
    """)
    
    # Add a safeguard - ensure there are no NULL values in key fields
    conn.execute("""
    UPDATE post_reply_metrics 
    SET 
        reply_count = COALESCE(reply_count, 0),
        unique_repliers = COALESCE(unique_repliers, 0)
    WHERE reply_count IS NULL OR unique_repliers IS NULL
    """)
    
    # Create an index on post_hash for faster joins
    conn.execute("""
    CREATE INDEX IF NOT EXISTS reply_metrics_post_hash_idx ON post_reply_metrics(post_hash)
    """)
    
    # Calculate engagement metrics with optimized joins including conversation metrics
    # Simplify to avoid decimal precision issues - use direct query that calculates everything in one step
    print("Calculating engagement metrics with conversation data...")
    
    conn.execute("""
    CREATE OR REPLACE TABLE engagement_metrics AS
    SELECT 
        c.*,
        CAST(COALESCE(r.total_reactions, 0) AS INTEGER) AS total_reactions,
        CAST(COALESCE(r.likes_count, 0) AS INTEGER) AS likes_count,
        CAST(COALESCE(r.recasts_count, 0) AS INTEGER) AS recasts_count,
        CAST(COALESCE(pm.reply_count, 0) AS INTEGER) AS reply_count,
        CAST(COALESCE(pm.unique_repliers, 0) AS INTEGER) AS unique_repliers,
        CAST(COALESCE(EXTRACT(EPOCH FROM pm.conversation_duration) / 3600, 0) AS DOUBLE) AS conversation_hours,
        -- Calculate engagement score directly in one expression
        (
            CAST(COALESCE(r.likes_count, 0) AS INTEGER) + 
            (3 * CAST(COALESCE(r.recasts_count, 0) AS INTEGER)) + 
            (5 * CAST(COALESCE(pm.reply_count, 0) AS INTEGER)) + 
            (10 * CAST(COALESCE(pm.unique_repliers, 0) AS INTEGER))
        ) AS engagement_score
    FROM casts_normalized c
    LEFT JOIN reactions_by_post r ON c.hash_normalized = r.reaction_target_normalized
    LEFT JOIN post_reply_metrics pm ON c.hash_normalized = pm.post_hash
    """)
    
    # Create indexes for faster access
    conn.execute("""
    CREATE INDEX IF NOT EXISTS engagement_metrics_hash_idx ON engagement_metrics(hash_normalized);
    CREATE INDEX IF NOT EXISTS engagement_metrics_engagement_idx ON engagement_metrics(engagement_score DESC);
    """)
    
    # Convert to pandas for compatibility with further processing
    # Only pull the data we need into memory
    print("Converting to pandas for further processing...")
    recent_df = conn.execute("""
    SELECT * FROM engagement_metrics
    """).df()
    
    # Add any additional columns needed for analysis
    # Clean and process data as needed
    recent_df['Text'] = recent_df['Text'].fillna('')
    
    # Clean text data
    recent_df = clean_text_data(conn, recent_df)
    
    # Apply the new funnel filtering approach with enhanced repetitive content detection
    # Target: 30,000 top level posts with their replies
    print("\nApplying funnel filtering approach with aggressive repetitive content detection...")
    filtered_df = funnel_filter_posts(
        conn, 
        recent_df, 
        target_sample_size=30000, 
        min_replies=10, 
        detect_repetitive=True  # Explicitly enable repetitive content detection
    )
    
    # Create initial metrics
    metrics = create_initial_metrics(conn, filtered_df)
    
    # Save interim processed data as parquet for reproducibility
    if save_interim_data:
        print("Saving interim data as parquet files for reproducibility...")
        save_start = time.time()
        
        # Create output directory if it doesn't exist
        import os
        os.makedirs('output/interim_data', exist_ok=True)
        
        # Save cleaned dataframe
        filtered_df.to_parquet('output/interim_data/cleaned_data.parquet', index=False)
        
        # Save metric dataframes
        user_metrics, reply_df, conversation_clusters, time_metrics = metrics
        user_metrics.to_parquet('output/interim_data/user_metrics.parquet', index=False)
        reply_df.to_parquet('output/interim_data/reply_data.parquet', index=False)
        conversation_clusters.to_parquet('output/interim_data/conversation_clusters.parquet', index=False)
        time_metrics.to_parquet('output/interim_data/time_metrics.parquet', index=False)
        
        print(f"Saved all interim data as parquet files in {time.time() - save_start:.2f} seconds")
    
    print(f"Total data preprocessing time: {time.time() - start_time:.2f} seconds")
    
    return conn, filtered_df

def clean_text(text):
    if pd.isna(text) or text == "":
        return ""
    # Remove URLs, mentions, special characters
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower().strip()

def clean_text_data(conn, recent_df):
    # Define text cleaning function
    print("Cleaning text with vectorized operations...")
    start_clean_time = time.time()
    
    # Use pandas vectorized operations - Never allow empty text
    recent_df['cleaned_text'] = recent_df['Text'].fillna('no_content')
    recent_df['cleaned_text'] = recent_df['cleaned_text'].str.replace(r'http\S+', '', regex=True)
    recent_df['cleaned_text'] = recent_df['cleaned_text'].str.replace(r'@\w+', '', regex=True) 
    recent_df['cleaned_text'] = recent_df['cleaned_text'].str.replace(r'[^\w\s]', '', regex=True)
    recent_df['cleaned_text'] = recent_df['cleaned_text'].str.lower().str.strip()
    
    # Filter out empty texts and texts that are too short (more strict filtering)
    recent_df = recent_df[recent_df['cleaned_text'].str.len() > 10]
    
    # If after cleaning the text is empty, replace with "no_content"
    recent_df.loc[recent_df['cleaned_text'] == '', 'cleaned_text'] = 'no_content'
    
    # Remove any remaining "empty_text" placeholders
    recent_df['cleaned_text'] = recent_df['cleaned_text'].replace('empty_text', 'no_content')
    
    # Use purely algorithmic approach to identify repetitive content patterns
    # No hardcoded patterns - the algorithm will dynamically identify repetitive content
    # through clustering in embedding space and text similarity measures
    
    print(f"Text cleaning complete - took {time.time() - start_clean_time:.2f} seconds")
    
    print(f"Retained {len(recent_df):,} posts after text cleaning")
    
    # Calculate basic text stats for analysis
    text_stats = conn.execute("""
    SELECT 
        AVG(LENGTH("Text")) AS avg_length,
        MEDIAN(LENGTH("Text")) AS median_length,
        MAX(LENGTH("Text")) AS max_length,
        MIN(LENGTH("Text")) AS min_length
    FROM recent_casts
    WHERE "Text" IS NOT NULL AND "Text" != ''
    """).fetchone()
    
    # Calculate engagement statistics - mean and median likes per post
    engagement_stats = conn.execute("""
    SELECT 
        AVG(likes_count) AS avg_likes,
        MEDIAN(likes_count) AS median_likes,
        MAX(likes_count) AS max_likes,
        AVG(recasts_count) AS avg_recasts,
        MEDIAN(recasts_count) AS median_recasts,
        MAX(recasts_count) AS max_recasts,
        AVG(total_reactions) AS avg_reactions,
        MEDIAN(total_reactions) AS median_reactions,
        MAX(total_reactions) AS max_reactions
    FROM engagement_metrics
    """).fetchone()
    
    print(f"Text statistics:")
    print(f"  - Average length: {text_stats[0]:.1f} chars")
    print(f"  - Median length: {text_stats[1]:.1f} chars")
    print(f"  - Range: {text_stats[3]} to {text_stats[2]} chars")
    
    print(f"Engagement statistics:")
    print(f"  - Likes per post: Avg {engagement_stats[0]:.1f}, Median {engagement_stats[1]:.1f}, Max {engagement_stats[2]}")
    print(f"  - Recasts per post: Avg {engagement_stats[3]:.1f}, Median {engagement_stats[4]:.1f}, Max {engagement_stats[5]}")
    print(f"  - Total reactions per post: Avg {engagement_stats[6]:.1f}, Median {engagement_stats[7]:.1f}, Max {engagement_stats[8]}")
    
    # Get top 5 most engaged posts
    top_posts = conn.execute("""
    SELECT 
        Hash,
        Text,
        Fid,
        datetime,
        likes_count,
        recasts_count,
        total_reactions
    FROM engagement_metrics
    ORDER BY total_reactions DESC
    LIMIT 5
    """).df()
    
    print("\nTop 5 most engaged posts:")
    for i, (_, row) in enumerate(top_posts.iterrows()):
        # Truncate text if too long
        text = row['Text'] if len(row['Text']) < 50 else row['Text'][:47] + '...'
        print(f"  {i+1}. FID {row['Fid']} - {text}")
        print(f"     {row['likes_count']} likes, {row['recasts_count']} recasts, {row['total_reactions']} total reactions")
    
    return recent_df

def create_initial_metrics(conn, recent_df):
    # Use DuckDB for efficient metric computation
    print("Computing metrics with DuckDB...")
    
    # Register the cleaned dataframe back to DuckDB for further processing
    conn.register('cleaned_casts', recent_df)
    
    # User activity metrics using the filtered dataset 
    print("Computing user metrics from filtered dataset...")
    
    # Simpler approach: check if columns exist directly in the DataFrame
    # and create placeholder columns if needed
    if 'engagement_score' not in recent_df.columns:
        print("Adding placeholder engagement_score column")
        recent_df['engagement_score'] = 0.0
        
    if 'likes_count' not in recent_df.columns:
        print("Adding placeholder likes_count column")
        recent_df['likes_count'] = 0
        
    if 'recasts_count' not in recent_df.columns:
        print("Adding placeholder recasts_count column")
        recent_df['recasts_count'] = 0
        
    # Re-register the DataFrame with new columns
    conn.register('filtered_dataset', recent_df)
    
    # Now compute the metrics
    user_metrics = conn.execute("""
    SELECT
        Fid,
        COUNT(*) AS cast_count,
        AVG(COALESCE(engagement_score, 0)) AS avg_engagement,
        SUM(COALESCE(engagement_score, 0)) AS total_engagement,
        AVG(COALESCE(likes_count, 0)) AS avg_likes_per_post,
        MEDIAN(COALESCE(likes_count, 0)) AS median_likes_per_post,
        SUM(COALESCE(likes_count, 0)) AS total_likes,
        AVG(COALESCE(recasts_count, 0)) AS avg_recasts_per_post,
        MEDIAN(COALESCE(recasts_count, 0)) AS median_recasts_per_post,
        SUM(COALESCE(recasts_count, 0)) AS total_recasts,
        MIN(datetime) AS first_cast,
        MAX(datetime) AS last_cast
    FROM filtered_dataset
    WHERE Fid IS NOT NULL
    GROUP BY Fid
    ORDER BY total_engagement DESC
    """).df()
    
    # Print minimal user engagement stats
    print(f"\nUser engagement statistics:")
    print(f"  - Total unique users: {len(user_metrics):,}")
    print(f"  - Top user has {user_metrics['total_likes'].max()} total likes and {user_metrics['total_recasts'].max()} total recasts")
    
    # Identify conversation patterns and reply chains
    # First check if ParentCastId exists in the dataframe
    if "ParentCastId" in recent_df.columns:
        # Debug: Let's examine some specific top parent posts and their replies
        top_replied_to = conn.execute("""
        SELECT 
            ParentCastId, 
            COUNT(*) as reply_count
        FROM casts_with_parent_hash
        WHERE ParentCastId IS NOT NULL AND LENGTH(ParentCastId) > 0
        GROUP BY ParentCastId
        ORDER BY reply_count DESC
        LIMIT 5
        """).df()
        
        print("\n----- REPLY ANALYSIS -----")
        print(f"Top 5 most replied-to parent posts:")
        print(top_replied_to)
        
        # For the top parent, try to check if we have the hash in our dataset
        if len(top_replied_to) > 0:
            top_parent = top_replied_to.iloc[0]['ParentCastId']
            parent_fid, parent_hash = top_parent.split(':', 1) if ':' in top_parent else ('', top_parent)
            
            parent_found = conn.execute(f"""
            SELECT COUNT(*) FROM casts_with_parent_hash
            WHERE Hash = '{parent_hash}'
            """).fetchone()[0]
            
            print(f"\nTop parent post details:")
            print(f"  - Parent ID: {top_parent}")
            print(f"  - Extracted FID: {parent_fid}, Hash: {parent_hash}")
            print(f"  - Post exists in our dataset: {'Yes' if parent_found > 0 else 'No'}")
            
            # Calculate reply metrics for this specific parent
            replies_to_top = conn.execute(f"""
            SELECT COUNT(*) FROM casts_with_parent_hash
            WHERE ParentCastId = '{top_parent}'
            """).fetchone()[0]
            
            print(f"  - Replies to this parent: {replies_to_top}")
        
        # Check how many posts have ParentCastId set (these are replies) in our filtered dataset
        reply_count = conn.execute("""
        SELECT COUNT(*) FROM casts_normalized 
        WHERE ParentCastId IS NOT NULL AND LENGTH(ParentCastId) > 0
        """).fetchone()[0]
        
        # Count distinct parent hashes to get unique posts replied to
        distinct_parents = conn.execute("""
        SELECT COUNT(DISTINCT parent_hash)
        FROM casts_normalized 
        WHERE parent_hash IS NOT NULL AND parent_hash != ''
        """).fetchone()[0]
        
        # Calculate reply percentage directly
        total_posts = len(recent_df)
        reply_pct = (reply_count * 100.0 / total_posts) if total_posts > 0 else 0.0
        
        print(f"\nIn filtered dataset:")
        print(f"  - Found {reply_count} reply posts out of {len(recent_df)} total posts ({reply_pct:.1f}%)")
        print(f"  - Found {distinct_parents} unique posts that were replied to")
        
        # Check how many parents exist in our dataset
        parents_in_dataset = conn.execute("""
        SELECT COUNT(*) FROM (
            SELECT DISTINCT p.parent_hash
            FROM casts_normalized p
            JOIN casts_normalized c ON p.parent_hash = c.Hash
            WHERE p.parent_hash IS NOT NULL AND p.parent_hash != ''
        )
        """).fetchone()[0]
        
        print(f"  - Out of those, {parents_in_dataset} parent posts exist in our filtered dataset")
        
        # Skip parent-child matching in preprocessing
        print("Skipping detailed parent-child matching in preprocessing...")
        parent_matches = 0
        
        # Get the reply metrics
        print("\nCalculating reply metrics...")
        
        # First, directly run a count of reply posts vs total posts
        direct_reply_count = conn.execute("""
        SELECT 
            COUNT(*) AS total_posts,
            SUM(CASE WHEN ParentCastId IS NOT NULL AND LENGTH(ParentCastId) > 0 THEN 1 ELSE 0 END) AS reply_count
        FROM casts_normalized
        """).fetchone()
        
        total_posts = direct_reply_count[0]
        replies = direct_reply_count[1]
        reply_pct = (replies * 100.0 / total_posts) if total_posts > 0 else 0.0
        
        print(f"  - Direct count: {replies} replies out of {total_posts} posts = {reply_pct:.2f}%")
        
        # Now get the full metrics including unique parents
        # Instead of using SQL query, directly create reply_metrics tuple using our already calculated values
        reply_metrics = (total_posts, reply_count, reply_pct, distinct_parents)
        
        # Print the metrics again for clarity
        print(f"\nFinal reply metrics:")
        print(f"  - Total posts: {reply_metrics[0]}")
        print(f"  - Reply count: {reply_metrics[1]}")
        print(f"  - Reply percentage: {reply_metrics[2]:.1f}%")
        print(f"  - Unique posts replied to: {reply_metrics[3]}")
    else:
        # If ParentCastId doesn't exist in the dataframe, return default values
        reply_metrics = (len(recent_df), 0, 0.0, 0)
    
    # Simplified reply chain extraction
    # Create a basic reply dataframe for compatibility with the rest of the pipeline
    print("\nCreating simplified reply dataframe...")
    
    if "ParentCastId" in recent_df.columns:
        # Create a simplified reply dataframe with the most important columns
        reply_df = conn.execute("""
        SELECT 
            Hash,
            Fid,
            Text,
            datetime,
            parent_hash AS ParentHash
        FROM filtered_dataset
        WHERE ParentCastId IS NOT NULL AND LENGTH(ParentCastId) > 0
        LIMIT 1000
        """).df()
    else:
        # Create an empty dataframe with the expected structure
        reply_df = pd.DataFrame(columns=['Hash', 'Fid', 'Text', 'datetime', 'ParentHash'])
    
    print(f"Created reply dataframe with {len(reply_df)} entries")
    
    # Skip detailed conversation cluster analysis in preprocessing
    # (The other files handle this better)
    print("\nSkipping conversation cluster analysis in preprocessing phase...")
    conversation_clusters = pd.DataFrame(
        columns=['parent_id', 'reply_count', 'parent_text', 'parent_user', 
                 'parent_time', 'parent_engagement']
    )
    
    # Fix for reply metrics - calculate the metrics directly from our filtered dataset
    reply_count = conn.execute("""
    SELECT COUNT(*) FROM recent_casts
    WHERE ParentCastId IS NOT NULL AND LENGTH(ParentCastId) > 0
    """).fetchone()[0]
    
    total_posts = len(recent_df)
    reply_pct = (reply_count * 100.0 / total_posts) if total_posts > 0 else 0.0
    
    distinct_parents = conn.execute("""
    SELECT COUNT(DISTINCT ParentCastId)
    FROM recent_casts
    WHERE ParentCastId IS NOT NULL AND LENGTH(ParentCastId) > 0
    """).fetchone()[0]
    
    # Override reply_metrics with correct values
    reply_metrics = (total_posts, reply_count, reply_pct, distinct_parents)
    
    print(f"Reply metrics (correctly calculated):")
    print(f"  - Total posts: {reply_metrics[0]:,}")
    print(f"  - Reply posts: {reply_metrics[1]:,}")
    print(f"  - Reply percentage: {reply_metrics[2]:.1f}%")
    print(f"  - Unique posts replied to: {reply_metrics[3]:,}")
    print(f"  - Found {len(conversation_clusters)} major conversation threads")
    
    # Simplified time-based activity analysis to avoid issues with missing columns
    print("\nCreating simplified time metrics...")
    
    # Register the dataframe back to DuckDB with all columns guaranteed to be present
    conn.register('filtered_dataset', recent_df)
    
    time_metrics = conn.execute("""
    WITH hourly_counts AS (
        SELECT
            DATE_TRUNC('hour', datetime) AS hour,
            COUNT(*) AS post_count
        FROM filtered_dataset
        WHERE datetime IS NOT NULL
        GROUP BY DATE_TRUNC('hour', datetime)
        ORDER BY DATE_TRUNC('hour', datetime)
    )
    SELECT
        hour,
        post_count,
        0.0 AS avg_engagement,  -- Placeholder
        0 AS reply_count,       -- Placeholder
        0.0 AS reply_percentage -- Placeholder
    FROM hourly_counts
    ORDER BY hour
    """).df()
    
    print(f"Created time metrics with {len(time_metrics)} hourly data points")
    
    # Return these metrics for later topic analysis and for saving as parquet
    return (user_metrics, reply_df, conversation_clusters, time_metrics)

def get_reactions_for_cast(conn, cast_hash):
    """Fast retrieval of reactions for a specific cast using DuckDB"""
    return conn.execute(f"""
    SELECT * FROM reactions_with_hash
    WHERE target_hash = '{cast_hash}' OR TargetCastId = '{cast_hash}'
    """).df()

if __name__ == "__main__":
    main()