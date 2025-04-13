from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import cdist
import os
import json
import time
from typing_extensions import TypedDict
from enum import Enum

def lda_kmeans_clustering(recent_df):
    """
    Approach 2: LDA + K-Means Clustering
    
    This approach uses LDA for topic modeling followed by K-means clustering 
    to identify trending topics in the dataset.
    
    Args:
        recent_df: DataFrame with cleaned posts
        
    Returns:
        dict: Structured trending topics result
    """
    print("Starting LDA + K-Means clustering approach...")
    start_time = time.time()
    
    # Filter out posts with 0 likes and 0 recasts since we're focusing on viral content
    initial_count = len(recent_df)
    
    # Handle potential NaN values in likes_count and recasts_count columns
    recent_df['likes_count'] = recent_df['likes_count'].fillna(0).astype(int)
    recent_df['recasts_count'] = recent_df['recasts_count'].fillna(0).astype(int)
    
    # Keep only posts with at least some engagement (likes or recasts > 0)
    recent_df = recent_df[(recent_df['likes_count'] > 0) | (recent_df['recasts_count'] > 0)]
    
    # Check if conversation metrics already exist from data preprocessing
    has_conversation_metrics = all(field in recent_df.columns for field in ['reply_count', 'unique_repliers'])
    
    # Make a copy of the dataframe to avoid SettingWithCopyWarning
    recent_df = recent_df.copy()
    
    # Only calculate metrics if not already present
    if not has_conversation_metrics and 'ParentCastId' in recent_df.columns:
        print("Adding conversation metrics to enhance topic modeling...")
        
        # Identify top-level posts vs replies
        recent_df.loc[:, 'is_reply'] = ~(recent_df['ParentCastId'].isnull() | (recent_df['ParentCastId'] == ''))
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
        
        # Add the metrics to the dataframe using proper .loc
        recent_df.loc[:, 'reply_count'] = recent_df['Hash'].map(
            lambda h: reply_counts.get(h, 0)
        )
        recent_df.loc[:, 'unique_repliers'] = recent_df['Hash'].map(
            lambda h: len(unique_repliers.get(h, set()))
        )
        
        has_parent_info = True
    else:
        if has_conversation_metrics:
            print("Using conversation metrics already calculated in data preprocessing...")
            has_parent_info = True
            
            # Ensure metrics are numeric
            import pandas  # Add explicit import in this scope
            if not pandas.api.types.is_numeric_dtype(recent_df['reply_count']):
                recent_df.loc[:, 'reply_count'] = pandas.to_numeric(recent_df['reply_count'], errors='coerce').fillna(0).astype(int)
            if not pandas.api.types.is_numeric_dtype(recent_df['unique_repliers']):
                recent_df.loc[:, 'unique_repliers'] = pandas.to_numeric(recent_df['unique_repliers'], errors='coerce').fillna(0).astype(int)
        else:
            print("No conversation metrics available - will not use conversation weighting...")
            has_parent_info = False
            # Add placeholder columns
            recent_df.loc[:, 'reply_count'] = 0
            recent_df.loc[:, 'unique_repliers'] = 0
    
    # If we have conversation metrics, calculate derived fields
    if has_parent_info:
        # Calculate conversation score (combining reply count and unique repliers)
        recent_df.loc[:, 'conversation_score'] = recent_df['reply_count'] * 10 + recent_df['unique_repliers'] * 20
        
        # Normalize conversation score (0-100 scale) if there are any non-zero scores
        max_convo_score = recent_df['conversation_score'].max()
        if max_convo_score > 0:
            recent_df.loc[:, 'conversation_score'] = recent_df['conversation_score'] * 100 / max_convo_score
        
        # Create an enhanced engagement score using Approach 3's improved formula
        # This better balances the importance of different engagement types
        recent_df.loc[:, 'enhanced_engagement'] = (
            recent_df['likes_count'] + 
            3 * recent_df['recasts_count'] + 
            5 * recent_df['reply_count'] + 
            10 * recent_df['unique_repliers']
        )
        
        # Follow Approach 3's methodology - focus on top-level posts for topic modeling
        if 'is_reply' not in recent_df.columns:
            # Make sure we have the is_reply column
            recent_df.loc[:, 'is_reply'] = ~(recent_df['ParentCastId'].isnull() | (recent_df['ParentCastId'] == ''))
            
        # Filter to top-level posts for better topic modeling (following Approach 3)
        topic_modeling_df = recent_df[~recent_df['is_reply']].copy()
        print(f"Focusing on {len(topic_modeling_df)} top-level posts for topic modeling (following Approach 3)")
        
        # Keep the full dataset for later clustering and labeling
        all_posts_df = recent_df.copy()
        
        # Use the enhanced engagement for topic modeling
        recent_df.loc[:, 'engagement_score'] = recent_df['enhanced_engagement']
        
        # Print statistics to verify metrics are working
        reply_sum = recent_df['reply_count'].sum()
        unique_repliers_sum = recent_df['unique_repliers'].sum()
        print(f"Total replies across all posts: {reply_sum}")
        print(f"Total unique repliers across all posts: {unique_repliers_sum}")
        print(f"Posts with at least one reply: {(recent_df['reply_count'] > 0).sum()}")
        print(f"Posts with 5+ replies: {(recent_df['reply_count'] >= 5).sum()}")
        print(f"Posts with 10+ unique repliers: {(recent_df['unique_repliers'] >= 10).sum()}")
    
    engagement_filtered_count = len(recent_df)
    print(f"Filtered out {initial_count - engagement_filtered_count} posts with 0 likes and 0 recasts.")
    
    # No hardcoded filtering here - all filtering is now done in data_preprocessing.py
    # This keeps the approach consistent across all methods
            
    filtered_count = len(recent_df)
    print(f"Proceeding with {filtered_count} posts after all filtering steps.")
    
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    os.makedirs('output/figures', exist_ok=True)
    
    # Step 1: Use Gemini to get context-specific stopwords before doing LDA
    print("Using Gemini to identify domain-specific stopwords for better LDA results...")
    
    # Use Gemini API to identify common generic words in this specific dataset
    import google.generativeai as genai
    from google.generativeai import GenerativeModel
    from google.generativeai import types
    
    # Initial set of basic stopwords for our domain
    base_stopwords = [
        # Basic conversation words and common verbs/adjectives
        'good', 'great', 'nice', 'thank', 'thanks', 'just', 'like', 'really',
        'make', 'know', 'think', 'want', 'going', 'get', 'got', 'way', 'better',
        'im', 'dont', 'doesnt', 'time', 'day', 'today', 'tomorrow', 'yesterday',
        # Common placeholder/content-free text
        'empty_text', 'no_content', 'explained', 'explaining', 'explains', 'explanation',
    ]
    
    # Get API key from environment
    api_key = os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        # Try to read from .env file as fallback
        try:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.environ.get('GOOGLE_API_KEY')
        except ImportError:
            print("dotenv package not installed, can't load from .env file")
    
    # Sample posts to send to Gemini for stopwords identification
    import random
    import pandas as pd  # Add explicit import here for pd reference
    sample_size = min(2000, len(recent_df))
    sampled_indices = random.sample(range(len(recent_df)), sample_size)
    sample_texts = [recent_df.iloc[idx]['cleaned_text'] for idx in sampled_indices if not pd.isna(recent_df.iloc[idx]['cleaned_text'])]
    sample_text = "\n\n".join(sample_texts[:1000])  # Use first 1000 for prompt size limitations
    
    # Configure Gemini and use it to identify domain-specific stopwords
    gemini_stopwords = []
    if api_key:
        print("Configuring Gemini to analyze common words in this dataset...")
        genai.configure(api_key=api_key)
        
        # Initialize Gemini
        model = GenerativeModel('gemini-2.0-flash')
        
        # Create a prompt to identify common meaningless words in this dataset
        prompt = f"""
        I need to identify common meaningless words in a social media dataset for text analysis.
        My goal is to filter out these words before running topic modeling (LDA).
        
        Here's a random sample of posts from the dataset:
        
        {sample_text}
        
        Please analyze this text and identify at least 200 common words in this dataset that:
        - Are generic and don't convey specific topics
        - Appear frequently but don't contribute to meaningful topic identification
        - Include common pronouns, prepositions, interjections, etc.
        - Include common adjectives and adverbs that don't indicate specific topics
        - Include Farcaster-specific terminology that is too general
        - Include internet slang and abbreviations
        
        DO NOT include subject-specific terms like 'bitcoin', 'web3', 'nft', etc. 
        I want to keep those for topic modeling. Only include truly generic terms.
        Also include versions like plurals, different tenses etc. (e.g., 'person', 'people', 'persons')
        
        Return the response as a JSON object with a single key "stopwords" containing the array of words.
        
        Example format:
        {{
            "stopwords": ["word1", "word2", "word3", "another", "example", ...]
        }}
        
        Your response should only contain a valid JSON object that I can parse directly.
        """
        
        try:
            # Create a simple class for structured output
            class StopwordsResponse(TypedDict):
                stopwords: list[str]
            
            # Add explicit instructions for proper JSON formatting
            enhanced_prompt = prompt + """
            
            IMPORTANT: Your response must be a proper JSON object with ONLY a single key 'stopwords' 
            containing an array of strings. No explanations or other text. Example:
            {"stopwords": ["word1", "word2", "word3"]}
            """
            
            response = model.generate_content(
                enhanced_prompt,
                generation_config=types.GenerationConfig(
                    temperature=0.2,
                    response_mime_type="application/json"  # Important: Request JSON response
                )
            )
            
            # Parse the JSON response
            try:
                stopwords_data = json.loads(response.text)
                
                # Check if response has the expected structure
                if isinstance(stopwords_data, dict) and 'stopwords' in stopwords_data:
                    gemini_stopwords = stopwords_data['stopwords']
                    print(f"Successfully extracted {len(gemini_stopwords)} dataset-specific stopwords from Gemini")
                    # Continue with response parsing
                    response_text = ""  # We don't need this anymore
                elif isinstance(stopwords_data, list):
                    # Direct list of stopwords
                    gemini_stopwords = stopwords_data
                    print(f"Successfully extracted {len(gemini_stopwords)} stopwords as a list")
                    response_text = ""  # We don't need this anymore
                else:
                    print("API returned unexpected JSON structure. Using default stopwords.")
                    response_text = response.text
                    gemini_stopwords = []
            except json.JSONDecodeError:
                # If JSON parsing fails, fall back to text
                print("Failed to parse JSON response. Falling back to text parsing.")
                response_text = response.text
            
            # If gemini_stopwords wasn't set (fallback)
            if 'gemini_stopwords' not in locals() or not gemini_stopwords:
                # Try to extract from text if JSON parsing failed
                if response_text:
                    # Find list of words in the response using regex as a fallback
                    import re
                    match = re.search(r'\[.*?\]', response_text, re.DOTALL)
                    if match:
                        try:
                            # Try to parse the extracted list as Python code
                            stopwords_list = eval(match.group(0))
                            gemini_stopwords = stopwords_list
                            print(f"Extracted {len(gemini_stopwords)} stopwords using regex fallback")
                        except Exception as e:
                            print(f"Error parsing stopwords list: {e}")
                            gemini_stopwords = []
                    else:
                        # Last resort: try to find any words in the response
                        words = re.findall(r"'([^']+)'", response_text)
                        if words:
                            gemini_stopwords = words
                            print(f"Extracted {len(gemini_stopwords)} stopwords using regex fallback")
                        else:
                            print("Could not extract stopwords list from Gemini response")
                            gemini_stopwords = []
                else:
                    print("No text response to parse. Using default stopwords.")
                    gemini_stopwords = []
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
    else:
        print("No API key available, using default stopwords list")
    
    # Combine with our base stopwords
    print("Adding dataset-specific stopwords to base list...")
    additional_stopwords = base_stopwords + gemini_stopwords
    
    # Remove duplicates and ensure all words are lowercase
    additional_stopwords = list(set([word.lower() for word in additional_stopwords]))
    
    print(f"Final stopwords list contains {len(additional_stopwords)} words")
    
    # Now proceed with text vectorization using enhanced stopwords
    print("Creating document-term matrix with optimized settings...")
    
    # For LDA, we need count vectors (not TF-IDF)
    max_features = 7500  # Increase features for better topic modeling
    
    # Create final extended stopwords list
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    extended_stop_words = list(ENGLISH_STOP_WORDS) + additional_stopwords
    
    # Create vectorizer with improved settings for better topic modeling
    vectorizer = CountVectorizer(
        max_features=max_features,
        stop_words=extended_stop_words,
        min_df=10,  # Term must appear in at least 10 documents (increased threshold)
        max_df=0.70,  # Filter out terms that appear in >70% of documents (more aggressive filtering)
        ngram_range=(1, 2),  # Include bigrams for better topic detection
        token_pattern=r'\b[a-zA-Z][a-zA-Z]{2,}\b'  # Only words with at least 3 letters
    )
    
    # Handle None values in cleaned_text for topic modeling dataset
    if 'topic_modeling_df' in locals():
        # Use the filtered dataset of top-level posts (like Approach 3)
        modeling_df = topic_modeling_df
    else:
        # Fallback to full dataset if filtering didn't happen
        modeling_df = recent_df
        
    modeling_df['cleaned_text'] = modeling_df['cleaned_text'].fillna("empty_text")
    
    print(f"Processing {len(modeling_df['cleaned_text'])} documents with {max_features} features...")
    print(f"Using top-level posts only for topic modeling, as in Approach 3's methodology")
    X = vectorizer.fit_transform(modeling_df['cleaned_text'])
    feature_names = vectorizer.get_feature_names_out()
    
    print(f"Created document-term matrix with {X.shape[0]} documents and {X.shape[1]} features")
    
    # Step 2: LDA Topic Modeling with optimal parameters
    print("Fitting LDA model with optimized processing...")
    import multiprocessing
    n_jobs = multiprocessing.cpu_count() - 1  # Use all cores except one
    print(f"Using {n_jobs} CPU cores for parallel processing")
    
    n_topics = 40  # Number of topics to model
    
    # Try to use CuML CUDA implementation if available
    try:
        from cuml.decomposition import LatentDirichletAllocation as CuLDA
        # Check if enough GPU memory is available (rough estimate)
        import cupy as cp
        free_mem = cp.cuda.runtime.memGetInfo()[0]
        required_mem = X.shape[0] * X.shape[1] * 8 * 2  # Rough estimate
        
        if free_mem > required_mem:
            print("Using CUDA-accelerated LDA implementation")
            # Convert to CuPy array
            X_gpu = cp.sparse.csr_matrix(X)
            lda = CuLDA(
                n_components=n_topics,
                random_state=42,
                max_iter=25,
                verbose=True
            )
            lda_output = lda.fit_transform(X_gpu)
            # Convert back to numpy
            lda_output = cp.asnumpy(lda_output)
            is_cuml = True
        else:
            raise ImportError("Not enough GPU memory")
            
    except ImportError as e:
        print(f"CUDA LDA not available: {e}. Using CPU implementation.")
        # For CPU implementation, use large batch size and efficient settings
        # Use aggressively large batch size to take advantage of RAM
        batch_size = max(10000, X.shape[0] // 10)  # 10% of data or 10k minimum
        print(f"Using large batch size of {batch_size} for better memory utilization")
        
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=30,  # Increased iterations for better convergence
            n_jobs=n_jobs,  # Utilize multiple CPU cores
            learning_method='batch',  # Use batch learning for better parallelization
            batch_size=batch_size,  # Much larger batch size for better memory utilization
            verbose=1,  # Show progress
            # LDA hyperparameters for better topic quality
            doc_topic_prior=0.3,  # Slightly lower alpha for more specific topic mixtures (default is 1.0)
            topic_word_prior=0.1,  # Lower beta for more focused topics (default is 1.0)
            perp_tol=0.1  # Less strict tolerance for faster convergence
        )
        
        # Fit and transform
        print("Starting LDA fit_transform - this may take a while...")
        lda_output = lda.fit_transform(X)
        is_cuml = False
    
    # Get topic-term distributions
    print("Extracting topic keywords...")
    topic_keywords = []
    
    # Define a function to filter out meaningless terms
    def is_meaningful_term(term):
        # Skip very short terms
        if len(term) < 3:
            return False
            
        # Keep special terms with $ or @ symbols (cryptocurrencies, mentions)
        if '$' in term or '@' in term or '#' in term:
            return True
            
        # Filter out terms that are mostly digits
        if sum(c.isdigit() for c in term) > len(term) // 2:
            return False
            
        # Keep domain-specific meaningful terms even if short
        domain_terms = {'nft', 'eth', 'btc', 'dao', 'defi', 'web3', 'ai', 'ml'}
        if term.lower() in domain_terms:
            return True
            
        return True
    
    # Process each topic to extract keywords
    for topic_idx, topic in enumerate(lda.components_):
        # Get top terms for this topic (get more than we need for filtering)
        top_terms_idx = topic.argsort()[:-50-1:-1]
        
        # Filter the terms to only meaningful ones
        filtered_terms = []
        for idx in top_terms_idx:
            term = feature_names[idx]
            if is_meaningful_term(term):
                filtered_terms.append(term)
                
            if len(filtered_terms) >= 15:  # Get up to 15 terms
                break
                
        # Make sure we have enough terms (at least 10)
        if len(filtered_terms) < 10:
            # If we don't have enough terms after filtering, get the top ones unfiltered
            filtered_terms = [feature_names[idx] for idx in top_terms_idx[:10]]
            
        # Take the top 10 terms for this topic
        topic_keywords.append(filtered_terms[:10])
    
    # Optimized silhouette score calculation by sampling
    if n_topics > 1:
        print("Calculating silhouette score with sampling...")
        # Use sampling to make silhouette calculation feasible
        sample_size = min(10000, X.shape[0])
        # Create random sample indices
        sample_indices = np.random.choice(X.shape[0], size=sample_size, replace=False)
        # Get sample of LDA output
        lda_output_sample = lda_output[sample_indices]
        
        # Define function to calculate silhouette score with sampling for CPU optimization
        def calculate_silhouette_sampled(data, labels, sample_size=5000):
            if len(data) <= sample_size:
                return silhouette_score(data, labels)
            else:
                # Use a random sample for silhouette calculation
                indices = np.random.choice(len(data), size=sample_size, replace=False)
                return silhouette_score(data[indices], labels[indices])
        
        # Calculate silhouette score with sampling for LDA topics
        topic_labels = np.argmax(lda_output_sample, axis=1)
        silhouette_lda = calculate_silhouette_sampled(lda_output_sample, topic_labels)
        print(f"LDA Topic Silhouette Score: {silhouette_lda:.3f}")
    
    # Calculate similarity between topics
    print("Calculating topic similarity matrix...")
    topic_similarity = cosine_similarity(lda.components_)
    
    # Save topic similarity matrix visualization
    plt.figure(figsize=(14, 12))
    sns.set(font_scale=1.1)
    mask = np.triu(np.ones_like(topic_similarity, dtype=bool))  # Create mask for upper triangle
    with sns.axes_style("white"):
        ax = sns.heatmap(
            topic_similarity, 
            annot=False,  # Disable annotations for large matrices
            cmap="YlGnBu", 
            fmt=".2f",
            linewidths=0.5,
            mask=mask,  # Only show lower triangle to reduce redundancy
            cbar_kws={'label': 'Cosine Similarity'}
        )
    plt.title('Topic Similarity Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('output/figures/topic_similarity_matrix.png', dpi=300)
    print("Saved topic similarity matrix visualization")
    
    # Hierarchical clustering of topics based on similarity
    print("Performing hierarchical clustering of topics...")
    topic_linkage = linkage(1 - topic_similarity, method='ward')
    
    # Plot dendrogram to visualize topic clusters
    plt.figure(figsize=(14, 7))
    
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
    plt.savefig('output/figures/topic_clustering_dendrogram.png')
    print("Saved topic clustering dendrogram")
    
    # Determine optimal number of topic clusters
    topic_cluster_threshold = 0.7  # Similarity threshold
    topic_clusters = fcluster(topic_linkage, t=topic_cluster_threshold, criterion='distance')
    
    # Create mapping of original topics to consolidated topics
    topic_mapping = {}
    consolidated_topic_keywords = []
    
    print("Consolidating similar topics...")
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
    row_sums = document_consolidated_topics.sum(axis=1)
    document_consolidated_topics = document_consolidated_topics / row_sums[:, np.newaxis]
    
    # Replace original LDA output with consolidated version
    lda_output = document_consolidated_topics
    topic_keywords = consolidated_topic_keywords
    n_consolidated_topics = len(consolidated_topic_keywords)
    
    print(f"Consolidated topics and their top keywords:")
    for i, keywords in enumerate(consolidated_topic_keywords):
        print(f"Topic {i+1}: {', '.join(keywords[:10])}")
    
    # Step 3: K-Means Clustering on Consolidated LDA Results
    print("\nPerforming K-Means clustering on consolidated LDA output...")
    
    # Determine optimal number of clusters
    print("Finding optimal number of clusters...")
    silhouette_scores = []
    K_range = range(3, 10)  # Try between 3 and 9 clusters
    
    # Try to leverage GPU for K-means if available
    try:
        import cupy as cp
        from cuml.cluster import KMeans as CuKMeans
        print("Using GPU-accelerated K-means")
        
        # Define silhouette function based on backend
        def silhouette_func(X, labels, sample_size=5000):
            # Convert to cupy arrays
            X_gpu = cp.array(X)
            labels_gpu = cp.array(labels)
            
            # Sample if needed
            if len(X) > sample_size:
                indices = cp.random.choice(len(X), size=sample_size, replace=False)
                X_sample = X_gpu[indices]
                labels_sample = labels_gpu[indices]
            else:
                X_sample = X_gpu
                labels_sample = labels_gpu
            
            # Convert back to numpy for sklearn silhouette
            X_sample_np = cp.asnumpy(X_sample)
            labels_sample_np = cp.asnumpy(labels_sample)
            
            return silhouette_score(X_sample_np, labels_sample_np)
        
        # Optimize by sampling for larger datasets
        lda_output_sample = lda_output
        if len(lda_output) > 10000:
            sample_indices = np.random.choice(len(lda_output), size=10000, replace=False)
            lda_output_sample = lda_output[sample_indices]
            sample_size = min(5000, len(lda_output_sample))
        else:
            sample_size = min(5000, len(lda_output))
        
        # Check if we can use GPU
        try:
            import cupy as cp
            use_gpu = True
            print("Using GPU acceleration with CuPy")
        except ImportError:
            use_gpu = False
            print("CuPy not available - using CPU implementation")
        
        # Use large batch sizes to leverage available memory
        for k in K_range:
            print(f"Testing K={k}...")
            # Use MiniBatchKMeans for improved memory and speed
            from sklearn.cluster import MiniBatchKMeans
            batch_size = min(10000, lda_output.shape[0] // 10)  # 10% of data or 10k max
            
            kmeans = MiniBatchKMeans(
                n_clusters=k, 
                random_state=42,
                batch_size=batch_size,
                n_init=3,
                verbose=1
            )
            
            # First fit full data
            print(f"Fitting MiniBatchKMeans with batch size {batch_size}...")
            clusters = kmeans.fit_predict(lda_output)
            
            # Then calculate silhouette score on sample
            try:
                clusters_sample = kmeans.predict(lda_output_sample)
                score = silhouette_func(
                    lda_output_sample, 
                    clusters_sample,
                    sample_size=min(5000, sample_size)  # Further sampling for speed
                )
                print(f"K={k}, Silhouette Score (sampled): {score:.3f}")
            except Exception as e:
                print(f"Silhouette calculation failed: {e}")
                score = 0.5 - abs(k - 5) * 0.1  # Fallback heuristic (favor k=5)
                print(f"Using fallback score: {score:.3f}")
                
            silhouette_scores.append(score)
    
    except ImportError:
        print("GPU acceleration not available, using CPU implementation...")
        # CPU implementation
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)
            clusters = kmeans.fit_predict(lda_output)
            
            # Calculate silhouette score with sampling for larger datasets
            if len(lda_output) > 10000:
                sample_indices = np.random.choice(len(lda_output), size=10000, replace=False)
                score = silhouette_score(
                    lda_output[sample_indices], 
                    clusters[sample_indices]
                )
            else:
                score = silhouette_score(lda_output, clusters)
                
            print(f"K={k}, Silhouette Score: {score:.3f}")
            silhouette_scores.append(score)
    
    # Plot silhouette scores
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    sns.lineplot(x=list(K_range), y=silhouette_scores, marker='o', color='royalblue', linewidth=2.5)
    plt.xlabel('Number of Clusters (K)', fontsize=12)
    plt.ylabel('Silhouette Score', fontsize=12)
    plt.title('Optimal Number of Clusters', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('output/figures/optimal_k_clusters.png', dpi=300)
    
    # Find optimal K (highest silhouette score)
    optimal_k = K_range[np.argmax(silhouette_scores)]
    print(f"Optimal number of clusters: {optimal_k}")
    
    # Apply K-means clustering with optimal K
    print(f"Running final K-means with {optimal_k} clusters...")
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10, verbose=1)
    clusters = kmeans.fit_predict(lda_output)
    
    # Now map the topics and clusters back to the original dataframe
    # Need to handle the case where we used the filtered topic_modeling_df
    if 'topic_modeling_df' in locals():
        # Create index mapping to handle the filtered dataset
        topic_modeling_indices = topic_modeling_df.index
        
        # Add LDA topics and clusters to the modeling dataframe first
        topic_modeling_df['lda_topics'] = lda_output.tolist()
        topic_modeling_df['lda_cluster'] = clusters
        
        # Then map these back to the full dataset for final output
        # This ensures we can use all posts (top-level and replies) in the final output
        # while still having done the topic modeling only on top-level posts
        topic_map = dict(zip(topic_modeling_indices, lda_output.tolist()))
        cluster_map = dict(zip(topic_modeling_indices, clusters))
        
        # Map to the original/full dataframe
        if 'all_posts_df' in locals():
            # Use the saved full dataset if available
            recent_df = all_posts_df
            
        # Use .loc for proper assignment
        recent_df.loc[:, 'lda_topics'] = pd.Series(topic_map).reindex(recent_df.index).tolist()
        recent_df.loc[:, 'lda_cluster'] = pd.Series(cluster_map).reindex(recent_df.index)
        
        # For posts without a mapped topic (like replies), assign them the same cluster as their parent
        # This simulates what would happen in a hierarchical model, letting replies inherit from parents
        if 'ParentCastId' in recent_df.columns:
            print("Assigning topics to replies based on their parent posts...")
            # Create a mapping from Hash to cluster
            hash_to_cluster = dict(zip(topic_modeling_df['Hash'], topic_modeling_df['lda_cluster']))
            
            # For each reply without a cluster, try to find its parent's cluster
            for idx in recent_df[recent_df['lda_cluster'].isna()].index:
                if recent_df.loc[idx, 'is_reply']:
                    parent_id = recent_df.loc[idx, 'ParentCastId']
                    if ':' in parent_id:
                        parent_hash = parent_id.split(':', 1)[1]
                    else:
                        parent_hash = parent_id
                        
                    # If we know the parent's cluster, assign it to this reply
                    if parent_hash in hash_to_cluster:
                        recent_df.loc[idx, 'lda_cluster'] = hash_to_cluster[parent_hash]
            
            # Fill any remaining NAs with the most common cluster
            most_common_cluster = recent_df['lda_cluster'].mode()[0]
            recent_df['lda_cluster'] = recent_df['lda_cluster'].fillna(most_common_cluster)
    else:
        # Direct assignment if we used the full dataset for topic modeling
        recent_df['lda_topics'] = lda_output.tolist()
        recent_df['lda_cluster'] = clusters
    
    # Cluster assignment was already done above, no need to repeat
    
    # Get cluster centers and dominant topics
    cluster_centers = kmeans.cluster_centers_
    
    # Calculate cluster sizes
    cluster_sizes = [(clusters == i).sum() for i in range(optimal_k)]
    
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
    plt.savefig('output/figures/cluster_distribution.png', dpi=300)
    
    # Find exemplar documents for each cluster (closest to cluster center)
    print("Finding exemplar documents for each cluster...")
    exemplars = {}
    for cluster_id in range(len(cluster_centers)):
        # Get documents in this cluster
        cluster_docs_idx = np.where(clusters == cluster_id)[0]
        
        # Skip empty clusters
        if len(cluster_docs_idx) == 0:
            exemplars[cluster_id] = []
            continue
            
        # Calculate distance to cluster center
        cluster_docs_vectors = lda_output[cluster_docs_idx]
        distances = cdist(cluster_docs_vectors, [cluster_centers[cluster_id]], 'euclidean').flatten()
        
        # Get indices of closest documents as candidates (for semantic filtering)
        closest_indices = np.argsort(distances)[:min(15, len(distances))]
        exemplar_indices = cluster_docs_idx[closest_indices]
        
        # Get the actual documents
        exemplar_candidates = recent_df.iloc[exemplar_indices]
        
        # Import pandas here to prevent UnboundLocalError
        import pandas as pd
        exemplar_candidates_with_vectors = pd.DataFrame({
            'index': exemplar_indices,
            'text': exemplar_candidates['Text'].values
        })
        
        # Apply semantic filtering using LDA vectors
        # Get vectors for candidates
        candidate_vectors = lda_output[exemplar_indices]
        
        # Track which indices to keep (start with the closest one)
        kept_indices = [0]  # Start with the closest one
        kept_vectors = [candidate_vectors[0]]
        
        # Compare each document against already kept documents
        for i in range(1, len(candidate_vectors)):
            current_vector = candidate_vectors[i]
            
            # Calculate max similarity to already kept documents
            similarities = np.array([cosine_similarity([current_vector], [kv])[0][0] for kv in kept_vectors])
            max_similarity = similarities.max() if len(similarities) > 0 else 0
            
            # If not too similar to existing documents, keep it
            if max_similarity < 0.85:  # Similarity threshold
                kept_indices.append(i)
                kept_vectors.append(current_vector)
            
            # Stop once we have enough diverse examples
            if len(kept_indices) >= 5:
                break
        
        # Get final exemplars
        final_exemplar_indices = exemplar_indices[kept_indices]
        final_exemplars = recent_df.iloc[final_exemplar_indices]
        
        # Store exemplars with metadata
        exemplars[cluster_id] = []
        for _, row in final_exemplars.iterrows():
            # Safely handle potential NaN values
            likes = 0 if pd.isna(row.get('likes_count', 0)) else int(row.get('likes_count', 0))
            recasts = 0 if pd.isna(row.get('recasts_count', 0)) else int(row.get('recasts_count', 0))
            
            exemplars[cluster_id].append({
                'text': row['Text'],
                'engagement_score': float(row.get('engagement_score', 0)),
                'likes': likes,
                'recasts': recasts
            })
        
        # Print exemplars for this cluster
        print(f"Exemplar documents for Cluster {cluster_id}:")
        for i, doc in enumerate(exemplars[cluster_id][:2]):  # Just print 2 to save space
            text = doc['text']
            if len(text) > 100:
                text = text[:97] + "..."
            print(f"  - {text}")
    
    # Use Gemini to label the clusters
    print("\nUsing Gemini to label clusters with meaningful topic names...")
    
    # Define TypedDict class for structured output
    class ClusterTopic(TypedDict):
        topic_name: str  # 5 words max
        explanation: str  # Brief explanation of why trending
        estimated_percentage: str  # Percentage of total conversation
        key_terms: list[str]
        engagement_level: str  # High, Medium, Low
        sentiment: str  # Positive, Neutral, Negative, Mixed
    
    # Init Gemini
    if api_key:
        genai.configure(api_key=api_key)
        model = GenerativeModel('gemini-2.0-flash')
        
        # Prepare cluster data for labeling
        cluster_data = []
        for cluster_id in range(optimal_k):
            # Get documents in this cluster
            cluster_docs = recent_df[recent_df['lda_cluster'] == cluster_id]
            
            # Check if the cluster is empty
            if len(cluster_docs) == 0:
                continue
                
            # Get top topic keywords for this cluster
            top_topic_idx = np.argmax(cluster_centers[cluster_id])
            keywords = topic_keywords[top_topic_idx][:10]  # Top 10 keywords
            
            # Calculate cluster metrics
            cluster_size = len(cluster_docs)
            percent_of_corpus = (cluster_size / len(recent_df)) * 100
            
            # Calculate engagement metrics
            try:
                avg_engagement = cluster_docs['engagement_score'].mean()
                max_engagement = cluster_docs['engagement_score'].max()
            except:
                avg_engagement = 0
                max_engagement = 0
            
            # Get VASTLY more high engagement posts for examples to utilize Gemini's 1M context window
            try:
                # Get much more high engagement posts for better cluster understanding
                # Dramatically increased from 25 to 100 for much more comprehensive coverage
                high_engagement_docs = cluster_docs.nlargest(
                    min(100, len(cluster_docs)), 
                    'engagement_score'
                )
                
                # Also get a much larger diverse set by random sampling to avoid bias
                # Increased from 15 to 150 samples to ensure broad representation
                if len(cluster_docs) > 200:
                    random_docs = cluster_docs.sample(min(150, len(cluster_docs)), random_state=42)
                    # Combine high engagement and random docs, then drop duplicates
                    high_engagement_docs = pd.concat([high_engagement_docs, random_docs]).drop_duplicates()
                elif len(cluster_docs) > 30:
                    # For smaller clusters, still get a good sample
                    random_docs = cluster_docs.sample(min(50, len(cluster_docs)), random_state=42)
                    high_engagement_docs = pd.concat([high_engagement_docs, random_docs]).drop_duplicates()
            except:
                # Fallback if engagement_score is not available - still get a large sample
                high_engagement_docs = cluster_docs.sample(min(200, len(cluster_docs)), random_state=42)
            
            # Format sample texts
            sample_texts = []
            for _, row in high_engagement_docs.iterrows():
                # Safely handle potential NaN values
                likes = 0 if pd.isna(row.get('likes_count', 0)) else int(row.get('likes_count', 0))
                recasts = 0 if pd.isna(row.get('recasts_count', 0)) else int(row.get('recasts_count', 0))
                
                formatted_text = f"[👍{likes}|↗️{recasts}]: {row.get('Text', '')}"
                sample_texts.append(formatted_text)
            
            # Add to cluster data - include more examples for better context, but stay within token limits
            # Estimate token count: average Farcaster post is ~100 chars or ~25 tokens
            # 100 posts = ~2500 tokens, which is safe for a cluster prompt while leaving room for the rest
            sample_limit = 100  # Reduced from 250 to 100 to ensure we stay within token limits
            
            cluster_data.append({
                'cluster_id': cluster_id,
                'keywords': keywords,
                'size': cluster_size,
                'percent': percent_of_corpus,
                'avg_engagement': avg_engagement,
                'examples': sample_texts[:sample_limit]  # Use 100 examples instead of 250 to avoid token limit issues
            })
        
        # Process each cluster with Gemini
        cluster_topics = []
        
        for cluster in cluster_data:
            # Create prompt for Gemini
            # Prepare enhanced cluster info with conversation metrics if available
            cluster_info = []
            cluster_info.append(f"Keywords: {', '.join(cluster['keywords'])}")
            cluster_info.append(f"Cluster size: {cluster['size']} posts ({cluster['percent']:.1f}% of all posts)")
            cluster_info.append(f"Average engagement: {cluster['avg_engagement']:.2f}")
            
            # Add conversation metrics if we have that data
            has_conversation_metrics = (
                'reply_count' in recent_df.columns and 
                'unique_repliers' in recent_df.columns and
                len(cluster['examples']) > 0
            )
            
            if has_conversation_metrics:
                # Calculate average conversation metrics for this cluster
                cluster_posts = []
                for post in cluster['examples']:
                    if isinstance(post, dict):
                        continue
                        
                    # Extract just the text part after the engagement metrics
                    # Format is typically: "[👍X|↗️Y]: Text content"
                    if isinstance(post, str) and ']:' in post:
                        text_content = post.split(']: ', 1)[1] if ']: ' in post else post
                        
                        # Find rows that match this text content
                        matching_rows = recent_df[recent_df['Text'].str.contains(text_content, regex=False, na=False)]
                        
                        if not matching_rows.empty:
                            cluster_posts.append(matching_rows.iloc[0])
                        
                if cluster_posts:
                    avg_replies = sum(p.get('reply_count', 0) if not pd.isna(p.get('reply_count', 0)) else 0 
                                     for p in cluster_posts) / len(cluster_posts)
                    avg_repliers = sum(p.get('unique_repliers', 0) if not pd.isna(p.get('unique_repliers', 0)) else 0 
                                      for p in cluster_posts) / len(cluster_posts)
                    
                    # Add conversation metrics to cluster info
                    cluster_info.append(f"Average replies per post: {avg_replies:.1f}")
                    cluster_info.append(f"Average unique repliers per post: {avg_repliers:.1f}")
            
            prompt = f"""
            I need to identify the specific topic being discussed in a cluster of social media posts from Farcaster.
            
            INFORMATION ABOUT THIS CLUSTER:
            {chr(10).join('- ' + info for info in cluster_info)}
            
            SAMPLE POSTS (with like👍 and recast↗️ counts):
            {' '.join(cluster['examples'])}
            
            Based on these keywords and sample posts, provide:
            1. A concise name for this topic (5 words max)
            2. A brief explanation of what this topic is about
            3. A list of 5-7 key terms that best describe this topic
            4. The engagement level (High, Medium, or Low)
            5. The sentiment (Positive, Neutral, Negative, or Mixed)
            
            {"PAY SPECIAL ATTENTION TO CONVERSATION METRICS: Topics with higher reply counts and more unique repliers often represent more significant trends that generate active discussion, not just passive consumption." if has_conversation_metrics else ""}
            
            Format your response as valid JSON with this structure:
            {{
                "topic_name": "Concise topic name",
                "explanation": "Brief explanation of the topic",
                "estimated_percentage": "{cluster['percent']:.1f}%",
                "key_terms": ["term1", "term2", "term3", "term4", "term5"],
                "engagement_level": "Medium",
                "sentiment": "Neutral"
            }}
            
            CRITICAL REQUIREMENTS:
            1. DO NOT create topics about Farcaster itself - avoid topic names like "Farcaster Social Media Platform", "Farcaster Community Discussion" or anything that just describes users talking about the platform itself. These are too generic.
            
            2. AVOID broad platform terms in topic name - don't use "Warpcast", "Farcaster", "frames", "casts" in the topic name unless discussing a very specific feature or update to these platforms.
            
            3. SPECIFICITY REQUIRED - focus on specific projects, events, technologies, or developments that users are discussing, not just the act of using Farcaster or social media.
            
            4. SUBSTANCE FOCUS - if the cluster seems to be about general Farcaster usage with no specific trend, try to identify a meaningful sub-topic that represents actual specific content being discussed.
            
            5. NAME IT PROPERLY - If discussing specific crypto tokens, NFT collections, or projects, use their proper names (e.g., "Solana Token Price Surge" rather than "Cryptocurrency Price Discussion")
            """
            
            # Call Gemini for cluster labeling
            try:
                # Create schema for structured output
                cluster_topic_schema = {
                    "type": "object",
                    "properties": {
                        "topic_name": {"type": "string"},
                        "explanation": {"type": "string"},
                        "estimated_percentage": {"type": "string"},
                        "key_terms": {"type": "array", "items": {"type": "string"}},
                        "engagement_level": {"type": "string", "enum": ["High", "Medium", "Low"]},
                        "sentiment": {"type": "string", "enum": ["Positive", "Neutral", "Negative", "Mixed"]}
                    },
                    "required": ["topic_name", "explanation", "key_terms", "engagement_level", "sentiment"]
                }
                
                response = model.generate_content(
                    prompt,
                    generation_config=types.GenerationConfig(
                        temperature=0.4,  # Increased from 0.2 to 0.4 for more diverse topic identification
                        response_mime_type="application/json"
                    )
                )
                
                # Parse response
                try:
                    topic_data = json.loads(response.text)
                    print(f"Successfully labeled cluster {cluster['cluster_id']} as '{topic_data.get('topic_name', 'Unknown')}'")
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON for cluster {cluster['cluster_id']}: {e}")
                    # Fallback
                    topic_data = {
                        "topic_name": f"Cluster {cluster['cluster_id']}",
                        "explanation": f"Keywords: {', '.join(cluster['keywords'][:5])}",
                        "estimated_percentage": f"{cluster['percent']:.1f}%",
                        "key_terms": cluster['keywords'][:5],
                        "engagement_level": "Medium",
                        "sentiment": "Neutral"
                    }
            except Exception as e:
                print(f"Error calling Gemini API for cluster {cluster['cluster_id']}: {e}")
                # Fallback
                topic_data = {
                    "topic_name": f"Cluster {cluster['cluster_id']}",
                    "explanation": f"Keywords: {', '.join(cluster['keywords'][:5])}",
                    "estimated_percentage": f"{cluster['percent']:.1f}%",
                    "key_terms": cluster['keywords'][:5],
                    "engagement_level": "Medium",
                    "sentiment": "Neutral"
                }
            
            # Calculate metrics for trending score
            trend_metrics = {
                'size': cluster['size'],
                'percent': cluster['percent'],
                'avg_engagement': float(cluster['avg_engagement'])
            }
            
            # Add engagement level mapping
            engagement_mapping = {
                'High': 1.0,
                'Medium': 0.5,
                'Low': 0.1
            }
            
            engagement_score = engagement_mapping.get(
                topic_data.get('engagement_level', 'Medium'),
                0.5
            )
            
            # Calculate trending score: combination of size, engagement and recency
            trending_score = (
                (trend_metrics['percent'] / 100) * 0.3 +
                min(trend_metrics['avg_engagement'] / 50, 1.0) * 0.4 +
                engagement_score * 0.3
            ) * 100
            
            # Add to cluster topics with metrics
            cluster_topics.append({
                'cluster_id': cluster['cluster_id'],
                'size': cluster['size'],
                'percent': cluster['percent'],
                'avg_engagement': float(cluster['avg_engagement']),
                'keywords': cluster['keywords'],
                'topic_data': topic_data,
                'trending_score': trending_score,
                'exemplars': exemplars.get(cluster['cluster_id'], [])[:3]  # Add top 3 exemplars
            })
    else:
        print("No API key available for Gemini, using fallback topic names")
        # Create fallback topic data without Gemini
        cluster_topics = []
        for cluster_id in range(optimal_k):
            # Get documents in this cluster
            cluster_docs = recent_df[recent_df['lda_cluster'] == cluster_id]
            
            # Check if the cluster is empty
            if len(cluster_docs) == 0:
                continue
                
            # Get top topic keywords for this cluster
            top_topic_idx = np.argmax(cluster_centers[cluster_id])
            keywords = topic_keywords[top_topic_idx][:10]  # Top 10 keywords
            
            # Calculate cluster metrics
            cluster_size = len(cluster_docs)
            percent_of_corpus = (cluster_size / len(recent_df)) * 100
            
            # Calculate engagement if possible
            try:
                avg_engagement = cluster_docs['engagement_score'].mean()
            except:
                avg_engagement = 0
             
            # Check if this might be a generic Farcaster topic
            farcaster_related_terms = ['farcaster', 'warpcast', 'cast', 'casts', 'frame', 'frames', 'social', 'media']
            
            # Filter out generic Farcaster-related terms for topic naming
            filtered_keywords = []
            for kw in keywords:
                if kw.lower() not in farcaster_related_terms:
                    filtered_keywords.append(kw)
            
            # If we filtered out too many terms, add some back to have enough for a name
            if len(filtered_keywords) < 3:
                # Add more specific keywords, even if they're Farcaster-related
                for kw in keywords:
                    if kw not in filtered_keywords and len(filtered_keywords) < 3:
                        filtered_keywords.append(kw)
            
            # Create a more specific topic name
            specific_topic_name = ' '.join(filtered_keywords[:3])
            
            # Create fallback topic data with a more specific name
            topic_data = {
                "topic_name": f"Topic: {specific_topic_name}",
                "explanation": f"Based on keywords: {', '.join(keywords)}",
                "estimated_percentage": f"{percent_of_corpus:.1f}%",
                "key_terms": keywords[:5],
                "engagement_level": "Medium",
                "sentiment": "Neutral"
            }
            
            # Add to cluster topics with metrics
            cluster_topics.append({
                'cluster_id': cluster_id,
                'size': cluster_size,
                'percent': percent_of_corpus,
                'avg_engagement': float(avg_engagement),
                'keywords': keywords,
                'topic_data': topic_data,
                'trending_score': percent_of_corpus * (1 + avg_engagement/100),
                'exemplars': exemplars.get(cluster_id, [])[:3]  # Add top 3 exemplars
            })
    
    # Filter out overly generic Farcaster topics before final sorting
    filtered_cluster_topics = []
    farcaster_terms = ['farcaster', 'warpcast', 'cast', 'frames', 'social', 'media', 'platform']
    
    for cluster in cluster_topics:
        # Skip extremely large clusters (>70% of corpus) with generic Farcaster-focused names
        if (cluster['percent'] > 70 and 
            any(term.lower() in cluster['topic_data']['topic_name'].lower() for term in farcaster_terms)):
            # Adjust the score down for large, generic clusters about Farcaster
            print(f"Reducing score for large, generic Farcaster topic: {cluster['topic_data']['topic_name']}")
            # Reduce score by 80% for these clusters
            cluster['trending_score'] *= 0.2
            
        # For all other clusters, check if the topic is just about Farcaster itself
        elif any(term.lower() in cluster['topic_data']['topic_name'].lower() for term in ['farcaster platform', 'social media platform']):
            # Reduce score for topics that are purely about the platform itself
            print(f"Reducing score for general platform topic: {cluster['topic_data']['topic_name']}")
            cluster['trending_score'] *= 0.5
            
        filtered_cluster_topics.append(cluster)
    
    # Sort clusters by trending score
    filtered_cluster_topics.sort(key=lambda x: x['trending_score'], reverse=True)
    
    # Print top clusters
    print("\nTop clusters by trending score:")
    for i, cluster in enumerate(filtered_cluster_topics[:5]):
        print(f"{i+1}. {cluster['topic_data']['topic_name']} (Score: {cluster['trending_score']:.1f})")
        print(f"   {cluster['topic_data']['explanation']}")
        print(f"   Size: {cluster['size']} posts ({cluster['percent']:.1f}% of corpus)")
        print(f"   Keywords: {', '.join(cluster['keywords'])}")
        print()
    
    # Build final result structure
    approach2_results = {
        'topics': [],
        'analysis_period': f"{recent_df['datetime'].min().strftime('%Y-%m-%d')} to {recent_df['datetime'].max().strftime('%Y-%m-%d')}" if 'datetime' in recent_df.columns else "",
        'total_posts_analyzed': len(recent_df)
    }
    
    # Add each cluster as a topic (using filtered clusters)
    for cluster in filtered_cluster_topics:
        # Skip empty clusters
        if cluster['size'] == 0:
            continue
            
        # Format topic data
        topic = {
            'name': cluster['topic_data']['topic_name'],
            'explanation': cluster['topic_data']['explanation'],
            'estimated_percentage': cluster['topic_data']['estimated_percentage'],
            'key_terms': [{'term': term, 'frequency': i+1} for i, term in enumerate(cluster['topic_data']['key_terms'])],
            'key_entities': [],  # We don't have entity extraction in this approach
            'engagement_level': cluster['topic_data']['engagement_level'],
            'trending_score': float(cluster['trending_score']),
            'exemplar_posts': cluster['exemplars']
        }
        
        # Add entities if we can extract them from keywords
        for keyword in cluster['keywords']:
            if keyword[0].isupper() or '$' in keyword or '@' in keyword:
                entity_type = 'Company' if '$' in keyword else 'Person' if '@' in keyword else 'Project'
                topic['key_entities'].append({
                    'name': keyword,
                    'type': entity_type,
                    'relevance': 'High'
                })
        
        approach2_results['topics'].append(topic)
    
    # Save results
    print("Saving approach 2 results...")
    os.makedirs('output', exist_ok=True)
    with open('output/approach2_results.json', 'w') as f:
        json.dump(approach2_results, f, indent=2)
    
    # Save cluster assignment as parquet for potential further analysis
    if 'lda_cluster' in recent_df.columns:
        recent_df[['lda_cluster']].to_parquet('output/interim_data/lda_clusters.parquet')
    
    print(f"Approach 2 completed in {time.time() - start_time:.2f} seconds")
    
    return approach2_results

if __name__ == "__main__":
    # This can be run as a standalone module
    print("This module is designed to be imported and used by main.py")