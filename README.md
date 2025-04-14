# Farcaster Trending Topics Analysis

This project identifies trending topics on the Farcaster social protocol by analyzing a 48-hour window of conversations using multiple methodologies. It also includes tools for analyzing related blockchain tokens and trends.

## Problem Statement

Detecting trending topics in social media platforms is valuable for understanding community interests, discovering emerging discussions, and tracking engagement patterns. This project compares three approaches to identify the top 5 trending topics within the Farcaster ecosystem.

## Approaches

1. **Direct LLM Analysis**: Leverage Gemini's context window to process thousands of sampled casts and directly extract topics.

2. **LDA + K-Means Clustering**: Apply traditional topic modeling with LDA, cluster the results with K-means, and use Gemini to label the clusters.

3. **Embedding-Based Clustering**: Generate text embeddings with all-MiniLM-L6-v2, cluster similar conversations, and use Gemini to identify the topic of each cluster.

4. **Topic Novelty Filtering**: Apply a smart filtering system that penalizes common baseline topics while promoting specific, emerging conversations. Uses Gemini to generate domain-relevant baseline topics that shouldn't be considered "trending."

## Dataset Information

We analyze the most recent 48-hour window available in the Farcaster ecosystem using:

1. **Casts Dataset**: Contains 161M public posts with text content, user IDs, timestamps
2. **Reactions Dataset**: Contains 295M public reactions (likes and recasts) for engagement analysis

The analysis uses engagement metrics to identify truly trending topics rather than just frequent discussions.

## Setup and Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/fc-trending-topics.git
   cd fc-trending-topics
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Set up your API keys:
   ```
   # For Gemini LLM access
   export GOOGLE_API_KEY=your_api_key_here  # On Windows: set GOOGLE_API_KEY=your_api_key_here
   
   # For blockchain analysis (optional)
   cp .env.example .env
   # Edit .env with your Alchemy API keys
   ```

4. Prepare your data:
   - Place your `casts.parquet` file in the root directory
   - Place your `farcaster_reactions.parquet` file in the root directory

## Usage

### Main Trending Topics Analysis

Run the main script to execute the full pipeline:

```
python src/main.py
```

This will:
1. Preprocess the data with DuckDB
2. Run all three topic detection approaches
3. Generate visualizations and comparison reports
4. Save results to the `output` directory

### Ghibli Token Analysis

To analyze the Ghibli-related ERC20 tokens across multiple blockchains:

```
python analyze_ghibli_tokens.py
```

This will:
1. Connect to various blockchain networks via Alchemy
2. Retrieve token data including creation dates, liquidity, and price history
3. Generate CSV and JSON reports with detailed analysis

## Project Structure

```
fc-trending-topics/
├── docs/                # Documentation files
├── src/                 # Source code
│   ├── main.py          # Main entry point
│   ├── data_preprocessing.py   # Data preprocessing with DuckDB
│   ├── approach1_direct_llm.py # Direct LLM analysis
│   ├── approach2_lda_kmeans.py # LDA + K-Means clustering
│   ├── approach3_embeddings.py # Embeddings + clustering
│   └── approach4_generator_critic.py # Generator-critic pattern
├── output/              # Generated output files and figures
│   └── figures/         # Visualizations
├── analyze_ghibli_tokens.py # Analysis of Ghibli-related ERC20 tokens
├── requirements.txt     # Dependencies
└── README.md            # Project documentation
```

## Requirements

- Python 3.10+
- Libraries: pandas, numpy, scikit-learn, sentence-transformers, hdbscan, matplotlib, seaborn
- Access to Gemini 2.0 Flash Lite API via Google AI
- TypedDict for structured API responses

### CUDA Support (Optional)
For GPU acceleration, ensure you have:
- NVIDIA GPU with CUDA support
- CUDA Toolkit 12.x installed
- cuML (RAPIDS) libraries installed
- Web3 for blockchain analysis

The codebase will automatically use CUDA acceleration if available for:

1. **Approach 2 (LDA + K-means)**:
   - LDA topic modeling with cuML
   - K-means clustering with cuML (10-20x faster)
   - Silhouette score calculation on GPU
   - Topic similarity matrix calculation with cuML

2. **Approach 3 (Embeddings)**:
   - Sentence embedding generation on GPU (5-10x faster)
   - PCA dimensionality reduction with cuML
   - HDBSCAN clustering on GPU
   - Pairwise distance calculations with cuML
   - Intra-cluster distance calculations
   - Cluster coherence calculations

3. **Data Preprocessing**:
   - Text cleaning with cuDF
   - Optimized memory allocation when GPU is detected

All GPU accelerations are implemented with graceful fallbacks to CPU if CUDA libraries are not available.
>>>>>>> 213a17c (Initial Implementation)

## Out of Scope (Future Possibilities)

### 1. Full Reinforcement Learning Approach

A reinforcement learning (RL) based approach could provide superior results for trending topic detection:

- **Concept**: Train a specialized model to identify novel, specific trending topics while filtering out general baseline discussions
- **Architecture**: A 1B parameter LLM fine-tuned with RLHF or a vector-based RL model that learns optimal topic boundaries
- **Training Process**: 
  1. Generate positive examples (specific trending topics) and negative examples (general topics like "Crypto" or "AI")
  2. Define reward signals based on topic specificity, novelty, and engagement metrics
  3. Train the model to maximize rewards through iterative improvement
- **Advantages**: Would learn optimal filtering parameters over time, adapting to changing platform dynamics
- **Challenges**: Requires significant labeled data, computational resources, and ongoing refinement

### 2. Negative-Only Learning Approach

A more focused variation of reinforcement learning that emphasizes what to avoid rather than what to find:

- **Concept**: Train exclusively on negative examples (baseline topics) and optimize for maximum distance from these topics
- **Architecture**: Similar to full RL but with simplified reward function centered on avoiding known patterns
- **Training Process**:
  1. Generate comprehensive negative examples of common baseline topics on Farcaster
  2. Train model to identify and penalize patterns matching these examples
  3. Apply positive scoring only for specificity and engagement metrics
- **Advantages**: Simpler to implement than full RL, focuses on what we know with certainty, requires less labeled data
- **Challenges**: May be too conservative, could filter out legitimately trending topics that share some characteristics with baseline topics

### 3. Generator-Critic LLM Approach

A GAN-like architecture using two language models to discover and refine trending topics:

- **Concept**: One LLM generates candidate trending topics, another evaluates them on potential to drive engagement
- **Architecture**: Generator LLM + Critic LLM working in tandem
- **Training Process**:
  1. Generator proposes candidate trending topics based on data analysis
  2. Critic scores each topic (1-10) on criteria like conversation potential and viewer appeal
  3. Generator is fine-tuned to maximize critic scores
  4. Process repeats iteratively to optimize performance
- **Advantages**: Most adaptive approach, optimizes directly for engagement, requires no explicit labeled data
- **Challenges**: Most complex implementation, requires two LLMs, potential for filter bubbles or unwanted optimization directions

These advanced approaches would be considered for future development after the current implementation proves successful. Each offers progressively greater sophistication but requires correspondingly more resources to implement.
