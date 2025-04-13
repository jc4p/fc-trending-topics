# Processing Large Datasets with Generator-Critic

This guide explains how to process large datasets (>100K posts) with the Generator-Critic approach by using checkpointing and chunking.

## Overview

The Generator-Critic approach has been optimized for large datasets with two key features:

1. **Checkpointing**: Save progress after each batch to resume processing if interrupted
2. **Chunking**: Process the dataset in manageable chunks and merge results

## Step 1: Divide and Process

First, break your large dataset into manageable chunks (20,000 posts works well):

```bash
# Process the first 20,000 posts
./run_generator_critic.sh --limit 20000 --output-prefix "chunk1"

# Process the next 20,000 posts
./run_generator_critic.sh --limit 20000 --offset 20000 --output-prefix "chunk2"

# Process the next 20,000 posts
./run_generator_critic.sh --limit 20000 --offset 40000 --output-prefix "chunk3"

# Continue as needed...
```

## Step 2: Merge Results

After processing all chunks, merge the results:

```bash
# Merge all chunks, keeping top 5 topics from each
python src/merge_gc_results.py --chunks chunk1,chunk2,chunk3 --output combined

# Or keep all topics without filtering
python src/merge_gc_results.py --chunks chunk1,chunk2,chunk3 --output combined --keep-all

# Or specify how many top topics to keep from each chunk
python src/merge_gc_results.py --chunks chunk1,chunk2,chunk3 --output combined --top 10
```

## Advanced Usage

### Handling Interruptions

If processing is interrupted, you can resume from where you left off:

```bash
# Resume processing the current chunk
./run_generator_critic.sh --resume --output-prefix "chunk1"
```

### Skipping Evaluation

For very large datasets, you can skip the evaluation phase for faster processing:

```bash
# Skip the critic model evaluation (uses default scores)
./run_generator_critic.sh --limit 20000 --output-prefix "chunk1" --skip-evaluation
```

### Batch Size Tuning

Adjust batch size based on your available memory:

```bash
# Use smaller batch size for lower memory usage
./run_generator_critic.sh --limit 20000 --batch-size 5000 --output-prefix "chunk1"

# Use larger batch size for faster processing
./run_generator_critic.sh --limit 20000 --batch-size 15000 --output-prefix "chunk1"
```

## Processing Strategy for Different Dataset Sizes

### Medium Datasets (20K-100K posts)

```bash
# Process in 2-5 chunks of 20,000 posts each
./run_generator_critic.sh --limit 20000 --output-prefix "chunk1"
./run_generator_critic.sh --limit 20000 --offset 20000 --output-prefix "chunk2"
# ... continue as needed

# Merge results
python src/merge_gc_results.py --chunks chunk1,chunk2 --output combined
```

### Large Datasets (100K-500K posts)

```bash
# Process in 5-25 chunks of 20,000 posts each
./run_generator_critic.sh --limit 20000 --output-prefix "chunk1"
./run_generator_critic.sh --limit 20000 --offset 20000 --output-prefix "chunk2"
# ... continue as needed

# Consider skipping evaluation for some chunks to speed up processing
./run_generator_critic.sh --limit 20000 --offset 40000 --output-prefix "chunk3" --skip-evaluation

# Merge results (keeping top topics from each chunk)
python src/merge_gc_results.py --chunks chunk1,chunk2,chunk3,... --output combined --top 3
```

### Very Large Datasets (500K+ posts)

For very large datasets, consider a sampling approach:

```bash
# Process evenly spaced samples throughout the dataset
./run_generator_critic.sh --limit 20000 --output-prefix "sample1"
./run_generator_critic.sh --limit 20000 --offset 100000 --output-prefix "sample2"
./run_generator_critic.sh --limit 20000 --offset 200000 --output-prefix "sample3"
# ... continue as needed

# Merge samples
python src/merge_gc_results.py --chunks sample1,sample2,sample3,... --output combined
```

## Performance Tips

1. **Memory Management**: Using batch size of 5,000-10,000 provides a good balance between speed and memory usage
2. **API Cost Control**: The critic model is expensive - use `--skip-evaluation` for initial exploratory runs
3. **Processing Time**: For a 20,000 post chunk, expect ~15-30 minutes depending on post complexity
4. **Error Recovery**: Always use unique output prefixes so you don't overwrite previous results