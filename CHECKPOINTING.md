# Generator-Critic with Checkpointing

This document explains how to use the checkpointing feature in the Generator-Critic approach for processing large datasets.

## Overview

The Generator-Critic approach now supports checkpointing, which allows you to:

1. Process large datasets in manageable chunks
2. Stop and resume processing at any time
3. Break the work into phases (generation vs. evaluation)
4. Recover from errors without losing progress

## Using Checkpointing

### Basic Usage

```bash
# Run with default settings (20,000 posts limit, 10,000 batch size)
./run_generator_critic.sh

# Run with a larger dataset
./run_generator_critic.sh --limit 50000

# Resume from where you left off
./run_generator_critic.sh --resume

# Use a specific batch size
./run_generator_critic.sh --batch-size 5000
```

### Advanced Options

```bash
# Skip evaluation phase (uses default scores if no checkpoint)
./run_generator_critic.sh --skip-evaluation

# Use a specific output prefix for multiple runs
./run_generator_critic.sh --output-prefix "run1"

# Clean up checkpoint files after successful completion
./run_generator_critic.sh --cleanup-checkpoints
```

## How It Works

The checkpointing system works in two phases:

### 1. Generator Phase

- Processes data in batches of configurable size
- Saves all generated topics to a checkpoint file after each batch
- Tracks which batch to resume from

### 2. Evaluation Phase

- Evaluates all topics with the critic model
- Saves evaluated topics to a separate checkpoint file
- Can be skipped if you want to use default evaluations

## Processing Strategy for Large Datasets

For datasets larger than 20,000 posts, we recommend the following approach:

1. Process in chunks of 20,000 posts with unique output prefixes:
   ```bash
   ./run_generator_critic.sh --limit 20000 --output-prefix "chunk1"
   ./run_generator_critic.sh --limit 20000 --offset 20000 --output-prefix "chunk2"
   ```

2. Then combine the results manually or use a script:
   ```python
   # Example combining script
   import json
   
   # Load results from each chunk
   with open('output/chunk1_approach4_results.json', 'r') as f:
       chunk1 = json.load(f)
       
   with open('output/chunk2_approach4_results.json', 'r') as f:
       chunk2 = json.load(f)
   
   # Combine topics
   combined = {
       "topics": chunk1["topics"] + chunk2["topics"],
       "analysis_period": f"{chunk1['analysis_period']} and {chunk2['analysis_period']}",
       "total_posts_analyzed": chunk1["total_posts_analyzed"] + chunk2["total_posts_analyzed"]
   }
   
   # Save combined results
   with open('output/combined_approach4_results.json', 'w') as f:
       json.dump(combined, f, indent=2)
   ```

## Troubleshooting

- If you encounter an error during processing, simply run with the `--resume` flag to continue
- If the critic model is failing, use `--skip-evaluation` to bypass it and use default evaluations
- If checkpoints are becoming too large, consider using smaller batch sizes with `--batch-size`

## Performance Tips

- Batch size of 10,000 posts provides a good balance between speed and memory usage
- For faster processing, reduce batch size to 5,000 (more checkpoints but lower memory usage)
- The critic model is the bottleneck - if it's failing, try `--skip-evaluation` and evaluate offline