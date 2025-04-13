#!/bin/bash

# process_all_chunks.sh - Process entire dataset in chunks automatically
# This script will divide the dataset into chunks and process each one sequentially

# Load environment variables from .env file if it exists
if [ -f .env ]; then
  echo "Loading environment variables from .env file"
  export $(grep -v '^#' .env | xargs)
fi

# Check for required environment variable
if [ -z "$GOOGLE_API_KEY" ]; then
  echo "Error: GOOGLE_API_KEY environment variable not set"
  echo "Please set it with: export GOOGLE_API_KEY=your_api_key_here"
  exit 1
fi

# Default settings
CHUNK_SIZE=20000
BATCH_SIZE=10000
SKIP_EVAL=false
CLEANUP=false
BASE_PREFIX="chunk"
START_OFFSET=0
END_INDEX=0 # 0 means process the entire dataset

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --chunk-size)
      CHUNK_SIZE="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --start-offset)
      START_OFFSET="$2"
      shift 2
      ;;
    --end-index)
      END_INDEX="$2"
      shift 2
      ;;
    --prefix)
      BASE_PREFIX="$2"
      shift 2
      ;;
    --skip-evaluation)
      SKIP_EVAL=true
      shift
      ;;
    --cleanup-checkpoints)
      CLEANUP=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Create required directories
mkdir -p output/checkpoints

# First, determine total dataset size
echo "Determining total dataset size..."
TOTAL_SIZE=$(python -c "
import pandas as pd
try:
    parquet_path = 'output/interim_data/cleaned_data.parquet'
    total_size = pd.read_parquet(parquet_path, columns=['datetime']).shape[0]
    print(total_size)
except Exception as e:
    print(f'Error: {e}')
    print('0')
")

if [[ ! "$TOTAL_SIZE" =~ ^[0-9]+$ ]]; then
    echo "Error determining dataset size: $TOTAL_SIZE"
    exit 1
fi

if [ "$TOTAL_SIZE" -eq 0 ]; then
    echo "Error: Could not determine dataset size or dataset is empty"
    exit 1
fi

echo "Total dataset size: $TOTAL_SIZE posts"

# If end index is 0 or greater than total size, use total size
if [ "$END_INDEX" -eq 0 ] || [ "$END_INDEX" -gt "$TOTAL_SIZE" ]; then
    END_INDEX=$TOTAL_SIZE
    echo "Will process until the end of the dataset (index $END_INDEX)"
else
    echo "Will process until index $END_INDEX"
fi

# Calculate number of chunks needed
if [ "$START_OFFSET" -ge "$END_INDEX" ]; then
    echo "Error: Start offset ($START_OFFSET) must be less than end index ($END_INDEX)"
    exit 1
fi

REMAINING_POSTS=$((END_INDEX - START_OFFSET))
NUM_CHUNKS=$(( (REMAINING_POSTS + CHUNK_SIZE - 1) / CHUNK_SIZE ))

echo "Will process $REMAINING_POSTS posts in $NUM_CHUNKS chunks of $CHUNK_SIZE posts each"
echo "Starting from offset $START_OFFSET"
echo ""

# Process all chunks sequentially
CURRENT_OFFSET=$START_OFFSET
CHUNK_NUM=1
SUCCESSFUL_CHUNKS=()

while [ $CURRENT_OFFSET -lt $END_INDEX ]; do
    # Calculate actual chunk size for this iteration (might be smaller for the last chunk)
    REMAINING=$((END_INDEX - CURRENT_OFFSET))
    THIS_CHUNK_SIZE=$CHUNK_SIZE
    if [ $REMAINING -lt $CHUNK_SIZE ]; then
        THIS_CHUNK_SIZE=$REMAINING
    fi
    
    # Create unique prefix for this chunk
    CHUNK_PREFIX="${BASE_PREFIX}_${CHUNK_NUM}"
    
    echo "========================================================"
    echo "Processing chunk $CHUNK_NUM/$NUM_CHUNKS: $THIS_CHUNK_SIZE posts from offset $CURRENT_OFFSET"
    echo "Saving results with prefix: $CHUNK_PREFIX"
    echo "========================================================"
    
    # Build the command
    CMD="./run_generator_critic.sh --batch-size $BATCH_SIZE --limit $THIS_CHUNK_SIZE --offset $CURRENT_OFFSET --output-prefix $CHUNK_PREFIX"
    
    # Add optional flags
    if [ "$SKIP_EVAL" = true ]; then
        CMD="$CMD --skip-evaluation"
    fi
    if [ "$CLEANUP" = true ]; then
        CMD="$CMD --cleanup-checkpoints"
    fi
    
    # Run the command
    echo "Running: $CMD"
    eval $CMD
    
    # Check if successful
    if [ $? -eq 0 ]; then
        echo "Chunk $CHUNK_NUM completed successfully"
        SUCCESSFUL_CHUNKS+=("$CHUNK_PREFIX")
    else
        echo "Error processing chunk $CHUNK_NUM"
        echo "You can resume from this point with:"
        echo "./process_all_chunks.sh --start-offset $CURRENT_OFFSET --chunk-size $CHUNK_SIZE --prefix $BASE_PREFIX"
        exit 1
    fi
    
    # Move to next chunk
    CURRENT_OFFSET=$((CURRENT_OFFSET + THIS_CHUNK_SIZE))
    CHUNK_NUM=$((CHUNK_NUM + 1))
    
    echo ""
done

# All chunks completed successfully
echo "========================================================"
echo "ALL CHUNKS PROCESSED SUCCESSFULLY!"
echo "Processed $NUM_CHUNKS chunks covering $REMAINING_POSTS posts"
echo "========================================================"

# Print summary of successful chunks
echo "Successful chunks:"
for chunk in "${SUCCESSFUL_CHUNKS[@]}"; do
    echo " - $chunk"
done

# Ask if user wants to merge results
if [ ${#SUCCESSFUL_CHUNKS[@]} -gt 1 ]; then
    echo ""
    echo "Do you want to merge all chunks into a single result file? (y/n)"
    read -r MERGE_RESPONSE
    
    if [[ "$MERGE_RESPONSE" =~ ^[Yy]$ ]]; then
        # Build comma-separated list of chunk prefixes
        CHUNK_LIST=$(IFS=,; echo "${SUCCESSFUL_CHUNKS[*]}")
        
        # Run merge script
        echo "Merging chunks: $CHUNK_LIST"
        python src/merge_gc_results.py --chunks "$CHUNK_LIST" --output "${BASE_PREFIX}_combined"
        
        if [ $? -eq 0 ]; then
            echo "Results successfully merged to output/${BASE_PREFIX}_combined_approach4_results.json"
        else
            echo "Error merging results"
        fi
    else
        echo "Skipping merge step. You can merge later with:"
        echo "python src/merge_gc_results.py --chunks \"$CHUNK_LIST\" --output \"${BASE_PREFIX}_combined\""
    fi
fi

echo "Processing completed!"