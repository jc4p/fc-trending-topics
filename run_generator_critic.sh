#!/bin/bash

# run_generator_critic.sh - Script to run the generator-critic approach with checkpointing
# This version uses the single-run approach with checkpointing, which was reported to work
# perfectly with a 20,000 post limit

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

# Create required directories
mkdir -p output/checkpoints
mkdir -p output/debug

# Default values
BATCH_SIZE=10000
LIMIT=20000
OUTPUT_PREFIX=""
RESUME=false
SKIP_EVAL=false
CLEANUP=false

# Parse command line arguments
OFFSET=0

while [[ $# -gt 0 ]]; do
  case $1 in
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --offset)
      OFFSET="$2"
      shift 2
      ;;
    --output-prefix)
      OUTPUT_PREFIX="$2"
      shift 2
      ;;
    --resume)
      RESUME=true
      shift
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

# Build the command
CMD="python src/approach4_generator_critic.py --batch-size $BATCH_SIZE --limit $LIMIT"

# Add offset if specified (must be greater than 0)
if [ "$OFFSET" -gt 0 ]; then
  CMD="$CMD --offset $OFFSET"
fi

# Add optional arguments if specified
if [ -n "$OUTPUT_PREFIX" ]; then
  CMD="$CMD --output-prefix $OUTPUT_PREFIX"
fi

if [ "$RESUME" = true ]; then
  CMD="$CMD --resume"
fi

if [ "$SKIP_EVAL" = true ]; then
  CMD="$CMD --skip-evaluation"
fi

if [ "$CLEANUP" = true ]; then
  CMD="$CMD --cleanup-checkpoints"
fi

# Print the command that will be executed
echo "Running: $CMD"
echo "Press Ctrl+C to stop (state will be saved in checkpoints)"
echo "==================="

# Execute the command
eval $CMD

# Check the exit status
STATUS=$?
if [ $STATUS -eq 0 ]; then
  echo "==================="
  echo "Process completed successfully"
else
  echo "==================="
  echo "Process exited with status $STATUS"
  echo "You can resume where you left off with: $CMD --resume"
fi

exit $STATUS