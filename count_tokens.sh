#!/bin/bash
# Check token counts in debug files

# Function to count tokens (very rough estimate)
count_tokens() {
  # Count words and divide by 0.75 (rough approximation)
  wc -w "$1" | awk '{print int($1 / 0.75)}'
}

# Find the latest critic raw response
CRITIC_FILE=$(ls -t output/debug/critic_raw_response.txt 2>/dev/null | head -1)
if [ -n "$CRITIC_FILE" ]; then
  echo "Critic response (approximate tokens): $(count_tokens "$CRITIC_FILE")"
fi

# Find the topics JSON from the checkpoint file
CHECKPOINT=$(ls -t output/checkpoints/*generator_checkpoint.json 2>/dev/null | head -1)
if [ -n "$CHECKPOINT" ]; then
  # Extract just the topics array to a temp file
  jq '.batch_topics' "$CHECKPOINT" > /tmp/topics.json
  echo "Topics JSON size (approximate tokens): $(count_tokens "/tmp/topics.json")"
  echo "Number of topics: $(jq '. | length' /tmp/topics.json)"
fi

# Calculate total input tokens (very rough estimate)
BASE_PROMPT_TOKENS=1500
if [ -n "$CHECKPOINT" ]; then
  TOPICS_TOKENS=$(count_tokens "/tmp/topics.json")
  TOTAL_INPUT=$(($BASE_PROMPT_TOKENS + $TOPICS_TOKENS))
  echo "Estimated total input tokens to critic: $TOTAL_INPUT"
fi
