#!/bin/bash

# Simplified script to run the generator-critic approach on a single chunk of data

# Clean up old results
rm -f output/combined_approach4_results.json

# Run the analysis with a smaller dataset
python src/approach4_generator_critic.py --batch-size 5000 --limit 30000 --debug

echo "Simplified analysis completed!"