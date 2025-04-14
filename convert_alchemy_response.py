#!/usr/bin/env python3
"""
Convert Alchemy API price history response to the format expected by analyze_token_roi.py
"""

import json
import sys
import os
from datetime import datetime

def convert_file(input_file, output_file=None):
    """Convert Alchemy API response format to the format expected by analyze_token_roi.py"""
    
    # Load the input file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Extract relevant information
    token_address = data.get('address', 'unknown')
    price_data = data.get('data', [])
    
    # Create output structure
    output_data = {
        "token_address": token_address,
        "token_name": "Ghiblification",
        "token_symbol": "Ghibli",
        "current_price": price_data[-1]['value'] if price_data else "0",
        "chain": "solana",
        "prices": price_data,
        "exported_at": datetime.now().isoformat()
    }
    
    # Generate output filename if not provided
    if output_file is None:
        dirname = os.path.dirname(input_file)
        filename = os.path.basename(input_file)
        base_name = os.path.splitext(filename)[0]
        output_file = os.path.join(dirname, f"{base_name}_converted.json")
    
    # Write the output file
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Converted {len(price_data)} price points")
    print(f"Original file: {input_file}")
    print(f"Converted file: {output_file}")
    
    return output_file

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_alchemy_response.py <input_file> [output_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        converted_file = convert_file(input_file, output_file)
        print(f"✅ Successfully converted file to: {converted_file}")
    except Exception as e:
        print(f"❌ Error converting file: {e}")
        sys.exit(1)