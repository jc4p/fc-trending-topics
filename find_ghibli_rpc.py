from web3 import Web3
import time
import os
import sys
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()

# Get RPC URL from environment variables
BASE_RPC_URL = os.getenv("BASE_RPC_URL")

# Initialize Web3 connection
w3 = Web3(Web3.HTTPProvider(BASE_RPC_URL))
assert w3.is_connected(), "Failed to connect to Base node"

# Target parameters
TARGET_DATE = 1743552000  # March 26, 2025 (UNIX timestamp)
# Create search terms both as UTF-8 and as ASCII hex strings that might appear in bytecode
SEARCH_STRINGS = ["ghiblification", "$GHIBLI", "GHIBLI"]
SEARCH_TERMS = []

# Display and create all variants of search terms
print("Search terms (multiple encodings to catch all possibilities):")
for term in SEARCH_STRINGS:
    # 1. Direct UTF-8 bytes (standard string encoding)
    utf8_bytes = term.encode('utf-8')
    SEARCH_TERMS.append(utf8_bytes)
    print(f"  {term} (UTF-8): {utf8_bytes.hex()}")
    
    # 2. ASCII hex representation (each character represented as its hex value)
    # This is how strings sometimes appear in contract bytecode
    ascii_hex = ''.join(f'{ord(c):02x}' for c in term)
    ascii_hex_bytes = bytes.fromhex(ascii_hex)
    SEARCH_TERMS.append(ascii_hex_bytes)
    print(f"  {term} (ASCII hex): {ascii_hex_bytes.hex()}")
    
    # 3. String with zero-padding (common in Solidity strings)
    padded_term = term + '\0' * (32 - len(term) % 32)
    padded_bytes = padded_term.encode('utf-8')
    SEARCH_TERMS.append(padded_bytes)
    print(f"  {term} (Zero-padded): {padded_bytes.hex()[:40]}...")

async def find_ghibli_contract(days_to_search=1):
    # Calculate block range (Base: ~2s/block, 43200 blocks/day)
    latest_block = w3.eth.get_block('latest')
    blocks_per_day = 43200
    target_block = latest_block['number'] - int((latest_block['timestamp'] - TARGET_DATE)/2)
    
    from_block = max(0, target_block - (blocks_per_day * days_to_search // 2))
    to_block = min(latest_block['number'], target_block + (blocks_per_day * days_to_search // 2))
    
    print(f"Searching blocks {from_block}-{to_block} ({to_block-from_block} blocks)")

    # Scan blocks for contract creations
    contract_candidates = []
    
    # Create progress bar
    pbar = tqdm(
        range(from_block, to_block+1),
        desc="Scanning blocks",
        unit="block",
        ncols=100,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )
    
    for block_num in pbar:
        block = w3.eth.get_block(block_num, full_transactions=True)
        
        for tx in block.transactions:
            if tx['to'] is None:  # Contract creation transaction
                # Check for any of the search terms in input data
                tx_input = tx['input']
                found_terms = []
                
                for term in SEARCH_TERMS:
                    if term in tx_input:
                        # Determine which original term this is
                        term_index = 0
                        for i, original_term in enumerate(SEARCH_STRINGS):
                            # Check if this term is any of the variants of the original
                            if (original_term.encode('utf-8') == term or 
                                bytes.fromhex(''.join(f'{ord(c):02x}' for c in original_term)) == term or
                                (original_term + '\0' * (32 - len(original_term) % 32)).encode('utf-8').startswith(term)):
                                term_index = i
                                break
                                
                        # Get the original string for readability
                        original_term = SEARCH_STRINGS[term_index]
                        if original_term not in found_terms:
                            found_terms.append(original_term)
                            # Update progress bar description when we find something
                            pbar.set_description(f"Found match for '{original_term}' in block {block_num}")
                
                if found_terms:
                    # Calculate contract address
                    sender_address = tx['from']
                    nonce = w3.eth.get_transaction_count(sender_address, block_identifier=block_num)
                    contract_address = w3.eth.get_contract_address({
                        'from': sender_address,
                        'nonce': nonce
                    })
                    
                    contract_candidates.append({
                        'address': contract_address,
                        'tx_hash': tx['hash'].hex(),
                        'block': block_num,
                        'timestamp': block['timestamp'],
                        'matched_terms': found_terms
                    })

    return contract_candidates

if __name__ == "__main__":
    import asyncio
    import argparse
    
    # Set up command-line arguments
    parser = argparse.ArgumentParser(description='Search for Ghibli contracts on Base blockchain')
    parser.add_argument('--days', type=int, default=1, help='Number of days to search around the target date')
    args = parser.parse_args()
    
    # Run the search with specified parameters
    results = asyncio.run(find_ghibli_contract(days_to_search=args.days))
    print(f"\nFound {len(results)} potential contracts:")
    for contract in results:
        matched_terms_str = ", ".join(contract['matched_terms'])
        print(f"• {contract['address']} created at {time.ctime(contract['timestamp'])}")
        print(f"  TX: {contract['tx_hash']} | Block: {contract['block']}")
        print(f"  Matched terms: {matched_terms_str}\n")

