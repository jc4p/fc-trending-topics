from web3 import Web3
import time
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get RPC URL from environment variables
BASE_RPC_URL = os.getenv("BASE_RPC_URL")

# Initialize Web3 connection
w3 = Web3(Web3.HTTPProvider(BASE_RPC_URL))
assert w3.is_connected(), "Failed to connect to Base node"

# Target parameters
TARGET_DATE = 1743552000  # March 26, 2025 (UNIX timestamp)
# Search terms in hex
SEARCH_TERMS = [
    "676869626c696669636174696f6e",  # "ghiblification" in hex
    "24474849424c49",                # "$GHIBLI" in hex
    "474849424c49"                   # "GHIBLI" in hex
]

async def find_ghibli_contract():
    # Calculate block range (Base: ~2s/block, 43200 blocks/day)
    latest_block = w3.eth.get_block('latest')
    blocks_per_day = 43200
    target_block = latest_block['number'] - int((latest_block['timestamp'] - TARGET_DATE)/2)
    
    from_block = max(0, target_block - blocks_per_day)
    to_block = min(latest_block['number'], target_block + blocks_per_day)
    
    print(f"Searching blocks {from_block}-{to_block} ({to_block-from_block} blocks)")

    # Scan blocks for contract creations
    contract_candidates = []
    for block_num in range(from_block, to_block+1):
        block = w3.eth.get_block(block_num, full_transactions=True)
        
        for tx in block.transactions:
            if tx['to'] is None:  # Contract creation transaction
                # Check for any of the search terms in input data
                tx_input = tx['input'].lower()
                found_terms = []
                
                for term in SEARCH_TERMS:
                    if term in tx_input:
                        found_terms.append(term)
                
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
    results = asyncio.run(find_ghibli_contract())
    print(f"\nFound {len(results)} potential contracts:")
    for contract in results:
        matched_terms_str = ", ".join([
            term[:10] + "..." if len(term) > 10 else term
            for term in contract['matched_terms']
        ])
        print(f"• {contract['address']} created at {time.ctime(contract['timestamp'])}")
        print(f"  TX: {contract['tx_hash']} | Block: {contract['block']}")
        print(f"  Matched terms: {matched_terms_str}\n")

