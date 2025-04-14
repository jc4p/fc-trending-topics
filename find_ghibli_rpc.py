from web3 import Web3
import time
import os
import sys
import json
import requests
import asyncio
import datetime
import base58
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple, Union

# Load environment variables from .env file
load_dotenv()

# Get RPC URLs and API keys from environment variables
BASE_RPC_URL = os.getenv("BASE_RPC_URL")
ETH_RPC_URL = os.getenv("ETH_RPC_URL", "https://eth-mainnet.g.alchemy.com/v2/" + os.getenv("ALCHEMY_API_KEY", ""))
SOLANA_RPC_URL = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
ALCHEMY_API_KEY = os.getenv("ALCHEMY_API_KEY")
COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY", "")
BIRDEYE_API_KEY = os.getenv("BIRDEYE_API_KEY", "")
DEXSCREENER_URL = "https://api.dexscreener.com/latest/dex/tokens"

# Initialize connections
connections = {}

# Base chain connection
try:
    base_web3 = Web3(Web3.HTTPProvider(BASE_RPC_URL))
    if base_web3.is_connected():
        print("✅ Connected to Base blockchain")
        connections["base"] = base_web3
    else:
        print("❌ Failed to connect to Base blockchain")
except Exception as e:
    print(f"❌ Error connecting to Base: {e}")

# Ethereum connection
try:
    eth_web3 = Web3(Web3.HTTPProvider(ETH_RPC_URL))
    if eth_web3.is_connected():
        print("✅ Connected to Ethereum blockchain")
        connections["ethereum"] = eth_web3
    else:
        print("❌ Failed to connect to Ethereum blockchain")
except Exception as e:
    print(f"❌ Error connecting to Ethereum: {e}")

# Set default web3 for backwards compatibility
w3 = connections.get("base", None)

# Target parameters
TARGET_DATE_STR = "2025-03-26"  # Target date in YYYY-MM-DD format

# Default block numbers for March 26, 2025 (pre-determined)
BLOCKS = {
    "base": {
        "target": 28091515,  # March 26, 2025 00:00:00 UTC
    },
    "ethereum": {
        "target": 20000000,  # Estimated, would need to be updated
    }
}

# Solana specific info
SOLANA_TOKENS = {
    "ghibli": "4TBi66vi32S7J8X1A6eWfaLHYmUXu7CStcEmsJQdpump"  # Coin address from pump.fun
}

# Create search terms both as UTF-8 and as ASCII hex strings that might appear in bytecode
SEARCH_STRINGS = ["ghiblification", "$GHIBLI", "GHIBLI"]
SEARCH_TERMS = []

# No hardcoded token symbols - we'll get them dynamically from DEX data

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

def get_block_by_timestamp(timestamp, chain="base", debug=False):
    """Use Alchemy API to get the closest block number for a given timestamp"""
    
    # Create the JSON-RPC payload
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "alchemy_getBlockByTimestamp",
        "params": [
            hex(timestamp),
            False  # Don't include transactions
        ]
    }
    
    # Send the request to Alchemy API
    headers = {
        "Content-Type": "application/json"
    }
    
    # Get the appropriate RPC URL for the chain
    if chain == "base":
        rpc_url = BASE_RPC_URL
    elif chain == "ethereum":
        rpc_url = ETH_RPC_URL
    else:
        print(f"Warning: Unsupported chain {chain}")
        return None

    # Check if using Alchemy
    if "alchemy.com" in rpc_url:
        # Use the domain as is
        alchemy_url = rpc_url
    else:
        # If not using Alchemy, this won't work
        print(f"Warning: Not using Alchemy provider for {chain}, falling back to estimation")
        return None
    
    if debug:
        print(f"\nDEBUG: Sending request to {alchemy_url}")
        print(f"DEBUG: Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(alchemy_url, headers=headers, data=json.dumps(payload))
        
        if debug:
            print(f"DEBUG: Response status: {response.status_code}")
            print(f"DEBUG: Response text: {response.text[:500]}...")
        
        response_json = response.json()
        
        # Check for errors
        if "error" in response_json:
            error = response_json["error"]
            print(f"API Error: {error.get('message', 'Unknown error')}, Code: {error.get('code', 'N/A')}")
            return None
            
        result = response_json.get("result")
        if result:
            block_num = int(result.get("number", 0), 16)
            if debug:
                print(f"DEBUG: Found block number: {block_num}")
            return block_num
        else:
            if debug:
                print("DEBUG: No result in response")
            return None
    except Exception as e:
        print(f"Error getting block by timestamp: {e}")
        return None

async def find_evm_contracts(chain="base", search_terms=None, days_to_search=1, 
                        quickscan=False, step_size=10, batch_size=10, 
                        target_date_str=None, from_block=None, to_block=None):
    """Search for contracts containing specific terms on an EVM chain (Ethereum, Base, etc.)"""
    # If no search terms provided, use the default ones
    if search_terms is None:
        search_terms = SEARCH_TERMS
        
    # Get the appropriate Web3 connection
    web3 = connections.get(chain)
    if not web3:
        print(f"⛔️ No connection to {chain} blockchain")
        return []
    
    # If blocks are already specified, use them
    if from_block is not None and to_block is not None:
        print(f"Using user-specified block range: {from_block}-{to_block}")
        target_block = (from_block + to_block) // 2
    else:
        # Convert target date to timestamp
        date_to_use = target_date_str if target_date_str else TARGET_DATE_STR
        
        if date_to_use == TARGET_DATE_STR and chain in BLOCKS:
            # Use pre-determined block
            target_block = BLOCKS[chain]["target"]
            print(f"Using pre-determined block {target_block} for {TARGET_DATE_STR} on {chain}")
        else:
            target_date = datetime.datetime.strptime(date_to_use, "%Y-%m-%d")
            target_timestamp = int(target_date.timestamp())
            
            # Get blocks for the date range using binary search
            print(f"Finding exact blocks for date range around {date_to_use} on {chain}...")
            target_block = get_block_for_date_utility(date_to_use, chain=chain, debug=False)
            
            if target_block is None:
                # Fallback to estimation if all methods fail
                print(f"Falling back to block estimation method for {chain}...")
                latest_block = web3.eth.get_block('latest')
                blocks_per_day = 43200 if chain == "base" else 7200  # ~2s/block on Base, ~12s/block on Ethereum
                target_timestamp = int(target_date.timestamp())
                target_block = latest_block.number - int((latest_block.timestamp - target_timestamp) / (2 if chain == "base" else 12))
        
        # Calculate range based on days around target
        blocks_per_day = 43200 if chain == "base" else 7200
        from_block = max(0, target_block - (blocks_per_day * days_to_search // 2))
        to_block = min(web3.eth.get_block('latest').number, target_block + (blocks_per_day * days_to_search // 2))
        
        # For faster scan, we can focus more closely around the target date first
        if quickscan and days_to_search > 1:
            # First scan just ±12 hours around target
            half_day_blocks = blocks_per_day // 2
            quick_from_block = max(0, target_block - half_day_blocks)
            quick_to_block = min(web3.eth.get_block('latest').number, target_block + half_day_blocks)
            from_block, to_block = quick_from_block, quick_to_block
    
    scan_type = "QUICKSCAN" if quickscan else "FULLSCAN"
    print(f"{scan_type} searching blocks {from_block}-{to_block} ({to_block-from_block} blocks) on {chain}")

    # Function to process a single block
    def process_block(block_num):
        try:
            results = []
            block = web3.eth.get_block(block_num, full_transactions=True)
            
            for tx in block.transactions:
                if tx.to is None:  # Contract creation transaction
                    # Check for any of the search terms in input data
                    tx_input = tx.input
                    found_terms = []
                    
                    for term in search_terms:
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
                    
                    if found_terms:
                        # Calculate contract address
                        sender_address = tx['from']
                        nonce = web3.eth.get_transaction_count(sender_address, block_identifier=block_num)
                        contract_address = web3.eth.get_contract_address({
                            'from': sender_address,
                            'nonce': nonce
                        })
                        
                        results.append({
                            'chain': chain,
                            'address': contract_address,
                            'tx_hash': tx.hash.hex(),
                            'block': block_num,
                            'timestamp': block.timestamp,
                            'matched_terms': found_terms
                        })
            return results
        except Exception as e:
            print(f"\nError processing block {block_num} on {chain}: {str(e)}")
            return []
    
    # Create progress bar
    block_range = list(range(from_block, to_block+1, step_size))
    pbar = tqdm(
        total=len(block_range),
        desc=f"{scan_type} on {chain} (step={step_size}, batch={batch_size})",
        unit="batch",
        ncols=100,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )
    
    contract_candidates = []
    
    # Process blocks in batches using ThreadPoolExecutor for parallelism
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        # Process blocks in batches
        for i in range(0, len(block_range), batch_size):
            batch_blocks = block_range[i:i+batch_size]
            batch_results = list(executor.map(process_block, batch_blocks))
            
            # Update progress
            pbar.update(len(batch_blocks))
            
            # Collect results
            for block_result in batch_results:
                if block_result:
                    for result in block_result:
                        contract_candidates.append(result)
                        # Update progress description if we found something
                        matched_terms = ", ".join(result['matched_terms'])
                        pbar.set_description(f"Found match for '{matched_terms}' in {chain} block {result['block']}")

    return contract_candidates

def get_token_price_history(token_address, chain="solana", start_date=None, end_date=None, interval="1h", debug=False, symbol=None):
    """Get historical price data for a token using Alchemy Price API"""
    # Set default dates if not provided
    if not start_date:
        # Default to 7 days ago
        start_date = (datetime.datetime.now() - datetime.timedelta(days=7)).strftime("%Y-%m-%dT00:00:00Z")
    if not end_date:
        # Default to now
        end_date = datetime.datetime.now().strftime("%Y-%m-%dT23:59:59Z")
    
    # Alchemy API key - use the one from environment or the demo key
    api_key = ALCHEMY_API_KEY if ALCHEMY_API_KEY else "docs-demo"
    
    # Alchemy Price API URL
    url = f"https://api.g.alchemy.com/prices/v1/{api_key}/tokens/historical"
    
    # Get chain ID
    if chain.lower() == "solana":
        chain_id = "solana-mainnet"  # Correct network ID for Solana
    elif chain.lower() == "base":
        chain_id = "base-mainnet"
    elif chain.lower() == "ethereum":
        chain_id = "eth-mainnet"
    else:
        chain_id = "eth-mainnet"  # Default
    
    # According to Alchemy API, we need to use either (network+address) OR symbol, not both
    price_data = None
    
    # Try first with network and address
    payload1 = {
        "network": chain_id,
        "address": token_address,
        "startTime": start_date,
        "endTime": end_date,
        "interval": interval
    }
    
    if debug:
        print(f"ATTEMPT 1: Using network '{chain_id}' and address '{token_address}'")
    
    try:
        # First try with network+address
        response = requests.post(url, json=payload1, headers={
            "accept": "application/json",
            "content-type": "application/json"
        })
        result = response.json()
        
        if debug:
            print(f"Response status: {response.status_code}")
            if response.status_code != 200:
                print(f"Error response: {json.dumps(result, indent=2)}")
        
        if response.status_code == 200 and "data" in result:
            return result["data"]
    except Exception as e:
        if debug:
            print(f"Error in first attempt: {e}")
    
    # If symbol is provided, try second attempt with just symbol
    if symbol:
        payload2 = {
            "symbol": symbol,
            "startTime": start_date,
            "endTime": end_date,
            "interval": interval
        }
        
        if debug:
            print(f"\nATTEMPT 2: Using symbol '{symbol}'")
        
        try:
            response = requests.post(url, json=payload2, headers={
                "accept": "application/json",
                "content-type": "application/json"
            })
            result = response.json()
            
            if debug:
                print(f"Response status: {response.status_code}")
                if response.status_code != 200:
                    print(f"Error response: {json.dumps(result, indent=2)}")
            
            if response.status_code == 200 and "data" in result:
                return result["data"]
        except Exception as e:
            if debug:
                print(f"Error in second attempt: {e}")
    
    # If both attempts failed, return error
    print(f"❌ Failed to get price history: All attempts failed")
    return None

def analyze_price_data(price_data):
    """Analyze price data to extract key metrics"""
    if not price_data:
        return {
            "success": False,
            "message": "No price data available"
        }
        
    # Normalize data structure - sometimes it's directly a list, sometimes inside a "prices" key
    prices_list = None
    if isinstance(price_data, list):
        prices_list = price_data
        print(f"Price data is a list with {len(prices_list)} entries")
    elif isinstance(price_data, dict) and "prices" in price_data:
        prices_list = price_data["prices"]
        print(f"Price data is a dict with {len(prices_list)} price entries")
    else:
        return {
            "success": False,
            "message": f"Unexpected price data format: {type(price_data)}"
        }
    
    try:
        # Convert to pandas DataFrame for analysis
        df = pd.DataFrame(prices_list)
        
        # Convert timestamp to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Sort by timestamp
        df = df.sort_values("timestamp")
        
        # Calculate key metrics
        highest_price = df["value"].max()
        highest_price_date = df.loc[df["value"].idxmax(), "timestamp"]
        
        lowest_price = df["value"].min()
        lowest_price_date = df.loc[df["value"].idxmin(), "timestamp"]
        
        # Calculate price changes for each interval
        df["price_change"] = df["value"].diff()
        df["price_change_pct"] = df["value"].pct_change() * 100
        
        # Find largest price movements
        largest_increase = df["price_change"].max()
        largest_increase_date = df.loc[df["price_change"].idxmax(), "timestamp"]
        
        largest_decrease = df["price_change"].min()
        largest_decrease_date = df.loc[df["price_change"].idxmin(), "timestamp"]
        
        # Calculate largest percentage movements
        largest_increase_pct = df["price_change_pct"].max()
        largest_increase_pct_date = df.loc[df["price_change_pct"].idxmax(), "timestamp"]
        
        largest_decrease_pct = df["price_change_pct"].min()
        largest_decrease_pct_date = df.loc[df["price_change_pct"].idxmin(), "timestamp"]
        
        # Calculate market cap if available
        market_cap_available = "marketCap" in df.columns
        if market_cap_available:
            highest_mcap = df["marketCap"].max()
            highest_mcap_date = df.loc[df["marketCap"].idxmax(), "timestamp"]
            
            lowest_mcap = df["marketCap"].min()
            lowest_mcap_date = df.loc[df["marketCap"].idxmin(), "timestamp"]
            
            latest_mcap = df.iloc[-1]["marketCap"] if not pd.isna(df.iloc[-1]["marketCap"]) else None
        else:
            highest_mcap = lowest_mcap = latest_mcap = None
            highest_mcap_date = lowest_mcap_date = None
        
        # Get current/latest price
        latest_price = df.iloc[-1]["value"]
        
        # Calculate volatility (standard deviation of percentage changes)
        volatility = df["price_change_pct"].std()
        
        # Prepare result
        result = {
            "success": True,
            "token_symbol": price_data.get("symbol", "Unknown") if isinstance(price_data, dict) else "Unknown",
            "data_points": len(df),
            "date_range": {
                "start": df.iloc[0]["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                "end": df.iloc[-1]["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            },
            "current_price": latest_price,
            "price_metrics": {
                "highest": {
                    "value": highest_price,
                    "date": highest_price_date.strftime("%Y-%m-%d %H:%M:%S")
                },
                "lowest": {
                    "value": lowest_price,
                    "date": lowest_price_date.strftime("%Y-%m-%d %H:%M:%S")
                }
            },
            "largest_movements": {
                "increase": {
                    "absolute": largest_increase,
                    "date": largest_increase_date.strftime("%Y-%m-%d %H:%M:%S"),
                    "percent": largest_increase_pct,
                    "percent_date": largest_increase_pct_date.strftime("%Y-%m-%d %H:%M:%S")
                },
                "decrease": {
                    "absolute": largest_decrease,
                    "date": largest_decrease_date.strftime("%Y-%m-%d %H:%M:%S"),
                    "percent": largest_decrease_pct,
                    "percent_date": largest_decrease_pct_date.strftime("%Y-%m-%d %H:%M:%S")
                }
            },
            "volatility": volatility
        }
        
        # Add market cap data if available
        if market_cap_available and highest_mcap is not None:
            result["market_cap"] = {
                "current": latest_mcap,
                "highest": {
                    "value": highest_mcap,
                    "date": highest_mcap_date.strftime("%Y-%m-%d %H:%M:%S")
                },
                "lowest": {
                    "value": lowest_mcap,
                    "date": lowest_mcap_date.strftime("%Y-%m-%d %H:%M:%S")
                }
            }
        
        return result
    except Exception as e:
        print(f"Error analyzing price data: {e}")
        return {
            "success": False,
            "message": f"Error analyzing price data: {str(e)}"
        }

def check_dexscreener_token(token_address, chain="solana"):
    """Get token information from DexScreener API"""
    # Special handling for pump.fun Solana tokens (addresses ending with 'pump')
    is_pump_fun_token = False
    if chain.lower() == "solana" and token_address.endswith("pump"):
        is_pump_fun_token = True
        print(f"Detected pump.fun token: {token_address}")
    
    url = f"{DEXSCREENER_URL}/{token_address}"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if "pairs" in data and len(data["pairs"]) > 0:
                # Filter pairs by chain if specified
                if chain:
                    pairs = [p for p in data["pairs"] if p.get("chainId", "").lower() == chain.lower()]
                else:
                    pairs = data["pairs"]
                
                if pairs:
                    # Get the main pair (usually the one with the highest liquidity)
                    main_pair = max(pairs, key=lambda x: float(x.get("liquidity", {}).get("usd", 0)))
                    
                    return {
                        "success": True,
                        "token_name": main_pair.get("baseToken", {}).get("name"),
                        "token_symbol": main_pair.get("baseToken", {}).get("symbol"),
                        "price_usd": main_pair.get("priceUsd"),
                        "price_change_24h": main_pair.get("priceChange", {}).get("h24"),
                        "volume_24h": main_pair.get("volume", {}).get("h24"),
                        "liquidity_usd": main_pair.get("liquidity", {}).get("usd"),
                        "fdv": main_pair.get("fdv"),
                        "pair_address": main_pair.get("pairAddress"),
                        "dex_id": main_pair.get("dexId"),
                        "chain_id": main_pair.get("chainId"),
                        "all_pairs": pairs
                    }
            
            return {
                "success": False,
                "message": "No pairs found for token"
            }
        else:
            return {
                "success": False,
                "message": f"DexScreener API error: {response.status_code}"
            }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error querying DexScreener: {str(e)}"
        }

def check_solana_token(token_address, debug=False):
    """Check information about a Solana token"""
    # API endpoint for Solana token information
    sol_url = f"{SOLANA_RPC_URL}"
    
    # Convert token address if needed
    try:
        if len(token_address) == 44 and token_address.startswith("4"):
            # Already in the correct format
            pass
        else:
            # Try to decode as base58
            token_address = base58.b58encode(bytes.fromhex(token_address)).decode()
    except Exception as e:
        if debug:
            print(f"Error processing token address: {e}")
    
    # 1. Get token account info
    get_account_payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getAccountInfo",
        "params": [
            token_address,
            {"encoding": "jsonParsed"}
        ]
    }
    
    token_result = {
        "chain": "solana",
        "address": token_address,
        "exists": False
    }
    
    try:
        print(f"Checking Solana token: {token_address}")
        headers = {"Content-Type": "application/json"}
        
        if debug:
            print(f"Sending request to {sol_url}:")
            print(json.dumps(get_account_payload, indent=2))
            
        response = requests.post(sol_url, headers=headers, json=get_account_payload)
        result = response.json()
        
        if debug:
            print("RPC Response:")
            print(json.dumps(result, indent=2)[:500] + "...")
            
        if "result" in result and result["result"]:
            account_data = result["result"]["value"]
            print(f"✅ Found Solana token account")
            
            token_result.update({
                "exists": True,
                "data": account_data
            })
            
            # 2. Get market data
            print("Getting market data from DexScreener...")
            dex_data = check_dexscreener_token(token_address, "solana")
            
            if dex_data["success"]:
                print(f"✅ Found market data from DexScreener")
                token_result.update({
                    "name": dex_data.get("token_name"),
                    "symbol": dex_data.get("token_symbol"),
                    "current_price": dex_data.get("price_usd"),
                    "price_change_24h": dex_data.get("price_change_24h"),
                    "volume_24h": dex_data.get("volume_24h"),
                    "liquidity": dex_data.get("liquidity_usd"),
                    "fully_diluted_value": dex_data.get("fdv"),
                    "dex": dex_data.get("dex_id"),
                    "all_pairs": dex_data.get("all_pairs")
                })
            
            # 3. Get price history
            try:
                print("Getting price history from Alchemy...")
                # Get 7-day historical data with 1-hour intervals
                start_date = (datetime.datetime.now() - datetime.timedelta(days=7)).strftime("%Y-%m-%dT00:00:00Z")
                end_date = datetime.datetime.now().strftime("%Y-%m-%dT23:59:59Z")
                
                # Extract symbol from dex data if available
                token_symbol = None
                if dex_data["success"] and "token_symbol" in dex_data:
                    token_symbol = dex_data["token_symbol"]
                    # Special handling for pump.fun Solana tokens
                    if token_address.endswith("pump"):
                        print(f"Using token symbol from pump.fun: {token_symbol}")
                
                price_history = get_token_price_history(
                    token_address, 
                    chain="solana", 
                    start_date=start_date, 
                    end_date=end_date, 
                    interval="1h",
                    debug=debug,
                    symbol=token_symbol
                )
                
                if price_history:
                    print(f"✅ Found price history data")
                    price_analysis = analyze_price_data(price_history)
                    if price_analysis["success"]:
                        token_result["price_analysis"] = price_analysis
            except Exception as e:
                print(f"Error getting price history: {e}")
            
            return token_result
        else:
            print(f"❌ No account found for address: {token_address}")
            return token_result
            
    except Exception as e:
        print(f"Error checking Solana token: {e}")
        token_result["error"] = str(e)
        return token_result

def get_block_for_date_utility(date_str, time_str=None, chain="base", debug=True):
    """Utility function to get block number for a specific date and time"""
    # Parse date (and time if provided)
    if time_str:
        # Parse date with time
        dt = datetime.datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
    else:
        # Parse just the date (use midnight)
        dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    
    # Convert to timestamp
    timestamp = int(dt.timestamp())
    
    print(f"\nLooking up block for {dt} (UTC) - Timestamp: {timestamp} on {chain}")
    
    # Try Alchemy method first
    print("\nMethod 1: Using Alchemy getBlockByTimestamp...")
    block_num = get_block_by_timestamp(timestamp, chain=chain, debug=debug)
    
    if block_num:
        # Get actual block time
        web3 = connections.get(chain)
        if not web3:
            print(f"No connection to {chain}")
            return None
            
        block = web3.eth.get_block(block_num)
        block_time = datetime.datetime.fromtimestamp(block.timestamp)
        
        print(f"\nResults:")
        print(f"Date/Time (UTC): {dt}")
        print(f"Unix Timestamp: {timestamp}")
        print(f"Block Number: {block_num}")
        print(f"Block Time: {block_time} (UTC)")
        print(f"Time Difference: {(block_time - dt).total_seconds()} seconds")
        return block_num
    else:
        print("\nMethod 1 failed. Trying alternative approach...")
        
        # Method 2: Binary search for the closest block
        print("\nMethod 2: Binary search for closest block...")
        
        try:
            web3 = connections.get(chain)
            if not web3:
                print(f"No connection to {chain}")
                return None
                
            # Get latest block
            latest_block = web3.eth.get_block('latest')
            latest_num = latest_block.number
            latest_time = latest_block.timestamp
            
            print(f"Latest block: {latest_num}, time: {datetime.datetime.fromtimestamp(latest_time)} (UTC)")
            
            if timestamp > latest_time:
                print("Error: Requested time is in the future!")
                return None
                
            # Get a rough estimate
            blocks_per_day = 43200 if chain == "base" else 7200  # ~2s/block on Base, ~12s/block on ETH
            days_diff = (latest_time - timestamp) / 86400  # seconds in a day
            estimated_blocks_back = int(days_diff * blocks_per_day)
            
            if estimated_blocks_back > latest_num:
                estimated_blocks_back = latest_num // 2
                
            estimated_block_num = latest_num - estimated_blocks_back
            
            print(f"Estimated starting block: {estimated_block_num}")
            
            # Binary search for the right block
            low = 1
            high = latest_num
            closest_block = None
            closest_diff = float('inf')
            
            # Limit to 20 iterations
            for i in range(20):
                if low > high:
                    break
                    
                mid = (low + high) // 2
                mid_block = web3.eth.get_block(mid)
                mid_time = mid_block.timestamp
                
                print(f"  Checking block {mid}: time={datetime.datetime.fromtimestamp(mid_time)} (UTC)")
                
                # Keep track of closest block seen
                time_diff = abs(mid_time - timestamp)
                if time_diff < closest_diff:
                    closest_diff = time_diff
                    closest_block = mid_block
                
                if mid_time < timestamp:
                    low = mid + 1
                elif mid_time > timestamp:
                    high = mid - 1
                else:
                    # Exact match
                    closest_block = mid_block
                    break
            
            if closest_block:
                block_time = datetime.datetime.fromtimestamp(closest_block.timestamp)
                
                print(f"\nResults (Method 2):")
                print(f"Date/Time (UTC): {dt}")
                print(f"Unix Timestamp: {timestamp}")
                print(f"Block Number: {closest_block.number}")
                print(f"Block Time: {block_time} (UTC)")
                print(f"Time Difference: {abs(closest_block.timestamp - timestamp)} seconds")
                return closest_block.number
                
        except Exception as e:
            print(f"Error during binary search: {e}")
        
        print("\nAll methods failed. Check your connection or try a different date.")
        return None

async def search_all_chains(days_to_search=1, quickscan=False, step_size=50, batch_size=5, 
                           target_date_str=None, from_block=None, to_block=None, 
                           include_solana=True, analyze_market=True):
    """Search for Ghibli-related contracts and tokens across all supported chains"""
    all_results = []
    
    # Search EVM chains (Base and Ethereum) in parallel
    evm_tasks = []
    for chain in connections.keys():
        task = find_evm_contracts(
            chain=chain,
            days_to_search=days_to_search,
            quickscan=quickscan,
            step_size=step_size,
            batch_size=batch_size,
            target_date_str=target_date_str,
            from_block=from_block if chain == "base" else None,
            to_block=to_block if chain == "base" else None
        )
        evm_tasks.append(task)
    
    # Wait for all EVM chain searches to complete
    evm_results = await asyncio.gather(*evm_tasks)
    
    # Combine EVM results
    for chain_results in evm_results:
        all_results.extend(chain_results)
    
    # Check for Solana tokens if enabled
    if include_solana:
        for token_name, token_address in SOLANA_TOKENS.items():
            print(f"\n{'='*80}")
            print(f"ANALYZING SOLANA TOKEN: {token_name.upper()} ({token_address})")
            print(f"{'='*80}")
            
            solana_result = check_solana_token(token_address, debug=True)
            
            # Add to results
            if solana_result["exists"]:
                result_entry = {
                    "chain": "solana",
                    "token_name": token_name,
                    "address": token_address,
                    "type": "token",
                    "matched_terms": [token_name.upper()],
                    "data": solana_result
                }
                
                # Include market data if available and requested
                if analyze_market and "price_analysis" in solana_result:
                    result_entry["market_data"] = solana_result["price_analysis"]
                
                all_results.append(result_entry)
    
    return all_results

if __name__ == "__main__":
    import asyncio
    import argparse
    
    # Set up command-line arguments
    parser = argparse.ArgumentParser(description='Search for Ghibli contracts and tokens across multiple blockchains')
    parser.add_argument('--days', type=int, default=1, help='Number of days to search around the target date')
    parser.add_argument('--date', type=str, help=f'Target date in YYYY-MM-DD format (default: {TARGET_DATE_STR})')
    parser.add_argument('--quickscan', action='store_true', help='Use larger block steps for faster initial scan')
    parser.add_argument('--step', type=int, default=50, help='Block step size for quickscan (default: 50)')
    parser.add_argument('--batch', type=int, default=5, help='Number of blocks to process in parallel (default: 5)')
    parser.add_argument('--from-block', type=int, help='Start block number (overrides date, Base chain only)')
    parser.add_argument('--to-block', type=int, help='End block number (overrides date, Base chain only)')
    parser.add_argument('--find-block', type=str, help='Just find block number for date (YYYY-MM-DD)')
    parser.add_argument('--time', type=str, help='Time for --find-block (HH:MM:SS), defaults to midnight')
    parser.add_argument('--chain', type=str, choices=['base', 'ethereum', 'solana'], default='base',
                        help='Blockchain to use for find-block operation')
    parser.add_argument('--no-solana', action='store_true', help='Skip Solana token checks')
    parser.add_argument('--analyze-market', action='store_true', help='Analyze token market data and price history')
    parser.add_argument('--token-address', type=str, help='Specific token address to analyze (especially for Solana)')
    parser.add_argument('--history-days', type=int, default=7, help='Number of days of price history to analyze (default: 7)')
    parser.add_argument('--interval', type=str, default='1h', choices=['5m', '1h', '4h', '1d'], 
                        help='Price history interval (5m, 1h, 4h, 1d)')
    args = parser.parse_args()
    
    # If a specific token address is provided for analysis
    if args.token_address:
        print(f"\nAnalyzing specific token: {args.token_address} on {args.chain}")
        
        if args.chain == "solana":
            # Add to our known tokens if not already there
            if args.token_address not in [addr for addr in SOLANA_TOKENS.values()]:
                SOLANA_TOKENS["custom"] = args.token_address
            
            # Direct token analysis
            token_result = check_solana_token(args.token_address, debug=True)
            
            if token_result["exists"]:
                print("\n" + "="*80)
                print(f"TOKEN ANALYSIS RESULTS")
                print("="*80)
                
                # Display token info
                if "name" in token_result:
                    print(f"Name: {token_result['name']}")
                if "symbol" in token_result:
                    print(f"Symbol: {token_result['symbol']}")
                if "current_price" in token_result:
                    print(f"Current Price: ${token_result['current_price']}")
                if "price_change_24h" in token_result:
                    print(f"24h Change: {token_result['price_change_24h']}%")
                if "volume_24h" in token_result:
                    print(f"24h Volume: ${token_result['volume_24h']}")
                if "liquidity" in token_result:
                    print(f"Liquidity: ${token_result['liquidity']}")
                if "fully_diluted_value" in token_result:
                    print(f"Fully Diluted Value: ${token_result['fully_diluted_value']}")
                if "dex" in token_result:
                    print(f"DEX: {token_result['dex']}")
                
                # Display price analysis if available
                if "price_analysis" in token_result and token_result["price_analysis"]["success"]:
                    pa = token_result["price_analysis"]
                    print("\n" + "-"*80)
                    print(f"PRICE HISTORY ANALYSIS ({pa['data_points']} data points)")
                    print(f"Period: {pa['date_range']['start']} to {pa['date_range']['end']}")
                    print("-"*80)
                    
                    print(f"\nPrice Range:")
                    print(f"  Highest: ${pa['price_metrics']['highest']['value']} on {pa['price_metrics']['highest']['date']}")
                    print(f"  Lowest: ${pa['price_metrics']['lowest']['value']} on {pa['price_metrics']['lowest']['date']}")
                    print(f"  Current: ${pa['current_price']}")
                    
                    print(f"\nLargest Movements:")
                    print(f"  Largest Increase: ${pa['largest_movements']['increase']['absolute']} " + 
                          f"({pa['largest_movements']['increase']['percent']:.2f}%) on {pa['largest_movements']['increase']['date']}")
                    print(f"  Largest Decrease: ${pa['largest_movements']['decrease']['absolute']} " + 
                          f"({pa['largest_movements']['decrease']['percent']:.2f}%) on {pa['largest_movements']['decrease']['date']}")
                    
                    print(f"\nVolatility: {pa['volatility']:.2f}%")
                    
                    if "market_cap" in pa:
                        print(f"\nMarket Cap:")
                        print(f"  Highest: ${pa['market_cap']['highest']['value']} on {pa['market_cap']['highest']['date']}")
                        print(f"  Lowest: ${pa['market_cap']['lowest']['value']} on {pa['market_cap']['lowest']['date']}")
                        print(f"  Current: ${pa['market_cap']['current']}")
            
            # Exit after token analysis
            sys.exit(0)
    
    # If using find-block mode, just find the block and exit
    if args.find_block:
        get_block_for_date_utility(args.find_block, args.time, chain=args.chain)
        sys.exit(0)
    
    # Override target date if specified
    if args.date:
        print(f"Using target date: {args.date}")
    
    # Run the search with specified parameters
    results = asyncio.run(search_all_chains(
        days_to_search=args.days,
        quickscan=args.quickscan,
        step_size=args.step,
        batch_size=args.batch,
        target_date_str=args.date,
        from_block=args.from_block,
        to_block=args.to_block,
        include_solana=not args.no_solana,
        analyze_market=args.analyze_market
    ))
    
    # Display results
    print(f"\nFound {len(results)} potential contracts/tokens across all chains:")
    for result in results:
        chain = result.get('chain', 'unknown').upper()
        
        if chain in ['BASE', 'ETHEREUM']:
            matched_terms_str = ", ".join(result.get('matched_terms', []))
            print(f"• [{chain}] {result['address']} created at {time.ctime(result['timestamp'])}")
            print(f"  TX: {result['tx_hash']} | Block: {result['block']}")
            print(f"  Matched terms: {matched_terms_str}\n")
        elif chain == 'SOLANA':
            token_name = result.get('token_name', 'Unknown')
            token_data = result.get('data', {})
            print(f"• [{chain}] {token_name.upper()} Token")
            print(f"  Address: {result['address']}")
            print(f"  URL: https://pump.fun/coin/{result['address']}")
            
            # Display token market data if available
            if "name" in token_data:
                print(f"  Name: {token_data['name']}")
            if "symbol" in token_data:
                print(f"  Symbol: {token_data['symbol']}")
            if "current_price" in token_data:
                print(f"  Current Price: ${token_data['current_price']}")
            if "price_change_24h" in token_data:
                print(f"  24h Change: {token_data['price_change_24h']}%")
            if "volume_24h" in token_data:
                print(f"  24h Volume: ${token_data['volume_24h']}")
            if "liquidity" in token_data:
                print(f"  Liquidity: ${token_data['liquidity']}")
            
            # Print market analysis if available
            if "market_data" in result:
                md = result["market_data"]
                print(f"\n  Price Analysis Summary:")
                print(f"    Highest Price: ${md['price_metrics']['highest']['value']} on {md['price_metrics']['highest']['date']}")
                print(f"    Lowest Price: ${md['price_metrics']['lowest']['value']} on {md['price_metrics']['lowest']['date']}")
                print(f"    Largest % Increase: {md['largest_movements']['increase']['percent']:.2f}%")
                print(f"    Largest % Decrease: {md['largest_movements']['decrease']['percent']:.2f}%")
            
            print("")