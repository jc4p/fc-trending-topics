#!/usr/bin/env python3
"""
Analyze Ghibli-related tokens and NFTs across multiple chains using Alchemy.
Tracks creation date, liquidity, price history, trading metrics, and NFT data.
"""

import os
import json
import asyncio
import datetime
import requests
import logging
from typing import Dict, List, Any, Optional, Tuple
from decimal import Decimal
import pandas as pd
from web3 import Web3, AsyncWeb3
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ghibli_token_analysis_debug.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ghibli_analyzer")

# Set third-party loggers to a higher level to reduce noise
logging.getLogger("web3").setLevel(logging.INFO)
logging.getLogger("urllib3").setLevel(logging.INFO)
logging.getLogger("requests").setLevel(logging.INFO)

# Load environment variables
load_dotenv()

# Configure initial token data
GHIBLI_TOKENS = [
    {"name": "Ghibli of TheDeFiLawyer", "ticker": None, "platform": "Zora", "launch_date": "2025-03-29",
     "start_price": 0.000010, "peak_price": 0.000020, "end_price": 0.000005, "contract": None},
    {"name": "Ghibli Style ✨", "ticker": None, "platform": "Zora", "launch_date": "2025-03-30",
     "start_price": 0.000015, "peak_price": 0.000020, "end_price": 0.000003, "contract": None},
    {"name": "Grok x Ghibli (GxG)", "ticker": "GxG", "platform": "Zora", "launch_date": "2025-03-30",
     "start_price": 0.04, "peak_price": 0.05, "end_price": 0.002, "contract": None},
    {"name": "Ghibli Cat", "ticker": None, "platform": "Zora", "launch_date": "2025-03-31",
     "start_price": 0.000008, "peak_price": 0.000010, "end_price": 0.000001, "contract": None},
    {"name": "The Floating Isle of Ghibli", "ticker": "GHIBLIFLOAT", "platform": "Zora", "launch_date": "2025-03-27",
     "start_price": 0.00005, "peak_price": 0.00005, "end_price": 0.0000005, "contract": None},
    {"name": "Ghibli Doge", "ticker": "GhibliDoge", "platform": "Zora", "launch_date": "2025-03-27",
     "start_price": 0.0003, "peak_price": 0.0003, "end_price": 0.000001, "contract": None},
    {"name": "Ghibli Elon", "ticker": "GHIBLI ELON", "platform": "Zora", "launch_date": "2025-03-28",
     "start_price": 0.000008, "peak_price": 0.000010, "end_price": 0.00000001, "contract": None},
    {"name": "Ghibli Morning Routine", "ticker": "GBMR", "platform": "Zora", "launch_date": "2025-03-29",
     "start_price": 0.00001, "peak_price": 0.00002, "end_price": 0, "contract": None},
    {"name": "Ghiblify Families", "ticker": "GFAM", "platform": "Bankr", "launch_date": "2025-03-28",
     "start_price": 0.00002, "peak_price": 0.00003, "end_price": 0.0000001, "contract": None},
    {"name": "Family GhibliShots", "ticker": "F-GHIB", "platform": "Clankr", "launch_date": "2025-03-29",
     "start_price": 0.0001, "peak_price": 0.00012, "end_price": 0.000001, "contract": None},
    {"name": "Ghibli Family Portraits", "ticker": "GFP", "platform": "Base", "launch_date": "2025-03-27",
     "start_price": 0.0006, "peak_price": 0.0009, "end_price": 0.000001, "contract": None},
    {"name": "Ghiblification", "ticker": "GHIBLI", "platform": "Base", "launch_date": "2025-03-26",
     "start_price": 0.00007, "peak_price": 0.05, "end_price": 0.0055, "contract": None},
    {"name": "Ghibli Ape", "ticker": "GAPE", "platform": "Base", "launch_date": "2025-03-29",
     "start_price": 0.00001, "peak_price": 0.00010, "end_price": 0.000074, "contract": None},
    {"name": "Ghibli Pepe", "ticker": "GPEPE", "platform": "Base", "launch_date": "2025-03-28",
     "start_price": 0.00001, "peak_price": 0.00005, "end_price": 0.000025, "contract": None},
    {"name": "Studio Ghibli", "ticker": "SGOB", "platform": "Base (Bankr)", "launch_date": "2025-03-27",
     "start_price": 0, "peak_price": 0.0000002, "end_price": 0.00000015, "contract": None},
]

# RPC endpoints
CHAIN_RPC = {
    "ethereum": os.getenv("ETH_RPC_URL"),
    "base": os.getenv("BASE_RPC_URL"),
    "optimism": os.getenv("OPTIMISM_RPC_URL"),
    "zora": os.getenv("ZORA_RPC_URL"),
}

# Alchemy API Key
ALCHEMY_API_KEY = os.getenv("ALCHEMY_API_KEY")

# Map platform names to chain keys
PLATFORM_TO_CHAIN = {
    "Zora": "zora",
    "Base": "base",
    "Base (Bankr)": "base",
    "Bankr": "base",  # Assuming Bankr is on Base
    "Clankr": "base",  # Assuming Clankr is on Base
    # Map platform names to looser translations to allow flexibility in matching
    "Zora Network": "zora",
    "Base Chain": "base",
    "Bankr": "base",
    "Clankr": "base",
}

# Initialize web3 connections for each chain
web3_clients = {}
async_web3_clients = {}

def initialize_web3_connections():
    """Initialize Web3 connections for each chain."""
    for chain, rpc_url in CHAIN_RPC.items():
        if not rpc_url:
            print(f"Warning: No RPC URL provided for {chain}")
            continue
        web3_clients[chain] = Web3(Web3.HTTPProvider(rpc_url))
        async_web3_clients[chain] = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(rpc_url))
        
        # Test connection
        try:
            block_number = web3_clients[chain].eth.block_number
            print(f"Connected to {chain}: Current block {block_number}")
        except Exception as e:
            print(f"Failed to connect to {chain}: {e}")

async def search_all_networks_for_token(token_data: Dict[str, Any]) -> Dict[str, Any]:
    """Search for a token across all networks if not found on the specified platform."""
    logger.info(f"Searching all networks for {token_data['name']} with ticker {token_data['ticker']}")
    
    # If no ticker, we can't easily search across networks
    if not token_data["ticker"]:
        logger.info(f"No ticker for {token_data['name']}, skipping cross-network search")
        return None, None
    
    # Try to find on all networks
    for network_key, rpc_url in CHAIN_RPC.items():
        # Skip the original platform's network (already searched)
        original_platform = token_data["platform"]
        original_network = PLATFORM_TO_CHAIN.get(original_platform)
        if network_key == original_network:
            continue
            
        if not rpc_url:
            continue
            
        # Determine base URL for this network
        if network_key == "ethereum":
            base_url = f"https://eth-mainnet.g.alchemy.com/v2/{ALCHEMY_API_KEY}"
        elif network_key == "base":
            base_url = f"https://base-mainnet.g.alchemy.com/v2/{ALCHEMY_API_KEY}"
        elif network_key == "optimism":
            base_url = f"https://opt-mainnet.g.alchemy.com/v2/{ALCHEMY_API_KEY}"
        elif network_key == "zora":
            base_url = f"https://zora-mainnet.g.alchemy.com/v2/{ALCHEMY_API_KEY}"
        else:
            continue
            
        logger.debug(f"Cross-network search: Looking for {token_data['name']} with ticker {token_data['ticker']} on {network_key}")
        
        # Use Alchemy's Token API for symbol search
        try:
            url = f"https://api.g.alchemy.com/prices/v1/{ALCHEMY_API_KEY}/tokens"
            headers = {"accept": "application/json"}
            params = {"symbols": token_data["ticker"]}
            
            response = requests.get(url, headers=headers, params=params)
            if response.status_code == 200:
                result = response.json()
                if "data" in result and token_data["ticker"] in result["data"]:
                    token_info = result["data"][token_data["ticker"]]
                    contract_address = token_info.get("contractAddress")
                    
                    if contract_address:
                        # Check creation date match if we can get it
                        contract_creation_near_expected = await check_token_creation_date_match(
                            network_key, contract_address, token_data["launch_date"])
                        
                        if contract_creation_near_expected:
                            logger.info(f"CROSS-NETWORK MATCH! Found {token_data['name']} ({token_data['ticker']}) on {network_key} at {contract_address}")
                            return network_key, contract_address
                        else:
                            logger.debug(f"Found token {token_data['ticker']} on {network_key} but creation date doesn't match")
        except Exception as e:
            logger.error(f"Error in cross-network search for {token_data['ticker']} on {network_key}: {e}")
    
    logger.info(f"No matches found for {token_data['name']} ({token_data['ticker']}) on any network")
    return None, None

async def check_token_creation_date_match(network_key, contract_address, expected_date):
    """Check if token creation date approximately matches expected date."""
    try:
        web3 = async_web3_clients.get(network_key)
        if not web3:
            return False
            
        # Get the base URL for this network
        if network_key == "ethereum":
            base_url = f"https://eth-mainnet.g.alchemy.com/v2/{ALCHEMY_API_KEY}"
        elif network_key == "base":
            base_url = f"https://base-mainnet.g.alchemy.com/v2/{ALCHEMY_API_KEY}"
        elif network_key == "optimism":
            base_url = f"https://opt-mainnet.g.alchemy.com/v2/{ALCHEMY_API_KEY}"
        elif network_key == "zora":
            base_url = f"https://zora-mainnet.g.alchemy.com/v2/{ALCHEMY_API_KEY}"
        else:
            return False
            
        # Try to find first transaction to/from this contract
        headers = {"accept": "application/json", "content-type": "application/json"}
        payload = {
            "id": 1,
            "jsonrpc": "2.0",
            "method": "alchemy_getAssetTransfers",
            "params": [
                {
                    "fromBlock": "0x0",
                    "toBlock": "latest",
                    "toAddress": contract_address,
                    "excludeZeroValue": False,
                    "category": ["external", "erc20", "erc721", "erc1155"]
                }
            ]
        }
        
        response = requests.post(base_url, json=payload, headers=headers)
        if response.status_code == 200 and "result" in response.json():
            result = response.json()
            
            if "result" in result and "transfers" in result["result"] and result["result"]["transfers"]:
                transfers = result["result"]["transfers"]
                # Find the earliest transaction
                earliest_tx = min(transfers, key=lambda x: int(x["blockNum"], 16))
                creation_block = int(earliest_tx["blockNum"], 16)
                
                # Get block timestamp
                block_data = await web3.eth.get_block(creation_block)
                creation_timestamp = block_data["timestamp"]
                creation_date = datetime.datetime.fromtimestamp(creation_timestamp).strftime("%Y-%m-%d")
                
                # Parse expected date
                expected_date_obj = datetime.datetime.strptime(expected_date, "%Y-%m-%d")
                actual_date_obj = datetime.datetime.strptime(creation_date, "%Y-%m-%d")
                
                # Allow a 3-day window for matching
                diff = abs((actual_date_obj - expected_date_obj).days)
                logger.debug(f"Token creation date: expected {expected_date}, actual {creation_date}, diff {diff} days")
                
                return diff <= 3  # Match if within 3 days of expected date
                
        logger.debug(f"Could not determine creation date for {contract_address} on {network_key}")
        return False  # No creation date found
        
    except Exception as e:
        logger.error(f"Error checking token creation date for {contract_address} on {network_key}: {e}")
        return False

async def fetch_token_contract_details(token_data: Dict[str, Any]) -> Dict[str, Any]:
    """Fetch contract details for a token using Alchemy API."""
    platform = token_data["platform"]
    chain_key = PLATFORM_TO_CHAIN.get(platform)
    
    if not chain_key or chain_key not in async_web3_clients:
        logger.warning(f"Chain {platform} not supported or RPC not configured")
        return token_data
    
    web3 = async_web3_clients[chain_key]
    
    # If we already have the contract address, use it directly
    if token_data.get("contract"):
        contract_address = token_data["contract"]
        logger.info(f"Using provided contract address {contract_address} for {token_data['name']}")
    else:
        # Use Alchemy's token API to search for the token
        logger.info(f"Searching for {token_data['name']} on {platform}...")
        print(f"Searching for {token_data['name']} on {platform}...")
        
        # Determine Alchemy base URL based on chain
        if chain_key == "ethereum":
            base_url = f"https://eth-mainnet.g.alchemy.com/v2/{ALCHEMY_API_KEY}"
        elif chain_key == "base":
            base_url = f"https://base-mainnet.g.alchemy.com/v2/{ALCHEMY_API_KEY}"
        elif chain_key == "optimism":
            base_url = f"https://opt-mainnet.g.alchemy.com/v2/{ALCHEMY_API_KEY}"
        elif chain_key == "zora":
            base_url = f"https://zora-mainnet.g.alchemy.com/v2/{ALCHEMY_API_KEY}"
        else:
            logger.warning(f"Unsupported chain: {chain_key}")
            print(f"Unsupported chain: {chain_key}")
            return token_data
            
        # Try different search methods for better coverage
        contract_address = None
        
        # First try to search using token ticker if available
        if token_data["ticker"] and not contract_address:
            try:
                # Use Alchemy's Token API for symbol search
                url = f"https://api.g.alchemy.com/prices/v1/{ALCHEMY_API_KEY}/tokens"
                headers = {"accept": "application/json"}
                params = {"symbols": token_data["ticker"]}
                
                logger.debug(f"Searching for ticker {token_data['ticker']} using Alchemy Tokens API")
                response = requests.get(url, headers=headers, params=params)
                if response.status_code == 200:
                    result = response.json()
                    logger.debug(f"Ticker search result: {json.dumps(result)[:1000]}...")
                    
                    if "data" in result and token_data["ticker"] in result["data"]:
                        token_info = result["data"][token_data["ticker"]]
                        contract_address = token_info.get("contractAddress")
                        if contract_address:
                            token_data["found_type"] = "ERC20"
                            token_data["decimals"] = token_info.get("decimals", 18)
                            token_data["contract_name"] = token_info.get("name", token_data["name"])
                            token_data["contract_symbol"] = token_info.get("symbol", token_data["ticker"])
                            token_data["found_network"] = chain_key
                            logger.info(f"Found token via ticker: {token_data['contract_name']} ({token_data['contract_symbol']}) at {contract_address}")
                            print(f"Found token via ticker: {token_data['contract_name']} ({token_data['contract_symbol']}) at {contract_address}")
            except Exception as e:
                logger.error(f"Error searching by ticker {token_data['ticker']}: {e}")
                print(f"Error searching by ticker {token_data['ticker']}: {e}")
        
        # If ticker search failed, try searching for ERC20 tokens by name
        if not contract_address:
            try:
                # Use Alchemy's searchTokens RPC call
                payload = {
                    "id": 1,
                    "jsonrpc": "2.0",
                    "method": "alchemy_searchTokens",
                    "params": [
                        {
                            "query": token_data["name"],
                            "filter": {
                                "standard": "ERC20"
                            }
                        }
                    ]
                }
                
                headers = {
                    "accept": "application/json",
                    "content-type": "application/json"
                }
                
                logger.debug(f"Searching for token by name: {token_data['name']}")
                response = requests.post(base_url, json=payload, headers=headers)
                if response.status_code == 200:
                    result = response.json()
                    logger.debug(f"Name search result: {json.dumps(result)[:1000]}...")
                    
                    if "result" in result and result["result"] and len(result["result"]["tokens"]) > 0:
                        # Find the most likely match
                        for token in result["result"]["tokens"]:
                            # Check for "ghibli" in name or symbol, or match ticker
                            token_name_lower = token["name"].lower()
                            token_symbol_lower = token["symbol"].lower()
                            
                            # Looser matching criteria - check for partial matches with 'ghibli'
                            if ("ghibli" in token_name_lower or 
                                "ghibl" in token_name_lower or 
                                "ghibli" in token_symbol_lower or
                                (token_data["ticker"] and token_data["ticker"].lower() in token_symbol_lower) or
                                (token_data["ticker"] and token_symbol_lower in token_data["ticker"].lower())):
                                
                                contract_address = token["address"]
                                token_data["found_type"] = "ERC20"
                                token_data["decimals"] = token.get("decimals", 18)
                                token_data["contract_name"] = token["name"]
                                token_data["contract_symbol"] = token["symbol"]
                                token_data["found_network"] = chain_key
                                logger.info(f"Found ERC20 token: {token['name']} ({token['symbol']}) at {contract_address}")
                                print(f"Found ERC20 token: {token['name']} ({token['symbol']}) at {contract_address}")
                                break
            
            except Exception as e:
                logger.error(f"Error searching for ERC20 token {token_data['name']}: {e}")
                print(f"Error searching for ERC20 token {token_data['name']}: {e}")
        
        # If still no match, try searching for NFTs (ERC721/ERC1155)
        if not contract_address:
            try:
                payload = {
                    "id": 1,
                    "jsonrpc": "2.0",
                    "method": "alchemy_searchNFTs",
                    "params": [
                        {
                            "query": token_data["name"]
                        }
                    ]
                }
                
                headers = {
                    "accept": "application/json",
                    "content-type": "application/json"
                }
                
                logger.debug(f"Searching for NFT collection: {token_data['name']}")
                response = requests.post(base_url, json=payload, headers=headers)
                if response.status_code == 200:
                    result = response.json()
                    logger.debug(f"NFT search result: {json.dumps(result)[:1000]}...")
                    
                    if "result" in result and result["result"] and len(result["result"]["contracts"]) > 0:
                        # Find the most likely match with looser criteria
                        for nft in result["result"]["contracts"]:
                            nft_name_lower = nft["name"].lower()
                            if "ghibli" in nft_name_lower or "ghibl" in nft_name_lower:
                                contract_address = nft["address"]
                                token_data["found_type"] = "NFT"
                                token_data["contract_name"] = nft["name"]
                                token_data["token_type"] = nft.get("tokenType", "Unknown")
                                token_data["found_network"] = chain_key
                                logger.info(f"Found NFT collection: {nft['name']} at {contract_address}")
                                print(f"Found NFT collection: {nft['name']} at {contract_address}")
                                break
            
            except Exception as e:
                logger.error(f"Error searching for NFT collection {token_data['name']}: {e}")
                print(f"Error searching for NFT collection {token_data['name']}: {e}")
                
        # If we're on Base, try a more focused search for new/recent tokens
        if not contract_address and chain_key == "base" and "base" in token_data["platform"].lower():
            # For Base-specific tokens, we might need to use a more targeted approach
            logger.info(f"Performing focused Base chain search for {token_data['name']}...")
            print(f"Performing focused Base chain search for {token_data['name']}...")
        
        # If we still didn't find anything on the specified network,
        # try searching on other networks
        if not contract_address:
            cross_network, cross_network_address = await search_all_networks_for_token(token_data)
            if cross_network and cross_network_address:
                contract_address = cross_network_address
                token_data["found_network"] = cross_network
                token_data["found_type"] = "ERC20"  # Assuming ERC20 for cross-network matches
                token_data["cross_network_match"] = True
                logger.info(f"Found {token_data['name']} on {cross_network} instead of {platform}")
                print(f"CROSS-NETWORK MATCH: Found {token_data['name']} on {cross_network} network instead of {platform}")
            else:
                logger.info(f"No cross-network matches found for {token_data['name']}")
                print(f"No cross-network matches found for {token_data['name']}")
    
    if not contract_address:
        print(f"Could not find contract address for {token_data['name']}")
        return token_data
    
    # Update token data with contract address
    token_data["contract"] = contract_address
    
    # Fetch token info based on token type
    try:
        # Use the network where token was found, which might be different from the original platform
        chain_key = token_data.get("found_network", PLATFORM_TO_CHAIN.get(token_data["platform"]))
        web3 = async_web3_clients.get(chain_key)
        
        if not web3:
            logger.warning(f"No web3 client for {chain_key}, using default chain")
            chain_key = PLATFORM_TO_CHAIN.get(token_data["platform"])
            web3 = async_web3_clients.get(chain_key)
        
        # Get creation timestamp
        creation_block = await find_token_creation_block(web3, contract_address, chain_key)
        if creation_block:
            block_data = await web3.eth.get_block(creation_block)
            token_data["creation_timestamp"] = block_data["timestamp"]
            token_data["creation_date"] = datetime.datetime.fromtimestamp(block_data["timestamp"]).strftime("%Y-%m-%d")
            logger.info(f"Token {token_data['name']} created on {token_data['creation_date']}")
        
        if token_data.get("found_type") == "ERC20":
            # Create ERC20 contract interface
            erc20_abi = [
                {"constant": True, "inputs": [], "name": "decimals", "outputs": [{"name": "", "type": "uint8"}], "payable": False, "stateMutability": "view", "type": "function"},
                {"constant": True, "inputs": [], "name": "totalSupply", "outputs": [{"name": "", "type": "uint256"}], "payable": False, "stateMutability": "view", "type": "function"},
                {"constant": True, "inputs": [], "name": "name", "outputs": [{"name": "", "type": "string"}], "payable": False, "stateMutability": "view", "type": "function"},
                {"constant": True, "inputs": [], "name": "symbol", "outputs": [{"name": "", "type": "string"}], "payable": False, "stateMutability": "view", "type": "function"}
            ]
            contract = web3.eth.contract(address=contract_address, abi=erc20_abi)
            
            # Get token details
            try:
                token_data["decimals"] = await contract.functions.decimals().call()
            except Exception:
                # Some tokens don't implement decimals()
                token_data["decimals"] = 18  # Default for most tokens
                
            try:
                total_supply_raw = await contract.functions.totalSupply().call()
                token_data["total_supply"] = total_supply_raw / (10 ** token_data["decimals"])
            except Exception as e:
                print(f"Error getting total supply: {e}")
                token_data["total_supply"] = "Unknown"
                
            # Get token name and symbol
            try:
                token_data["contract_name"] = await contract.functions.name().call()
            except Exception:
                pass  # Use the name we already have
                
            try:
                token_data["contract_symbol"] = await contract.functions.symbol().call()
            except Exception:
                pass  # Use the ticker we already have
                
        elif token_data.get("found_type") == "NFT":
            # Get NFT collection details using Alchemy API
            if chain_key:
                if chain_key == "ethereum":
                    base_url = f"https://eth-mainnet.g.alchemy.com/v2/{ALCHEMY_API_KEY}"
                elif chain_key == "base":
                    base_url = f"https://base-mainnet.g.alchemy.com/v2/{ALCHEMY_API_KEY}"
                elif chain_key == "optimism":
                    base_url = f"https://opt-mainnet.g.alchemy.com/v2/{ALCHEMY_API_KEY}"
                elif chain_key == "zora":
                    base_url = f"https://zora-mainnet.g.alchemy.com/v2/{ALCHEMY_API_KEY}"
                
                # Get NFT metadata
                headers = {"accept": "application/json", "content-type": "application/json"}
                payload = {
                    "id": 1,
                    "jsonrpc": "2.0",
                    "method": "alchemy_getContractMetadata",
                    "params": [
                        contract_address
                    ]
                }
                
                response = requests.post(base_url, json=payload, headers=headers)
                if response.status_code == 200:
                    result = response.json()
                    if "result" in result and result["result"]:
                        metadata = result["result"]
                        token_data["contract_name"] = metadata.get("name", token_data.get("contract_name", "Unknown"))
                        token_data["token_type"] = metadata.get("tokenType", token_data.get("token_type", "Unknown"))
                        token_data["total_supply"] = metadata.get("totalSupply", "Unknown")
                        token_data["symbol"] = metadata.get("symbol", token_data.get("ticker", "Unknown"))
                        token_data["opensea_floor_price"] = metadata.get("openSea", {}).get("floorPrice", "Unknown")
                        
                # Get NFT sales data
                payload = {
                    "id": 1,
                    "jsonrpc": "2.0",
                    "method": "alchemy_getNFTSales",
                    "params": [
                        {
                            "contractAddress": contract_address,
                            "startDate": "2025-03-01",  # Start from March 1, 2025
                            "endDate": "2025-04-13"     # Until today
                        }
                    ]
                }
                
                response = requests.post(base_url, json=payload, headers=headers)
                if response.status_code == 200:
                    result = response.json()
                    if "result" in result and "nftSales" in result["result"]:
                        sales = result["result"]["nftSales"]
                        if sales:
                            # Calculate total volume and average price
                            total_volume = 0
                            prices = []
                            for sale in sales:
                                if "sellerFee" in sale and "amount" in sale["sellerFee"]:
                                    price = float(sale["sellerFee"]["amount"])
                                    total_volume += price
                                    prices.append(price)
                            
                            token_data["sales_count"] = len(sales)
                            token_data["total_volume"] = total_volume
                            token_data["avg_price"] = total_volume / len(sales) if len(sales) > 0 else 0
                        else:
                            token_data["sales_count"] = 0
                            token_data["total_volume"] = 0
                            token_data["avg_price"] = 0
        
    except Exception as e:
        print(f"Error fetching token details for {token_data['name']}: {e}")
    
    return token_data

async def find_token_creation_block(web3, contract_address, chain_key):
    """Find the block where a token contract was created."""
    try:
        # Determine Alchemy base URL based on chain
        if chain_key == "ethereum":
            base_url = f"https://eth-mainnet.g.alchemy.com/v2/{ALCHEMY_API_KEY}"
        elif chain_key == "base":
            base_url = f"https://base-mainnet.g.alchemy.com/v2/{ALCHEMY_API_KEY}"
        elif chain_key == "optimism":
            base_url = f"https://opt-mainnet.g.alchemy.com/v2/{ALCHEMY_API_KEY}"
        elif chain_key == "zora":
            base_url = f"https://zora-mainnet.g.alchemy.com/v2/{ALCHEMY_API_KEY}"
        else:
            print(f"Unsupported chain for finding creation block: {chain_key}")
            return None
            
        # Use Alchemy's getAssetTransfers API to find the contract creation transaction
        headers = {"accept": "application/json", "content-type": "application/json"}
        
        # First try to get contract creation with eth_getCode method
        code_payload = {
            "id": 1,
            "jsonrpc": "2.0",
            "method": "eth_getCode",
            "params": [contract_address, "latest"]
        }
        
        response = requests.post(base_url, json=code_payload, headers=headers)
        if response.status_code != 200 or "result" not in response.json():
            print(f"Could not verify contract code for {contract_address}")
            return None
            
        # Now try to find the first transaction to/from this contract
        transfer_payload = {
            "id": 1,
            "jsonrpc": "2.0",
            "method": "alchemy_getAssetTransfers",
            "params": [
                {
                    "fromBlock": "0x0",
                    "toBlock": "latest",
                    "toAddress": contract_address,
                    "excludeZeroValue": False,
                    "category": ["external", "erc20", "erc721", "erc1155"]
                }
            ]
        }
        
        response = requests.post(base_url, json=transfer_payload, headers=headers)
        if response.status_code == 200 and "result" in response.json():
            result = response.json()
            
            if "result" in result and "transfers" in result["result"] and len(result["result"]["transfers"]) > 0:
                transfers = result["result"]["transfers"]
                # Find the earliest transaction
                earliest_tx = min(transfers, key=lambda x: int(x["blockNum"], 16))
                creation_block = int(earliest_tx["blockNum"], 16)
                print(f"Found first transaction at block {creation_block}")
                return creation_block
                
        # If no transactions found to the contract, try using getLogs to find contract creation
        # This is typically more accurate but more complicated to implement
        print("Could not find creation block using transfer history")
        return None
        
    except Exception as e:
        print(f"Error finding creation block: {e}")
        return None

async def get_liquidity_data(token_data):
    """Get liquidity data for a token or NFT."""
    if not token_data.get("contract"):
        return token_data
    
    chain_key = PLATFORM_TO_CHAIN.get(token_data["platform"])
    if not chain_key or chain_key not in async_web3_clients:
        return token_data
    
    # Set base URL for Alchemy API
    if chain_key == "ethereum":
        base_url = f"https://eth-mainnet.g.alchemy.com/v2/{ALCHEMY_API_KEY}"
    elif chain_key == "base":
        base_url = f"https://base-mainnet.g.alchemy.com/v2/{ALCHEMY_API_KEY}"
    elif chain_key == "optimism":
        base_url = f"https://opt-mainnet.g.alchemy.com/v2/{ALCHEMY_API_KEY}"
    elif chain_key == "zora":
        base_url = f"https://zora-mainnet.g.alchemy.com/v2/{ALCHEMY_API_KEY}"
    else:
        return token_data
    
    contract_address = token_data["contract"]
    
    # Handle differently based on token type
    if token_data.get("found_type") == "ERC20":
        try:
            # For ERC20 tokens, we need to find liquidity pools
            # For DEXes like Uniswap, we can use their subgraph or APIs
            
            # 1. Try to find Uniswap pool (if on Ethereum, Optimism or Base)
            if chain_key in ["ethereum", "optimism", "base"]:
                # Determine appropriate Uniswap subgraph URL based on chain
                if chain_key == "ethereum":
                    # Uniswap V3 on Ethereum
                    subgraph_url = "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3"
                elif chain_key == "optimism":
                    # Uniswap V3 on Optimism
                    subgraph_url = "https://api.thegraph.com/subgraphs/name/ianlapham/optimism-post-regenesis"
                elif chain_key == "base":
                    # Uniswap V3 on Base
                    subgraph_url = "https://api.thegraph.com/subgraphs/name/ianlapham/base-v3"
                
                # Query for pool data
                query = """
                {
                  pools(where: {or: [
                    {token0: "%s"}, 
                    {token1: "%s"}
                  ]}, orderBy: totalValueLockedUSD, orderDirection: desc, first: 5) {
                    id
                    token0 {
                      symbol
                      id
                    }
                    token1 {
                      symbol
                      id
                    }
                    totalValueLockedToken0
                    totalValueLockedToken1
                    totalValueLockedUSD
                    volumeUSD
                  }
                }
                """ % (contract_address.lower(), contract_address.lower())
                
                try:
                    response = requests.post(subgraph_url, json={"query": query})
                    if response.status_code == 200:
                        data = response.json()
                        if "data" in data and "pools" in data["data"] and data["data"]["pools"]:
                            pools = data["data"]["pools"]
                            
                            # Sum up total liquidity across pools
                            total_tvl = sum(float(pool.get("totalValueLockedUSD", 0)) for pool in pools)
                            token_data["liquidity_pools"] = len(pools)
                            token_data["current_liquidity_usd"] = total_tvl
                            
                            # Get details of the largest pool
                            largest_pool = pools[0]
                            token_data["largest_pool_address"] = largest_pool["id"]
                            token_data["largest_pool_tvl"] = float(largest_pool.get("totalValueLockedUSD", 0))
                            
                            # Determine if token is token0 or token1 in the pool
                            is_token0 = largest_pool["token0"]["id"].lower() == contract_address.lower()
                            pair_token = largest_pool["token1" if is_token0 else "token0"]
                            token_data["paired_with"] = pair_token["symbol"]
                except Exception as e:
                    print(f"Error fetching Uniswap data: {e}")
            
            # Initialize with unknown values if we couldn't find data
            if "current_liquidity_usd" not in token_data:
                token_data["current_liquidity_usd"] = "Unknown"
                token_data["liquidity_pools"] = 0
        
        except Exception as e:
            print(f"Error fetching liquidity data for {token_data['name']}: {e}")
            token_data["current_liquidity_usd"] = "Unknown"
            token_data["liquidity_pools"] = 0
    
    elif token_data.get("found_type") == "NFT":
        # For NFTs, we don't have traditional liquidity pools
        # Instead, we can look at floor price, volume, unique holders, etc.
        try:
            # Get holder data using getNFTsForCollection API
            headers = {"accept": "application/json", "content-type": "application/json"}
            
            # Just get count of tokens and unique owners
            payload = {
                "id": 1,
                "jsonrpc": "2.0",
                "method": "alchemy_getTokenMetadataWithOwners",
                "params": [
                    contract_address, 
                    []  # Empty array gives us a basic count
                ]
            }
            
            try:
                response = requests.post(base_url, json=payload, headers=headers)
                if response.status_code == 200:
                    result = response.json()
                    if "result" in result and "ownerCount" in result["result"]:
                        token_data["unique_owners"] = result["result"]["ownerCount"]
            except Exception as e:
                print(f"Error fetching NFT owner data: {e}")
                token_data["unique_owners"] = "Unknown"
                
        except Exception as e:
            print(f"Error fetching NFT data for {token_data['name']}: {e}")
    
    return token_data

async def get_price_history(token_data):
    """Get price history for a token or NFT using Alchemy's Historical Token Prices API."""
    if not token_data.get("contract"):
        return token_data
    
    # Calculate days since launch
    try:
        # Use creation_date from blockchain if available, otherwise use the provided launch_date
        if token_data.get("creation_date"):
            launch_date = datetime.datetime.strptime(token_data["creation_date"], "%Y-%m-%d")
        else:
            launch_date = datetime.datetime.strptime(token_data["launch_date"], "%Y-%m-%d")
            
        today = datetime.datetime.strptime("2025-04-13", "%Y-%m-%d")
        days_since_launch = (today - launch_date).days
        token_data["days_since_launch"] = days_since_launch
        
        # Set dates for 5 and 10 days after launch
        five_days_after = launch_date + datetime.timedelta(days=5)
        ten_days_after = launch_date + datetime.timedelta(days=10)
        
        # Format dates for API (ISO 8601)
        start_date_iso = launch_date.strftime('%Y-%m-%dT00:00:00Z')
        end_date_iso = today.strftime('%Y-%m-%dT23:59:59Z')
        
        chain_key = PLATFORM_TO_CHAIN.get(token_data["platform"])
        
        if token_data.get("found_type") == "ERC20":
            # Try using Alchemy's Historical Token Prices API for accurate data
            url = f"https://api.g.alchemy.com/prices/v1/{ALCHEMY_API_KEY}/tokens/historical"
            
            # Prepare request data
            headers = {
                "accept": "application/json", 
                "content-type": "application/json"
            }
            
            # If we have the contract address, use it with the network
            if token_data.get("contract"):
                network_id = None
                if chain_key == "ethereum":
                    network_id = "ETHEREUM_MAINNET"
                elif chain_key == "base":
                    network_id = "BASE_MAINNET"
                elif chain_key == "optimism":
                    network_id = "OPTIMISM_MAINNET"
                elif chain_key == "zora":
                    # Zora might not be supported by Alchemy Prices API yet
                    network_id = None
                
                if network_id:
                    payload = {
                        "contractAddress": token_data["contract"],
                        "network": network_id,
                        "startTime": start_date_iso,
                        "endTime": end_date_iso,
                        "interval": "1d"  # Daily intervals
                    }
                    
                    print(f"Fetching price history for {token_data['name']} ({token_data['contract']}) on {network_id}")
                    
                    try:
                        response = requests.post(url, headers=headers, json=payload)
                        if response.status_code == 200:
                            result = response.json()
                            
                            if "data" in result and "prices" in result["data"]:
                                prices = result["data"]["prices"]
                                if prices:
                                    # Extract price data for key dates
                                    price_data = {}
                                    peak_price = 0
                                    peak_date = None
                                    
                                    # Process each price point
                                    for price_point in prices:
                                        timestamp = price_point.get("timestamp")
                                        price = price_point.get("price", 0)
                                        
                                        if timestamp and price:
                                            price_date = datetime.datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                                            price_data[price_date] = price
                                            
                                            # Track peak price
                                            if price > peak_price:
                                                peak_price = price
                                                peak_date = price_date
                                    
                                    # Find prices at specific points
                                    if price_data:
                                        # Find the closest date for 5 days after launch
                                        five_day_date = min(price_data.keys(), key=lambda d: abs((d - five_days_after).total_seconds()))
                                        if abs((five_day_date - five_days_after).total_seconds()) < 86400 * 2:  # Within 2 days
                                            token_data["price_5days_after_launch"] = price_data[five_day_date]
                                        
                                        # Find the closest date for 10 days after launch
                                        ten_day_date = min(price_data.keys(), key=lambda d: abs((d - ten_days_after).total_seconds()))
                                        if abs((ten_day_date - ten_days_after).total_seconds()) < 86400 * 2:  # Within 2 days
                                            token_data["price_10days_after_launch"] = price_data[ten_day_date]
                                        
                                        # Calculate days to peak
                                        if peak_date:
                                            days_to_peak = (peak_date - launch_date).days
                                            token_data["days_to_peak"] = days_to_peak
                                            
                                            # Record actual peak price from data
                                            token_data["actual_peak_price"] = peak_price
                                    
                                    print(f"Found price data for {token_data['name']}: {len(prices)} data points")
                            else:
                                print(f"No price data returned for {token_data['name']}")
                        else:
                            print(f"Failed to fetch price history via Alchemy API: {response.status_code}, {response.text}")
                    
                    except Exception as e:
                        print(f"Error fetching Alchemy price history: {e}")
            
            # If we have ticker/symbol, try that approach too
            if token_data.get("contract_symbol") or token_data.get("ticker"):
                symbol = token_data.get("contract_symbol") or token_data.get("ticker")
                if symbol:
                    payload = {
                        "symbol": symbol,
                        "startTime": start_date_iso,
                        "endTime": end_date_iso,
                        "interval": "1d"  # Daily intervals
                    }
                    
                    print(f"Fetching price history for {token_data['name']} by symbol {symbol}")
                    
                    try:
                        response = requests.post(url, headers=headers, json=payload)
                        if response.status_code == 200:
                            result = response.json()
                            
                            if "data" in result and "prices" in result["data"]:
                                prices = result["data"]["prices"]
                                if prices:
                                    # Similar processing as above for symbol-based query
                                    price_data = {}
                                    peak_price = 0
                                    peak_date = None
                                    
                                    for price_point in prices:
                                        timestamp = price_point.get("timestamp")
                                        price = price_point.get("price", 0)
                                        
                                        if timestamp and price:
                                            price_date = datetime.datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                                            price_data[price_date] = price
                                            
                                            # Track peak price
                                            if price > peak_price:
                                                peak_price = price
                                                peak_date = price_date
                                    
                                    # Find prices at specific points if we haven't already got them
                                    if price_data:
                                        if "price_5days_after_launch" not in token_data:
                                            five_day_date = min(price_data.keys(), key=lambda d: abs((d - five_days_after).total_seconds()))
                                            if abs((five_day_date - five_days_after).total_seconds()) < 86400 * 2:
                                                token_data["price_5days_after_launch"] = price_data[five_day_date]
                                        
                                        if "price_10days_after_launch" not in token_data:
                                            ten_day_date = min(price_data.keys(), key=lambda d: abs((d - ten_days_after).total_seconds()))
                                            if abs((ten_day_date - ten_days_after).total_seconds()) < 86400 * 2:
                                                token_data["price_10days_after_launch"] = price_data[ten_day_date]
                                        
                                        if "days_to_peak" not in token_data and peak_date:
                                            days_to_peak = (peak_date - launch_date).days
                                            token_data["days_to_peak"] = days_to_peak
                                            token_data["actual_peak_price"] = peak_price
                                    
                                    print(f"Found price data for {token_data['name']} by symbol {symbol}: {len(prices)} data points")
                            else:
                                print(f"No price data returned for symbol {symbol}")
                        else:
                            print(f"Failed to fetch price history via Alchemy API for symbol {symbol}: {response.status_code}")
                    
                    except Exception as e:
                        print(f"Error fetching Alchemy price history by symbol: {e}")
            
            # If Alchemy API didn't work, try DEX data or fall back to interpolation
            if "price_5days_after_launch" not in token_data or "price_10days_after_launch" not in token_data or "days_to_peak" not in token_data:
                print(f"Using interpolation for missing price data for {token_data['name']}")
                
                # Use provided price data for interpolation
                start_price = token_data["start_price"]
                peak_price = token_data["peak_price"]
                end_price = token_data["end_price"]
                
                # Use a heuristic for days to peak if not available
                if "days_to_peak" not in token_data:
                    days_to_peak = int(days_since_launch * 0.2)  # Assume peak at 20% of days since launch
                    token_data["days_to_peak"] = days_to_peak
                else:
                    days_to_peak = token_data["days_to_peak"]
                
                # Estimate prices at day 5 and 10 if not available
                if "price_5days_after_launch" not in token_data:
                    if days_since_launch <= 5:
                        # If less than 5 days, interpolate between start and current
                        price_5days = start_price + (end_price - start_price) * (5 / days_since_launch)
                    else:
                        if days_to_peak <= 5:
                            # Peak before 5 days, interpolate between peak and current
                            price_5days = peak_price + (end_price - peak_price) * ((5 - days_to_peak) / (days_since_launch - days_to_peak))
                        else:
                            # Peak after 5 days, interpolate between start and peak
                            price_5days = start_price + (peak_price - start_price) * (5 / days_to_peak)
                    token_data["price_5days_after_launch"] = price_5days
                
                if "price_10days_after_launch" not in token_data:
                    if days_since_launch <= 10:
                        # If less than 10 days, interpolate between start and current
                        price_10days = start_price + (end_price - start_price) * (10 / days_since_launch)
                    else:
                        if days_to_peak <= 10:
                            # Peak before 10 days, interpolate between peak and current
                            price_10days = peak_price + (end_price - peak_price) * ((10 - days_to_peak) / (days_since_launch - days_to_peak))
                        else:
                            # Peak after 10 days, interpolate between start and peak
                            price_10days = start_price + (peak_price - start_price) * (10 / days_to_peak)
                    token_data["price_10days_after_launch"] = price_10days
            
            # Calculate max tradeable amount at peak (estimate if not available)
            if token_data.get("largest_pool_tvl") and "max_tradeable_at_peak" not in token_data:
                # Assume 0.5% slippage tolerance
                max_tradeable = token_data["largest_pool_tvl"] * 0.01  # 1% of pool TVL
                token_data["max_tradeable_at_peak"] = max_tradeable
            elif "max_tradeable_at_peak" not in token_data:
                token_data["max_tradeable_at_peak"] = "Unknown"
            
        elif token_data.get("found_type") == "NFT":
            # For NFTs, handle differently
            token_data["price_5days_after_launch"] = "N/A for NFT"
            token_data["price_10days_after_launch"] = "N/A for NFT"
            token_data["days_to_peak"] = "N/A for NFT"
            token_data["max_tradeable_at_peak"] = "N/A for NFT"
        
        # Ensure we have values for all required fields
        if "price_5days_after_launch" not in token_data:
            token_data["price_5days_after_launch"] = "Unknown"
        
        if "price_10days_after_launch" not in token_data:
            token_data["price_10days_after_launch"] = "Unknown"
        
        if "days_to_peak" not in token_data:
            token_data["days_to_peak"] = "Unknown"
        
        if "max_tradeable_at_peak" not in token_data:
            token_data["max_tradeable_at_peak"] = "Unknown"
        
    except Exception as e:
        print(f"Error calculating price history for {token_data['name']}: {e}")
        token_data["price_5days_after_launch"] = "Error"
        token_data["price_10days_after_launch"] = "Error"
        token_data["days_to_peak"] = "Error"
        token_data["max_tradeable_at_peak"] = "Error"
    
    return token_data

async def analyze_all_tokens():
    """Analyze all tokens in the list."""
    initialize_web3_connections()
    
    enriched_tokens = []
    for token in GHIBLI_TOKENS:
        # Fetch contract details
        token = await fetch_token_contract_details(token)
        
        # Get liquidity data
        token = await get_liquidity_data(token)
        
        # Get price history
        token = await get_price_history(token)
        
        enriched_tokens.append(token)
    
    return enriched_tokens

def generate_report(tokens):
    """Generate a report of token analysis."""
    # Convert to DataFrame for easy display
    df = pd.DataFrame(tokens)
    
    # Select and reorder columns for display based on token type
    token_columns = [
        "name", "ticker", "platform", "found_network", "cross_network_match", "launch_date", "contract", "found_type",
        "contract_name", "contract_symbol", "creation_date", 
        "start_price", "peak_price", "actual_peak_price", "end_price", "current_price", "days_to_peak",
        "price_5days_after_launch", "price_10days_after_launch",
        "current_liquidity_usd", "liquidity_pools", "paired_with", "max_tradeable_at_peak",
        "total_supply", "days_since_launch"
    ]
    
    nft_columns = [
        "name", "platform", "found_network", "cross_network_match", "launch_date", "contract", "found_type", 
        "contract_name", "token_type", "creation_date", 
        "total_supply", "unique_owners", "sales_count", "total_volume", "avg_price",
        "opensea_floor_price", "days_since_launch"
    ]
    
    # Create separate DataFrames for tokens and NFTs
    erc20_tokens = df[df["found_type"] == "ERC20"] if "found_type" in df.columns else pd.DataFrame()
    nfts = df[df["found_type"] == "NFT"] if "found_type" in df.columns else pd.DataFrame()
    not_found = df[~df["contract"].notna()] if "contract" in df.columns else df
    
    # Count cross-network matches
    cross_network_matches = df[df.get("cross_network_match") == True] if "cross_network_match" in df.columns else pd.DataFrame()
    logger.info(f"Found {len(cross_network_matches)} tokens on different networks than originally specified")
    if not cross_network_matches.empty:
        logger.info(f"Cross-network matches: {', '.join(cross_network_matches['name'])}")
    
    # Filter columns that exist in each DataFrame
    token_display_columns = [col for col in token_columns if col in df.columns]
    nft_display_columns = [col for col in nft_columns if col in df.columns]
    
    # Generate report
    print("\n=== GHIBLI TOKEN ANALYSIS REPORT ===")
    
    print("\n--- FOUND ERC20 TOKENS ---")
    if not erc20_tokens.empty:
        print(erc20_tokens[token_display_columns].to_string(index=False))
    else:
        print("No ERC20 tokens found")
    
    print("\n--- FOUND NFT COLLECTIONS ---")
    if not nfts.empty:
        print(nfts[nft_display_columns].to_string(index=False))
    else:
        print("No NFT collections found")
    
    print("\n--- TOKENS/NFTS NOT FOUND ---")
    if not not_found.empty:
        print(not_found[["name", "ticker", "platform", "launch_date"]].to_string(index=False))
    
    # Create a summary CSV with key metrics
    summary_columns = [
        "name", "ticker", "platform", "launch_date", "contract", "found_type",
        "start_price", "peak_price", "end_price", "days_to_peak",
        "price_5days_after_launch", "price_10days_after_launch", 
        "current_liquidity_usd", "max_tradeable_at_peak"
    ]
    
    # Filter to columns that exist
    summary_columns = [col for col in summary_columns if col in df.columns]
    
    # Save to CSV files
    df.to_csv("ghibli_tokens_analysis_full.csv", index=False)
    df[summary_columns].to_csv("ghibli_tokens_analysis_summary.csv", index=False)
    print("\nAnalysis saved to:")
    print("- ghibli_tokens_analysis_full.csv (all data)")
    print("- ghibli_tokens_analysis_summary.csv (key metrics)")
    
    # Create advanced metrics
    if not erc20_tokens.empty:
        # Calculate price performance metrics
        try:
            erc20_tokens_with_prices = erc20_tokens[erc20_tokens["price_5days_after_launch"] != "Unknown"].copy()
            if not erc20_tokens_with_prices.empty:
                # Convert string values to float if needed
                for col in ["price_5days_after_launch", "price_10days_after_launch", "start_price", "peak_price", "end_price"]:
                    if col in erc20_tokens_with_prices.columns:
                        erc20_tokens_with_prices[col] = pd.to_numeric(erc20_tokens_with_prices[col], errors='coerce')
                
                # Calculate percentage changes
                erc20_tokens_with_prices["change_5days_pct"] = ((erc20_tokens_with_prices["price_5days_after_launch"] / 
                                                              erc20_tokens_with_prices["start_price"]) - 1) * 100
                
                if "price_10days_after_launch" in erc20_tokens_with_prices.columns:
                    erc20_tokens_with_prices["change_10days_pct"] = ((erc20_tokens_with_prices["price_10days_after_launch"] / 
                                                                  erc20_tokens_with_prices["start_price"]) - 1) * 100
                
                erc20_tokens_with_prices["peak_vs_start_pct"] = ((erc20_tokens_with_prices["peak_price"] / 
                                                              erc20_tokens_with_prices["start_price"]) - 1) * 100
                
                erc20_tokens_with_prices["current_vs_peak_pct"] = ((erc20_tokens_with_prices["end_price"] / 
                                                                erc20_tokens_with_prices["peak_price"]) - 1) * 100
                
                # Save performance metrics
                performance_columns = [
                    "name", "ticker", "platform", "launch_date", 
                    "start_price", "price_5days_after_launch", "price_10days_after_launch", 
                    "peak_price", "end_price", "days_to_peak",
                    "change_5days_pct", "change_10days_pct", "peak_vs_start_pct", "current_vs_peak_pct"
                ]
                
                performance_columns = [col for col in performance_columns if col in erc20_tokens_with_prices.columns]
                erc20_tokens_with_prices[performance_columns].to_csv("ghibli_tokens_performance.csv", index=False)
                print("- ghibli_tokens_performance.csv (price performance metrics)")
                
        except Exception as e:
            print(f"Error calculating advanced metrics: {e}")
    
    # Create a summary dictionary for easier analysis
    summary = {
        "total_tokens_analyzed": len(tokens),
        "erc20_tokens_found": len(erc20_tokens),
        "nft_collections_found": len(nfts),
        "not_found": len(not_found),
        "platforms": df["platform"].value_counts().to_dict(),
        "tokens_by_launch_date": df["launch_date"].value_counts().to_dict(),
    }
    
    # Add aggregate performance metrics if available
    try:
        if not erc20_tokens.empty and "days_to_peak" in erc20_tokens.columns:
            days_to_peak_values = [d for d in erc20_tokens["days_to_peak"] if isinstance(d, (int, float))]
            if days_to_peak_values:
                summary["avg_days_to_peak"] = sum(days_to_peak_values) / len(days_to_peak_values)
        
        if not erc20_tokens_with_prices.empty and "change_5days_pct" in erc20_tokens_with_prices.columns:
            summary["avg_5day_return_pct"] = erc20_tokens_with_prices["change_5days_pct"].mean()
            summary["avg_peak_return_pct"] = erc20_tokens_with_prices["peak_vs_start_pct"].mean()
    except:
        pass
    
    # Save full data to JSON
    output = {
        "summary": summary,
        "tokens": tokens
    }
    
    with open("ghibli_tokens_analysis.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print("Full analysis data saved to ghibli_tokens_analysis.json")
    
    return df

async def main():
    """Main function to run the analysis."""
    print("Starting Ghibli token analysis...")
    tokens = await analyze_all_tokens()
    generate_report(tokens)

if __name__ == "__main__":
    asyncio.run(main())