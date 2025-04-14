#!/usr/bin/env python3
"""
Analyze ROI for a specific trend timestamp, calculating returns at various intervals.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys

def load_price_data(file_path):
    """Load price data from the JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Extract price data
    prices = data.get('prices', [])
    if not prices and 'data' in data:
        prices = data.get('data', [])
    
    return prices, data.get('token_name', 'Unknown'), data.get('token_symbol', 'Unknown')

def find_nearest_price_point(prices, target_timestamp):
    """Find the price data point closest to the target timestamp"""
    if isinstance(target_timestamp, str):
        if '+' in target_timestamp or 'Z' in target_timestamp:
            # Parse ISO format with timezone
            target_dt = datetime.fromisoformat(target_timestamp.replace('Z', '+00:00'))
        else:
            # Parse local datetime and assume UTC
            target_dt = datetime.strptime(target_timestamp, "%Y-%m-%d %H:%M:%S")
    else:
        target_dt = target_timestamp
    
    # Convert all timestamps to UTC naive datetime objects for comparison
    timestamps = []
    for p in prices:
        ts = p['timestamp']
        if '+' in ts or 'Z' in ts:
            dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
            # Convert to naive UTC
            dt = dt.replace(tzinfo=None)
        else:
            dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        timestamps.append(dt)
    
    # Ensure target is also naive
    if hasattr(target_dt, 'tzinfo') and target_dt.tzinfo is not None:
        target_dt = target_dt.replace(tzinfo=None)
    
    # Find the closest timestamp
    time_diffs = [abs((ts - target_dt).total_seconds()) for ts in timestamps]
    closest_idx = time_diffs.index(min(time_diffs))
    
    print(f"Closest price point to {target_timestamp} is at {timestamps[closest_idx]}")
    return prices[closest_idx], timestamps[closest_idx]

def calculate_roi(prices, entry_time, intervals, investment=50):
    """Calculate ROI for various time intervals from the entry point"""
    # First find the closest price point to entry time
    entry_price_data, actual_entry_time = find_nearest_price_point(prices, entry_time)
    entry_price = float(entry_price_data['value'])
    tokens_bought = investment / entry_price
    
    results = {
        'entry_time': actual_entry_time.isoformat(),
        'entry_price': entry_price,
        'tokens_bought': tokens_bought,
        'intervals': []
    }
    
    # Calculate ROI for each interval
    for hours in intervals:
        target_exit_time = actual_entry_time + timedelta(hours=hours)
        exit_price_data, actual_exit_time = find_nearest_price_point(prices, target_exit_time)
        exit_price = float(exit_price_data['value'])
        
        value_at_exit = tokens_bought * exit_price
        roi_amount = value_at_exit - investment
        roi_percent = (roi_amount / investment) * 100
        
        results['intervals'].append({
            'hours': hours,
            'exit_time': actual_exit_time.isoformat(),
            'exit_price': exit_price,
            'value': value_at_exit,
            'roi_amount': roi_amount,
            'roi_percent': roi_percent
        })
    
    return results

def print_roi_table(results, token_name, token_symbol, investment):
    """Print a formatted table of ROI results"""
    print(f"\n{'='*80}")
    print(f"ROI ANALYSIS FOR {token_name} ({token_symbol})")
    print(f"Entry Time: {results['entry_time']}")
    print(f"Entry Price: ${results['entry_price']:.6f}")
    print(f"Initial Investment: ${investment:.2f}")
    print(f"Tokens Purchased: {results['tokens_bought']:.4f} {token_symbol}")
    print(f"{'-'*80}")
    print(f"{'Hours':^10}|{'Exit Time':^25}|{'Exit Price':^12}|{'Value':^12}|{'ROI $':^12}|{'ROI %':^12}")
    print(f"{'-'*80}")
    
    for interval in results['intervals']:
        print(f"{interval['hours']:^10}|{interval['exit_time']:^25}|${interval['exit_price']:<11.6f}|${interval['value']:<11.2f}|${interval['roi_amount']:<11.2f}|{interval['roi_percent']:<11.2f}%")
    
    print(f"{'='*80}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python analyze_trend_roi.py <price_data_file> <trend_timestamp> [investment_amount]")
        return
    
    price_file = sys.argv[1]
    trend_timestamp = sys.argv[2]
    investment = float(sys.argv[3]) if len(sys.argv) > 3 else 50
    
    # Define the intervals to analyze (in hours)
    intervals = [4, 6, 8, 12, 18, 24, 36, 48, 72, 96, 120, 168]
    
    # Also add a special option to analyze the maximum price point
    add_max_price = True
    
    try:
        prices, token_name, token_symbol = load_price_data(price_file)
        
        # Find the absolute maximum price point in our dataset
        max_price_data = max(prices, key=lambda p: float(p['value']))
        max_price = float(max_price_data['value'])
        max_price_time = datetime.fromisoformat(max_price_data['timestamp'].replace('Z', '+00:00')).replace(tzinfo=None)
        print(f"Absolute maximum price in dataset: ${max_price} at {max_price_time}")
        
        results = calculate_roi(prices, trend_timestamp, intervals, investment)
        entry_price = results['entry_price']
        tokens_bought = results['tokens_bought']
        
        # Calculate the absolute best possible ROI (buying at entry, selling at global maximum)
        max_value = tokens_bought * max_price
        max_roi_percent = ((max_value - investment) / investment) * 100
        
        # Add this to the results for comparison
        results['absolute_max'] = {
            'exit_time': max_price_time.isoformat(),
            'exit_price': max_price,
            'value': max_value,
            'roi_amount': max_value - investment,
            'roi_percent': max_roi_percent
        }
        
        print_roi_table(results, token_name, token_symbol, investment)
        
        # Find best ROI timeframe among intervals
        best_interval = max(results['intervals'], key=lambda x: x['roi_percent'])
        print(f"\nBest ROI from predefined intervals: {best_interval['roi_percent']:.2f}% after {best_interval['hours']} hours")
        print(f"${investment:.2f} → ${best_interval['value']:.2f} on {best_interval['exit_time']}")
        
        # Report on the absolute maximum possible ROI
        print(f"\nABSOLUTE MAXIMUM POSSIBLE ROI: {max_roi_percent:.2f}%")
        print(f"${investment:.2f} → ${max_value:.2f} by selling at the global maximum on {max_price_time}")
        entry_time = datetime.fromisoformat(results['entry_time'].replace('Z', '+00:00')).replace(tzinfo=None)
        hours_to_max = (max_price_time - entry_time).total_seconds() / 3600
        print(f"This would require holding for {hours_to_max:.2f} hours after entry")
        
        # Find worst ROI timeframe
        worst_interval = min(results['intervals'], key=lambda x: x['roi_percent'])
        print(f"\nWorst ROI: {worst_interval['roi_percent']:.2f}% after {worst_interval['hours']} hours")
        print(f"${investment:.2f} → ${worst_interval['value']:.2f} on {worst_interval['exit_time']}")
        
    except Exception as e:
        print(f"Error analyzing ROI: {e}")

if __name__ == "__main__":
    main()