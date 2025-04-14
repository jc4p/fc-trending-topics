#!/usr/bin/env python3
"""
Analyze token ROI (Return on Investment) for different entry and exit points.
This script analyzes exported price data from find_ghibli_rpc.py to calculate 
how much profit or loss an investor would have realized by buying at different 
points after token creation and selling at various time intervals.
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tabulate import tabulate


def load_price_history(file_path):
    """Load price history data from exported JSON file"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        print(f"Loaded price data for {data.get('token_name', 'Unknown')} ({data.get('token_symbol', 'Unknown')})")
        return data
    except Exception as e:
        print(f"Error loading price data: {e}")
        return None


def prepare_price_dataframe(price_data):
    """Convert price data into a pandas DataFrame with proper timestamps"""
    try:
        # Extract the price data list
        if isinstance(price_data['prices'], list):
            prices = price_data['prices']
        elif isinstance(price_data['prices'], dict) and 'prices' in price_data['prices']:
            prices = price_data['prices']['prices']
        else:
            print(f"Unexpected price data format")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(prices)
        
        # Convert timestamps to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601', errors='coerce')
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Calculate hours since first data point
        first_timestamp = df['timestamp'].min()
        df['hours_since_start'] = (df['timestamp'] - first_timestamp).dt.total_seconds() / 3600
        
        # Ensure no NaN values
        df = df.dropna(subset=['value', 'timestamp'])
        
        # Add time markers for common entry points
        for hour in [6, 12, 18, 24, 36, 48, 72]:
            df[f'hour_{hour}'] = np.abs(df['hours_since_start'] - hour).values
        
        return df
    except Exception as e:
        print(f"Error preparing price data: {e}")
        print(f"Price data structure: {type(price_data['prices'])}")
        if isinstance(price_data['prices'], dict):
            print(f"Keys: {price_data['prices'].keys()}")
        return None


def calculate_roi_matrix(df, investment_amount=100):
    """Calculate ROI matrix for different entry and exit points"""
    # Define entry points (hours after launch)
    entry_hours = [6, 12, 18, 24, 36, 48, 72]
    
    # Define holding periods (in days)
    holding_periods = [0.25, 0.5, 1, 2, 3, 5, 7, 14, 30]
    
    # Create results table
    results = []
    
    for entry_hour in entry_hours:
        # Find closest point to entry hour
        entry_idx = df[f'hour_{entry_hour}'].idxmin()
        entry_price = float(df.loc[entry_idx, 'value'])
        entry_time = df.loc[entry_idx, 'timestamp']
        tokens_bought = investment_amount / entry_price
        
        # Add day of the week and formatted date
        day_of_week = entry_time.strftime('%A')
        date = entry_time.strftime('%Y-%m-%d')
        row = {
            'Entry Hour': entry_hour,
            'Day': day_of_week,
            'Date': date, 
            'Entry Price': entry_price,
            'Entry Time': entry_time
        }
        
        # Calculate ROI for each holding period
        for period in holding_periods:
            # Calculate target exit time
            target_exit_time = entry_time + timedelta(days=period)
            
            # Find closest data point to target exit time
            exit_idx = (df['timestamp'] - target_exit_time).abs().idxmin()
            exit_price = float(df.loc[exit_idx, 'value'])
            exit_time = df.loc[exit_idx, 'timestamp']
            
            # Calculate ROI
            value_at_exit = tokens_bought * exit_price
            roi_pct = ((value_at_exit - investment_amount) / investment_amount) * 100
            
            # Add to row
            row[f'{period}d ROI%'] = roi_pct
            row[f'{period}d Value'] = value_at_exit
            
        results.append(row)
    
    return pd.DataFrame(results)


def generate_roi_report(roi_df, token_data, price_df, investment_amount=100):
    """Generate a detailed ROI report"""
    token_name = token_data.get('token_name', 'Unknown')
    token_symbol = token_data.get('token_symbol', 'Unknown')
    token_address = token_data.get('token_address', 'Unknown')
    
    # Get first and last timestamp
    first_timestamp = price_df['timestamp'].min().strftime('%Y-%m-%d %H:%M:%S')
    last_timestamp = price_df['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S')
    
    # Format the report
    report = [
        f"ROI ANALYSIS FOR {token_name} ({token_symbol})",
        f"Token Address: {token_address}",
        f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Data Range: {first_timestamp} to {last_timestamp}",
        f"Initial Investment: ${investment_amount}",
        f"\nROI MATRIX (Return on ${investment_amount} investment):\n"
    ]
    
    # Format the ROI matrix for display
    display_df = roi_df.copy()
    
    # Format the entry price
    display_df['Entry Price'] = display_df['Entry Price'].apply(lambda x: f"${x:.6f}")
    
    # Format timestamps to be more readable
    display_df['Entry Time'] = display_df['Entry Time'].dt.strftime('%H:%M')
    
    # Format ROI percentages and values
    for col in display_df.columns:
        if 'ROI%' in col:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}%")
        elif 'Value' in col:
            display_df[col] = display_df[col].apply(lambda x: f"${x:.2f}")
    
    # Add the formatted table to the report
    report.append(tabulate(display_df, headers='keys', tablefmt='grid', showindex=False))
    
    # Add comprehensive insights across all ROI periods
    report.append("\nINSIGHTS:")
    
    # Calculate actual creation time (this uses the actual data start time)
    df = roi_df.copy()
    first_entry_date = df.iloc[0]['Entry Time'].replace(hour=0, minute=0, second=0)
    
    # Find best possible ROI across all periods and entry points
    roi_columns = [col for col in df.columns if 'ROI%' in col]
    best_roi_overall = -float('inf')
    best_entry_overall = None
    best_period_overall = None
    
    worst_roi_overall = float('inf')
    worst_entry_overall = None
    worst_period_overall = None
    
    # For each ROI period, find the best and worst entry points
    best_entries_by_period = {}
    worst_entries_by_period = {}
    
    for period_col in roi_columns:
        period_name = period_col.replace(' ROI%', '')
        best_idx = df[period_col].idxmax()
        worst_idx = df[period_col].idxmin()
        
        best_entries_by_period[period_name] = df.loc[best_idx]
        worst_entries_by_period[period_name] = df.loc[worst_idx]
        
        # Check if this is the best overall
        if df.loc[best_idx, period_col] > best_roi_overall:
            best_roi_overall = df.loc[best_idx, period_col]
            best_entry_overall = df.loc[best_idx]
            best_period_overall = period_name
            
        # Check if this is the worst overall
        if df.loc[worst_idx, period_col] < worst_roi_overall:
            worst_roi_overall = df.loc[worst_idx, period_col]
            worst_entry_overall = df.loc[worst_idx]
            worst_period_overall = period_name
    
    # Add detailed explanation about the hour-based entry points
    first_timestamp = price_df['timestamp'].min()
    last_timestamp = price_df['timestamp'].max()
    time_range_days = (last_timestamp - first_timestamp).total_seconds() / 86400
    
    report.append(f"NOTE: This analysis is based on {len(price_df)} data points over {time_range_days:.1f} days")
    report.append(f"      Each 'Entry Hour' represents hours since the first data point ({first_timestamp.strftime('%Y-%m-%d %H:%M:%S')})")
    report.append(f"      • Hour 6 = entry at {(first_timestamp + pd.Timedelta(hours=6)).strftime('%Y-%m-%d %H:%M')}")
    report.append(f"      • Hour 24 = entry at {(first_timestamp + pd.Timedelta(hours=24)).strftime('%Y-%m-%d %H:%M')} (1 day after first data point)")
    report.append(f"      • Hour 72 = entry at {(first_timestamp + pd.Timedelta(hours=72)).strftime('%Y-%m-%d %H:%M')} (3 days after first data point)")
    
    # Report best overall ROI
    report.append(f"Best possible ROI: {best_roi_overall:.2f}% ({best_period_overall}) by entering at Hour {best_entry_overall['Entry Hour']} ({best_entry_overall['Day']}, {best_entry_overall['Date']})")
    report.append(f"Worst possible ROI: {worst_roi_overall:.2f}% ({worst_period_overall}) by entering at Hour {worst_entry_overall['Entry Hour']} ({worst_entry_overall['Day']}, {worst_entry_overall['Date']})")
    
    # Report best entry points for common holding periods
    report.append("\nBest entry points by holding period:")
    for period in ['1d', '3d', '7d', '14d']:
        if period in best_entries_by_period:
            entry = best_entries_by_period[period]
            period_col = f"{period} ROI%"
            report.append(f"  {period}: Hour {entry['Entry Hour']} ({entry['Day']}, {entry['Date']}) with ROI of {entry[period_col]:.2f}%")
    
    return '\n'.join(report)


def save_report(report, token_name, output_dir='output/roi_reports'):
    """Save the ROI report to a file"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename
    filename = f"{output_dir}/{token_name.lower().replace(' ', '_')}_roi_report.txt"
    
    with open(filename, 'w') as f:
        f.write(report)
    
    print(f"ROI report saved to {filename}")
    return filename


def plot_roi_chart(roi_df, token_data, investment_amount=100, output_dir='output/roi_reports'):
    """Generate ROI visualization charts"""
    os.makedirs(output_dir, exist_ok=True)
    token_name = token_data.get('token_name', 'Unknown')
    token_symbol = token_data.get('token_symbol', 'Unknown')
    
    # Prepare data for plotting
    plot_df = roi_df.copy()
    plot_df = plot_df.set_index('Entry Hour')
    
    # Select just the ROI columns
    roi_columns = [col for col in plot_df.columns if 'ROI%' in col]
    roi_data = plot_df[roi_columns]
    
    # Rename columns for better labels
    roi_data.columns = [col.replace('d ROI%', 'd') for col in roi_data.columns]
    
    # Plot heatmap
    plt.figure(figsize=(12, 8))
    
    # Create a custom colormap that's red for negative, green for positive
    cmap = plt.cm.RdYlGn
    
    # Plotting the heatmap
    im = plt.imshow(roi_data.values, cmap=cmap, aspect='auto')
    
    # Adding colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('ROI %')
    
    # Set x and y ticks
    plt.xticks(np.arange(len(roi_data.columns)), roi_data.columns, rotation=45)
    plt.yticks(np.arange(len(roi_data.index)), roi_data.index)
    
    # Add value annotations
    for i in range(len(roi_data.index)):
        for j in range(len(roi_data.columns)):
            text = plt.text(j, i, f"{roi_data.iloc[i, j]:.1f}%",
                           ha="center", va="center", color="black")
    
    plt.xlabel('Holding Period (Days)')
    plt.ylabel('Entry Point (Hours After Launch)')
    plt.title(f'ROI Matrix for {token_name} ({token_symbol}) - ${investment_amount} Investment')
    
    plt.tight_layout()
    
    # Save the chart
    chart_path = f"{output_dir}/{token_name.lower().replace(' ', '_')}_roi_chart.png"
    plt.savefig(chart_path)
    print(f"ROI chart saved to {chart_path}")
    
    # Plot line chart showing ROI over time for different entry points
    plt.figure(figsize=(14, 8))
    
    for col in roi_data.columns:
        plt.plot(roi_data.index, roi_data[col], marker='o', label=f'{col} days')
    
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.xlabel('Entry Point (Hours After Launch)')
    plt.ylabel('ROI %')
    plt.title(f'ROI by Entry Time - {token_name} ({token_symbol})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the line chart
    line_chart_path = f"{output_dir}/{token_name.lower().replace(' ', '_')}_roi_line_chart.png"
    plt.savefig(line_chart_path)
    print(f"ROI line chart saved to {line_chart_path}")
    
    return chart_path, line_chart_path


def find_latest_price_file(price_dir='output/price_data'):
    """Find the most recent price data file, prioritizing files with earlier historical data"""
    if not os.path.exists(price_dir):
        print(f"Price data directory {price_dir} not found")
        return None
    
    # First, check for our special converted file with March data
    march_file = os.path.join(price_dir, "ghibli_2025-03-25_to_2025-03-31_converted.json")
    if os.path.exists(march_file):
        print(f"Found historical data file from March: {march_file}")
        return march_file
    
    # Otherwise, look for other price history files
    files = [f for f in os.listdir(price_dir) if f.endswith('_price_history.json') or f.endswith('_converted.json')]
    
    if not files:
        print(f"No price history files found in {price_dir}")
        return None
    
    # Get the most recent file
    latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(price_dir, f)))
    return os.path.join(price_dir, latest_file)


def main():
    """Main function to analyze token ROI"""
    parser = argparse.ArgumentParser(description='Analyze token ROI for different entry/exit points')
    parser.add_argument('--file', type=str, help='Path to price history JSON file')
    parser.add_argument('--investment', type=float, default=100, help='Initial investment amount in USD (default: $100)')
    args = parser.parse_args()
    
    # Find the price history file
    if args.file and os.path.exists(args.file):
        price_file = args.file
    else:
        print("No file specified or file not found. Searching for the latest price data file...")
        price_file = find_latest_price_file()
        if not price_file:
            print("No price data file found. Please run find_ghibli_rpc.py first or specify a file path.")
            return
    
    print(f"Analyzing price data from: {price_file}")
    
    # Load the price data
    token_data = load_price_history(price_file)
    if not token_data:
        return
    
    # Prepare the DataFrame
    price_df = prepare_price_dataframe(token_data)
    if price_df is None or len(price_df) < 10:
        print("Insufficient price data for analysis")
        return
    
    print(f"Prepared price data with {len(price_df)} data points")
    
    # Calculate ROI matrix
    roi_df = calculate_roi_matrix(price_df, args.investment)
    
    # Generate report
    report = generate_roi_report(roi_df, token_data, price_df, args.investment)
    
    # Save report
    report_file = save_report(report, token_data.get('token_name', 'token'))
    
    # Generate and save charts
    chart_file, line_chart_file = plot_roi_chart(roi_df, token_data, args.investment)
    
    print("\n" + "="*80)
    print(report)
    print("="*80)
    print(f"\nAnalysis complete! See {report_file} for full report.")
    print(f"Charts saved to {chart_file} and {line_chart_file}")


if __name__ == "__main__":
    main()