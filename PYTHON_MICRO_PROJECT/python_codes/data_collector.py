import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import numpy as np
import os

# Etherscan API key
ETHERSCAN_API_KEY = "848HZHG8QKCE4DIMBV7P13CDSUARHXDX4T"

# Function to get normal transactions for an address
def get_normal_transactions(address):
    url = f"https://api.etherscan.io/api?module=account&action=txlist&address={address}&startblock=0&endblock=99999999&sort=asc&apikey={ETHERSCAN_API_KEY}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        if data['status'] == '1':
            return data['result']
        else:
            print(f"API Error: {data['message']}")
            return []
    else:
        print(f"Request failed with status code: {response.status_code}")
        return []

# Function to get internal transactions for an address
def get_internal_transactions(address):
    url = f"https://api.etherscan.io/api?module=account&action=txlistinternal&address={address}&startblock=0&endblock=99999999&sort=asc&apikey={ETHERSCAN_API_KEY}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        if data['status'] == '1':
            return data['result']
        else:
            print(f"API Error: {data['message']}")
            return []
    else:
        print(f"Request failed with status code: {response.status_code}")
        return []

# Function to get ether balance for an address
def get_balance(address):
    url = f"https://api.etherscan.io/api?module=account&action=balance&address={address}&tag=latest&apikey={ETHERSCAN_API_KEY}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        if data['status'] == '1':
            # Convert wei to ether
            return int(data['result']) / 1e18
        else:
            print(f"API Error: {data['message']}")
            return 0
    else:
        print(f"Request failed with status code: {response.status_code}")
        return 0

# Process transactions and calculate features
def calculate_features(address, transactions, internal_transactions):
    # Initialize feature dictionary
    features = {
        'Avg min between sent tnx': 0,
        'Avg min between received tnx': 0,
        'Time Diff between first and last (Mins)': 0,
        'Sent tnx': 0,
        'Received Tnx': 0,
        'Unique Received From Addresses': 0,
        'Unique Sent To Addresses': 0,
        'max value received': 0,
        'avg val received': 0,
        'max val sent': 0,
        'avg val sent': 0,
        'total Ether sent': 0,
        'total ether received': 0,
        'total ether balance': 0,
        'Address': address
    }
    
    # Combine normal and internal transactions
    all_transactions = transactions + internal_transactions
    
    if not all_transactions:
        print(f"No transactions found for address {address}")
        return features
    
    # Sort transactions by timestamp
    all_transactions.sort(key=lambda x: int(x['timeStamp']))
    
    # Separate sent and received transactions
    sent_transactions = [tx for tx in all_transactions if tx['from'].lower() == address.lower()]
    received_transactions = [tx for tx in all_transactions if tx['to'].lower() == address.lower()]
    
    # Calculate time differences for sent transactions
    sent_times = []
    for tx in sent_transactions:
        sent_times.append(int(tx['timeStamp']))
    
    sent_time_diffs = []
    for i in range(1, len(sent_times)):
        diff_minutes = (sent_times[i] - sent_times[i-1]) / 60
        sent_time_diffs.append(diff_minutes)
    
    # Calculate time differences for received transactions
    received_times = []
    for tx in received_transactions:
        received_times.append(int(tx['timeStamp']))
    
    received_time_diffs = []
    for i in range(1, len(received_times)):
        diff_minutes = (received_times[i] - received_times[i-1]) / 60
        received_time_diffs.append(diff_minutes)
    
    # Calculate values for sent transactions
    sent_values = []
    sent_addresses = set()
    total_sent = 0
    for tx in sent_transactions:
        value_eth = int(tx['value']) / 1e18
        sent_values.append(value_eth)
        sent_addresses.add(tx['to'].lower())
        total_sent += value_eth
    
    # Calculate values for received transactions
    received_values = []
    received_addresses = set()
    total_received = 0
    for tx in received_transactions:
        value_eth = int(tx['value']) / 1e18
        received_values.append(value_eth)
        received_addresses.add(tx['from'].lower())
        total_received += value_eth
    
    # Time difference between first and last transaction
    if len(all_transactions) > 1:
        first_tx_time = int(all_transactions[0]['timeStamp'])
        last_tx_time = int(all_transactions[-1]['timeStamp'])
        time_diff_mins = (last_tx_time - first_tx_time) / 60
    else:
        time_diff_mins = 0
    
    # Populate features dictionary
    features['Avg min between sent tnx'] = np.mean(sent_time_diffs) if sent_time_diffs else 0
    features['Avg min between received tnx'] = np.mean(received_time_diffs) if received_time_diffs else 0
    features['Time Diff between first and last (Mins)'] = time_diff_mins
    features['Sent tnx'] = len(sent_transactions)
    features['Received Tnx'] = len(received_transactions)
    features['Unique Received From Addresses'] = len(received_addresses)
    features['Unique Sent To Addresses'] = len(sent_addresses)
    features['max value received'] = max(received_values) if received_values else 0
    features['avg val received'] = np.mean(received_values) if received_values else 0
    features['max val sent'] = max(sent_values) if sent_values else 0
    features['avg val sent'] = np.mean(sent_values) if sent_values else 0
    features['total Ether sent'] = total_sent
    features['total ether received'] = total_received
    
    # Get current balance
    current_balance = get_balance(address)
    features['total ether balance'] = current_balance
    
    return features

def process_wallet_addresses(addresses):
    all_features = []
    
    for i, address in enumerate(addresses):
        print(f"Processing address {i+1}/{len(addresses)}: {address}")
        
        # Get transactions
        normal_txs = get_normal_transactions(address)
        print(f"  Found {len(normal_txs)} normal transactions")
        
        # Wait to avoid hitting API rate limits
        time.sleep(0.2)
        
        internal_txs = get_internal_transactions(address)
        print(f"  Found {len(internal_txs)} internal transactions")
        
        # Calculate features
        features = calculate_features(address, normal_txs, internal_txs)
        all_features.append(features)
        
        # Wait to avoid hitting API rate limits
        time.sleep(0.5)
    
    # Create a DataFrame from features
    df = pd.DataFrame(all_features)
    
    # Ensure columns match the expected format for the model
    expected_columns = [
        'Avg min between sent tnx', 'Avg min between received tnx',
        'Time Diff between first and last (Mins)', 'Sent tnx', 'Received Tnx',
        'Unique Received From Addresses', 'Unique Sent To Addresses',
        'max value received', 'avg val received', 'max val sent', 'avg val sent',
        'total Ether sent', 'total ether received', 'total ether balance', 'Address'
    ]
    
    # Reorder and ensure all columns exist
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0
    
    df = df[expected_columns]
    
    return df

def main():
    print("Ethereum Wallet Data Collector")
    print("-----------------------------")
    
    # Get wallet addresses from user
    input_type = input("Enter 'manual' to input addresses manually or 'file' to read from a file: ").strip().lower()
    
    addresses = []
    
    if input_type == 'manual':
        while True:
            address = input("\nEnter Ethereum wallet address (or 'done' to finish): ").strip()
            if address.lower() == 'done':
                break
            if address.startswith('0x') and len(address) == 42:
                addresses.append(address)
            else:
                print("Invalid Ethereum address format. Please try again.")
    
    elif input_type == 'file':
        file_path = input("Enter the path to the file containing addresses (one per line) or defauls to 'wallet_addresses.csv' : ").strip() or 'csv_files/wallet_addresses.csv'
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    address = line.strip()
                    if address.startswith('0x') and len(address) == 42:
                        addresses.append(address)
        except Exception as e:
            print(f"Error reading file: {e}")
            return
    
    else:
        print("Invalid input type.")
        return
    
    if not addresses:
        print("No valid addresses provided.")
        return
    
    print(f"\nProcessing {len(addresses)} wallet addresses...")
    
    # Process addresses and get features
    wallet_data = process_wallet_addresses(addresses)
    
    # Save to CSV
    output_file = 'csv_files/wallet_data.csv'
    wallet_data.to_csv(output_file, index=False)
    
    print(f"\nData collection complete. Data saved to {output_file}")
    print(f"Found data for {len(wallet_data)} addresses.")
    print("\nPreview of collected data:")
    print(wallet_data.head())
    
    print(f"\nYou can now use this data with your fraud detection model.")

if __name__ == "__main__":
    main()