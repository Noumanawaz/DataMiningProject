"""
Script to download time series anomaly detection datasets.
"""

import os
import argparse
import urllib.request
import zipfile
import tarfile
from pathlib import Path
import shutil


def download_file(url: str, dest_path: str):
    """Download a file from URL."""
    print(f'Downloading {url}...')
    urllib.request.urlretrieve(url, dest_path)
    print(f'Downloaded to {dest_path}')


def extract_zip(zip_path: str, extract_to: str):
    """Extract zip file."""
    print(f'Extracting {zip_path}...')
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f'Extracted to {extract_to}')


def extract_tar(tar_path: str, extract_to: str):
    """Extract tar file."""
    print(f'Extracting {tar_path}...')
    with tarfile.open(tar_path, 'r:*') as tar_ref:
        tar_ref.extractall(extract_to)
    print(f'Extracted to {extract_to}')


def download_smap_msl(data_dir: str):
    """Download SMAP and MSL datasets."""
    base_url = "https://github.com/elisejiuqizhang/TS-AD-Datasets/raw/main/"
    
    datasets = ['SMAP', 'MSL']
    
    for dataset in datasets:
        dataset_dir = os.path.join(data_dir, dataset)
        os.makedirs(dataset_dir, exist_ok=True)
        
        for split in ['train', 'test']:
            split_dir = os.path.join(dataset_dir, split)
            os.makedirs(split_dir, exist_ok=True)
            
            # Download data
            data_url = f"{base_url}{dataset}/{split}/data.npy"
            data_path = os.path.join(split_dir, 'data.npy')
            
            try:
                download_file(data_url, data_path)
            except Exception as e:
                print(f'Error downloading {data_url}: {e}')
            
            # Download labels for test set
            if split == 'test':
                label_url = f"{base_url}{dataset}/{split}/label.npy"
                label_path = os.path.join(split_dir, 'label.npy')
                
                try:
                    download_file(label_url, label_path)
                except Exception as e:
                    print(f'Error downloading {label_url}: {e}')


def download_smd(data_dir: str):
    """Download SMD (Server Machine Dataset)."""
    base_url = "https://github.com/elisejiuqizhang/TS-AD-Datasets/raw/main/SMD/"
    
    smd_dir = os.path.join(data_dir, 'SMD')
    os.makedirs(smd_dir, exist_ok=True)
    
    # Create directories
    train_dir = os.path.join(smd_dir, 'train')
    test_dir = os.path.join(smd_dir, 'test')
    test_label_dir = os.path.join(smd_dir, 'test_label')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(test_label_dir, exist_ok=True)
    
    # Download machine files (example: machine-1-1 to machine-1-8)
    machines = [f'machine-1-{i}' for i in range(1, 9)]
    
    for machine in machines:
        # Train data
        train_url = f"{base_url}train/{machine}.npy"
        train_path = os.path.join(train_dir, f'{machine}.npy')
        
        try:
            download_file(train_url, train_path)
        except Exception as e:
            print(f'Error downloading {train_url}: {e}')
        
        # Test data
        test_url = f"{base_url}test/{machine}.npy"
        test_path = os.path.join(test_dir, f'{machine}.npy')
        
        try:
            download_file(test_url, test_path)
        except Exception as e:
            print(f'Error downloading {test_url}: {e}')
        
        # Test labels
        label_url = f"{base_url}test_label/{machine}.npy"
        label_path = os.path.join(test_label_dir, f'{machine}.npy')
        
        try:
            download_file(label_url, label_path)
        except Exception as e:
            print(f'Error downloading {label_url}: {e}')


def download_ebay(data_dir: str):
    """Download eBay anomalies dataset."""
    base_url = "https://github.com/elisejiuqizhang/TS-AD-Datasets/raw/main/eBay/"
    
    ebay_dir = os.path.join(data_dir, 'eBay')
    os.makedirs(ebay_dir, exist_ok=True)
    
    for split in ['train', 'test']:
        split_dir = os.path.join(ebay_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        
        # Download data
        data_url = f"{base_url}{split}/data.npy"
        data_path = os.path.join(split_dir, 'data.npy')
        
        try:
            download_file(data_url, data_path)
        except Exception as e:
            print(f'Error downloading {data_url}: {e}')
        
        # Download labels for test set
        if split == 'test':
            label_url = f"{base_url}{split}/label.npy"
            label_path = os.path.join(split_dir, 'label.npy')
            
            try:
                download_file(label_url, label_path)
            except Exception as e:
                print(f'Error downloading {label_url}: {e}')


def main():
    parser = argparse.ArgumentParser(description='Download Time Series Anomaly Detection Datasets')
    parser.add_argument('--data_dir', type=str, default='./data/raw', help='Directory to save datasets')
    parser.add_argument('--dataset', type=str, default='all', 
                       choices=['all', 'smap', 'msl', 'smd', 'ebay'],
                       help='Dataset to download')
    args = parser.parse_args()
    
    os.makedirs(args.data_dir, exist_ok=True)
    
    print(f'Downloading datasets to: {args.data_dir}')
    
    if args.dataset == 'all' or args.dataset == 'smap':
        print('\n=== Downloading SMAP dataset ===')
        download_smap_msl(args.data_dir)
    
    if args.dataset == 'all' or args.dataset == 'msl':
        print('\n=== Downloading MSL dataset ===')
        download_smap_msl(args.data_dir)
    
    if args.dataset == 'all' or args.dataset == 'smd':
        print('\n=== Downloading SMD dataset ===')
        download_smd(args.data_dir)
    
    if args.dataset == 'all' or args.dataset == 'ebay':
        print('\n=== Downloading eBay dataset ===')
        download_ebay(args.data_dir)
    
    print('\nDownload completed!')


if __name__ == '__main__':
    main()

