import os
import pandas as pd
import numpy as np

print("=== Exploring Data Directory Structure ===")

# Check both common data directories
data_dirs = ['/mnt/data/', '/mnt/imported/data/']

for data_dir in data_dirs:
    print(f"\n--- Checking {data_dir} ---")
    if os.path.exists(data_dir):
        print(f"Directory exists: {data_dir}")
        for root, dirs, files in os.walk(data_dir):
            level = root.replace(data_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f'{indent}{os.path.basename(root)}/')
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    size = os.path.getsize(file_path)
                    print(f'{subindent}{file} ({size} bytes)')
                except:
                    print(f'{subindent}{file} (size unknown)')
    else:
        print(f"Directory does not exist: {data_dir}")

# Try to identify and analyze any CSV files found
print("\n=== Analyzing Data Files ===")
for data_dir in data_dirs:
    if os.path.exists(data_dir):
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(root, file)
                    print(f"\n--- Analyzing {file_path} ---")
                    try:
                        df = pd.read_csv(file_path)
                        print(f"Shape: {df.shape}")
                        print(f"Columns: {list(df.columns)}")
                        print(f"Data types:\n{df.dtypes}")
                        print(f"First few rows:\n{df.head()}")
                        print(f"Basic statistics:\n{df.describe()}")
                        
                        # Check for missing values
                        missing = df.isnull().sum()
                        if missing.sum() > 0:
                            print(f"Missing values:\n{missing[missing > 0]}")
                        else:
                            print("No missing values found")
                            
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}") 