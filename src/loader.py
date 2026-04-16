"""
Data loader module for PrimeTrade.ai sentiment and trading analysis.
Handles loading, validation, and initial exploration of datasets.
"""

import pandas as pd
import sys


def load_trades(path):
    """
    Load and validate historical trades dataset.
    
    Args:
        path (str): Path to historical_trades.csv
        
    Returns:
        pd.DataFrame: Loaded trades dataframe with normalized column names
        
    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file is empty or corrupted
    """
    try:
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError("Trades CSV is empty")
        
        # Normalize column names
        df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
        
        print(f"\nLoaded trades data from {path}")
        print(f"  Shape: {df.shape[0]} rows x {df.shape[1]} columns")
        print(f"  Columns: {list(df.columns)}")
        print(f"\n  Data types:")
        print(df.dtypes)
        print(f"\n  First few rows:")
        print(df.head(3))
        
        return df
    
    except FileNotFoundError as e:
        print(f"Error: Could not find trades file at {path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading trades: {str(e)}")
        sys.exit(1)


def load_sentiment(path):
    """
    Load and validate fear and greed sentiment dataset.
    
    Args:
        path (str): Path to fear_greed.csv
        
    Returns:
        pd.DataFrame: Loaded sentiment dataframe with normalized column names
        
    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file is empty or corrupted
    """
    try:
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError("Sentiment CSV is empty")
        
        # Normalize column names
        df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
        
        print(f"\nLoaded sentiment data from {path}")
        print(f"  Shape: {df.shape[0]} rows x {df.shape[1]} columns")
        print(f"  Columns: {list(df.columns)}")
        print(f"\n  Data types:")
        print(df.dtypes)
        print(f"\n  First few rows:")
        print(df.head(3))
        
        return df
    
    except FileNotFoundError as e:
        print(f"Error: Could not find sentiment file at {path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading sentiment: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    trades = load_trades("data/raw/historical_trades.csv")
    sentiment = load_sentiment("data/raw/fear_greed.csv")
