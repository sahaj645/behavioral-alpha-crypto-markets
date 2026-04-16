"""
Data cleaning module for PrimeTrade.ai sentiment and trading analysis.
Handles datetime parsing, outlier removal, and dataset merging.
"""

import pandas as pd
import numpy as np


def clean_trades(df):
    """
    Clean historical trades dataset.
    
    - Parse time column to datetime and extract date
    - Cast numeric columns to float
    - Remove rows with null closedPnL
    - Remove outliers using IQR method (1.5 * IQR)
    - Print before/after statistics
    
    Args:
        df (pd.DataFrame): Raw trades dataframe
        
    Returns:
        pd.DataFrame: Cleaned trades dataframe
    """
    print("\n" + "="*70)
    print("CLEANING TRADES DATA")
    print("="*70)
    
    df = df.copy()
    rows_before = len(df)
    
    # Parse time to datetime and extract date
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df['date'] = df['time'].dt.date
    
    # Cast numeric columns
    numeric_cols = ['closed_pnl', 'size', 'leverage', 'execution_price']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print(f"\nBefore cleaning:")
    print(f"  Total rows: {rows_before}")
    print(f"  Missing closedPnL: {df['closed_pnl'].isna().sum()}")
    
    # Drop null closedPnL
    df = df.dropna(subset=['closed_pnl'])
    print(f"\nAfter dropping null closedPnL: {len(df)} rows")
    
    # Remove outliers using IQR method
    Q1 = df['closed_pnl'].quantile(0.25)
    Q3 = df['closed_pnl'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    rows_with_outliers = len(df[(df['closed_pnl'] < lower_bound) | (df['closed_pnl'] > upper_bound)])
    print(f"  Outliers detected (IQR method): {rows_with_outliers}")
    
    df = df[(df['closed_pnl'] >= lower_bound) & (df['closed_pnl'] <= upper_bound)]
    
    rows_after = len(df)
    print(f"\nAfter removing outliers: {rows_after} rows")
    print(f"  ✓ Removed {rows_before - rows_after} rows ({100*(rows_before-rows_after)/rows_before:.1f}%)")
    print(f"\n  PnL Stats (cleaned):")
    print(f"    Mean: ${df['closed_pnl'].mean():.2f}")
    print(f"    Median: ${df['closed_pnl'].median():.2f}")
    print(f"    Std: ${df['closed_pnl'].std():.2f}")
    
    return df.reset_index(drop=True)


def clean_sentiment(df):
    """
    Clean fear and greed sentiment dataset.
    
    - Parse Date to datetime and extract date
    - Strip whitespace from Classification
    - Create ordered categorical with proper hierarchy
    - Print value counts
    
    Args:
        df (pd.DataFrame): Raw sentiment dataframe
        
    Returns:
        pd.DataFrame: Cleaned sentiment dataframe
    """
    print("\n" + "="*70)
    print("CLEANING SENTIMENT DATA")
    print("="*70)
    
    df = df.copy()
    
    # Parse date to datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
    else:
        print("\n✗ Warning: 'date' column not found in sentiment data")
    
    # Clean Classification column
    if 'classification' in df.columns:
        df['classification'] = df['classification'].str.strip()
    
    # Create ordered category
    sentiment_order = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
    df['classification'] = pd.Categorical(
        df['classification'],
        categories=sentiment_order,
        ordered=True
    )
    
    print(f"\nSentiment Distribution:")
    print(df['classification'].value_counts(sort=False))
    print(f"\nTotal sentiment records: {len(df)}")
    
    return df.reset_index(drop=True)


def merge_datasets(trades, sentiment):
    """
    Merge trades and sentiment datasets on date (left join).
    
    - Convert dates to datetime for proper merging
    - Perform left join to keep all trades
    - Report merge success rate
    
    Args:
        trades (pd.DataFrame): Cleaned trades dataframe
        sentiment (pd.DataFrame): Cleaned sentiment dataframe
        
    Returns:
        pd.DataFrame: Merged dataframe
    """
    print("\n" + "="*70)
    print("MERGING DATASETS")
    print("="*70)
    
    # Convert date to datetime for merge
    trades_merge = trades.copy()
    sentiment_merge = sentiment.copy()
    
    trades_merge['date'] = pd.to_datetime(trades_merge['date'])
    sentiment_merge['date'] = pd.to_datetime(sentiment_merge['date'])
    
    # Left join: keep all trades, match with sentiment where available
    merged = trades_merge.merge(
        sentiment_merge[['date', 'classification']],
        on='date',
        how='left'
    )
    
    # Calculate merge success rate
    trades_with_sentiment = merged['classification'].notna().sum()
    merge_rate = 100 * trades_with_sentiment / len(merged)
    
    print(f"\nMerge Results:")
    print(f"  Total trades: {len(merged)}")
    print(f"  Trades with sentiment label: {trades_with_sentiment}")
    print(f"  ✓ Merge success rate: {merge_rate:.1f}%")
    
    if merged['classification'].isna().sum() > 0:
        print(f"  ⚠ Unmatched trades (no sentiment): {merged['classification'].isna().sum()}")
    
    return merged
