"""
Visualization module for PrimeTrade.ai sentiment and trading analysis.
Generates 10 production-grade charts for all analysis outputs.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path


# Set global style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

FIGURES_DIR = Path('data/figures')
FIGURES_DIR.mkdir(exist_ok=True)


def bar_pnl_by_sentiment(df):
    """
    Create bar chart: average PnL per sentiment class.
    Color-coded from red (fear) to green (greed).
    
    Args:
        df (pd.DataFrame): Analysis result from pnl_by_sentiment()
    """
    print("\n📊 Generating Chart 1: PnL by Sentiment...")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Color gradient: red → yellow → green
    colors = ['#d62728', '#ff7f0e', '#ffff00', '#90ee90', '#2ca02c']
    
    bars = ax.bar(range(len(df)), df['mean'].values, color=colors, edgecolor='black', linewidth=1.5)
    
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df.index, rotation=45, ha='right')
    ax.set_ylabel('Average PnL ($)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Sentiment Classification', fontsize=12, fontweight='bold')
    ax.set_title('Average Trader PnL by Sentiment Zone', fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, df['mean'].values)):
        ax.text(bar.get_x() + bar.get_width()/2, val + 50, f'${val:.0f}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    filepath = FIGURES_DIR / 'bar_pnl_by_sentiment.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {filepath}")
    plt.close()


def winrate_by_sentiment(df):
    """
    Create horizontal bar chart: win rate per sentiment.
    Sorted by win rate, color-coded by performance.
    
    Args:
        df (pd.DataFrame): Analysis result from win_rate_by_sentiment()
    """
    print("\n📊 Generating Chart 2: Win Rate by Sentiment...")
    
    df_sorted = df.sort_values('win_rate', ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Color by win rate
    colors = plt.cm.RdYlGn(df_sorted['win_rate'] / 100)
    
    bars = ax.barh(range(len(df_sorted)), df_sorted['win_rate'].values, color=colors, edgecolor='black', linewidth=1.5)
    
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted.index)
    ax.set_xlabel('Win Rate (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sentiment Classification', fontsize=12, fontweight='bold')
    ax.set_title('Trader Win Rate by Sentiment Zone', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(0, 100)
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, df_sorted['win_rate'].values)):
        ax.text(val + 1, i, f'{val:.1f}%', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    filepath = FIGURES_DIR / 'winrate_by_sentiment.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {filepath}")
    plt.close()


def long_short_heatmap(df):
    """
    Create seaborn heatmap: sentiment (rows) × Long/Short (cols), values = mean PnL.
    
    Args:
        df (pd.DataFrame): Pivot table from long_short_sentiment_analysis()
    """
    print("\n📊 Generating Chart 3: Long vs Short Heatmap...")
    
    if df is None:
        print("  ⚠ Skipped (no data)")
        return
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    sns.heatmap(df, annot=True, fmt='.2f', cmap='RdYlGn', center=0, 
                cbar_kws={'label': 'Mean PnL ($)'}, linewidths=1, ax=ax)
    
    ax.set_title('Mean PnL: Long vs Short across Sentiment Zones', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Position Side', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sentiment Classification', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    filepath = FIGURES_DIR / 'long_short_heatmap.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {filepath}")
    plt.close()


def top_traders_heatmap(df):
    """
    Create annotated heatmap: top 10 traders (rows) × sentiment zones (cols), values = mean PnL.
    
    Args:
        df (pd.DataFrame): Heatmap data from top_trader_analysis()
    """
    print("\n📊 Generating Chart 4: Top Traders Heatmap...")
    
    if df is None:
        print("  ⚠ Skipped (no data)")
        return
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    sns.heatmap(df, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                cbar_kws={'label': 'Mean PnL ($)'}, linewidths=1, ax=ax, vmin=-500, vmax=500)
    
    ax.set_title('Top 10 Traders: Mean PnL across Sentiment Zones', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Sentiment Classification', fontsize=12, fontweight='bold')
    ax.set_ylabel('Trader Account', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    filepath = FIGURES_DIR / 'top_traders_heatmap.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {filepath}")
    plt.close()


def leverage_vs_sentiment(df):
    """
    Create boxplot of leverage distribution per sentiment class.
    Shows median, quartiles, and outliers.
    
    Args:
        df (pd.DataFrame): Raw merged dataframe
    """
    print("\n📊 Generating Chart 5: Leverage vs Sentiment...")
    
    if 'leverage' not in df.columns or 'classification' not in df.columns:
        print("  ⚠ Skipped (missing columns)")
        return
    
    df_clean = df.dropna(subset=['leverage', 'classification']).copy()
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    sentiment_order = ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']
    df_clean['classification'] = pd.Categorical(df_clean['classification'], 
                                                  categories=sentiment_order, ordered=True)
    
    bp = ax.boxplot([df_clean[df_clean['classification'] == s]['leverage'].values 
                      for s in sentiment_order if s in df_clean['classification'].unique()],
                     labels=[s for s in sentiment_order if s in df_clean['classification'].unique()],
                     patch_artist=True)
    
    # Color boxes
    colors = ['#d62728', '#ff7f0e', '#ffff00', '#90ee90', '#2ca02c']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Leverage (x)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Sentiment Classification', fontsize=12, fontweight='bold')
    ax.set_title('Leverage Distribution by Sentiment Zone', fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    filepath = FIGURES_DIR / 'leverage_vs_sentiment.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {filepath}")
    plt.close()


def pnl_distribution_by_sentiment(df):
    """
    Create violin plot of closedPnL per sentiment class.
    Shows full distribution shape.
    
    Args:
        df (pd.DataFrame): Raw merged dataframe
    """
    print("\n📊 Generating Chart 6: PnL Distribution by Sentiment...")
    
    if 'closed_pnl' not in df.columns or 'classification' not in df.columns:
        print("  ⚠ Skipped (missing columns)")
        return
    
    df_clean = df.dropna(subset=['closed_pnl', 'classification']).copy()
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    sentiment_order = ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']
    df_clean['classification'] = pd.Categorical(df_clean['classification'],
                                                  categories=sentiment_order, ordered=True)
    
    parts = ax.violinplot([df_clean[df_clean['classification'] == s]['closed_pnl'].values 
                            for s in sentiment_order if s in df_clean['classification'].unique()],
                           positions=range(len([s for s in sentiment_order if s in df_clean['classification'].unique()])),
                           showmeans=True, showmedians=True)
    
    ax.set_xticks(range(len([s for s in sentiment_order if s in df_clean['classification'].unique()])))
    ax.set_xticklabels([s for s in sentiment_order if s in df_clean['classification'].unique()], rotation=45, ha='right')
    ax.set_ylabel('Closed PnL ($)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Sentiment Classification', fontsize=12, fontweight='bold')
    ax.set_title('PnL Distribution by Sentiment Zone (Violin Plot)', fontsize=14, fontweight='bold', pad=20)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    filepath = FIGURES_DIR / 'pnl_distribution_by_sentiment.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {filepath}")
    plt.close()


def trade_volume_by_sentiment(df):
    """
    Create pie chart + bar chart side by side for trade volume distribution.
    
    Args:
        df (pd.DataFrame): Raw merged dataframe
    """
    print("\n📊 Generating Chart 7: Trade Volume by Sentiment...")
    
    if 'classification' not in df.columns:
        print("  ⚠ Skipped (missing classification)")
        return
    
    sentiment_counts = df['classification'].value_counts()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    colors = ['#d62728', '#ff7f0e', '#ffff00', '#90ee90', '#2ca02c']
    
    # Pie chart
    ax1.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%',
            colors=colors, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax1.set_title('Trade Volume Distribution (%)', fontsize=12, fontweight='bold')
    
    # Bar chart
    ax2.bar(range(len(sentiment_counts)), sentiment_counts.values, color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_xticks(range(len(sentiment_counts)))
    ax2.set_xticklabels(sentiment_counts.index, rotation=45, ha='right')
    ax2.set_ylabel('Number of Trades', fontsize=11, fontweight='bold')
    ax2.set_title('Trade Count by Sentiment', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    for i, val in enumerate(sentiment_counts.values):
        ax2.text(i, val + 10, str(val), ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    filepath = FIGURES_DIR / 'trade_volume_by_sentiment.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {filepath}")
    plt.close()


def symbol_performance_heatmap(df):
    """
    Create heatmap: top 10 symbols (rows) × sentiment zones (cols), values = mean PnL.
    
    Args:
        df (pd.DataFrame): Heatmap data from symbol_sentiment_analysis()
    """
    print("\n📊 Generating Chart 8: Symbol Performance Heatmap...")
    
    if df is None:
        print("  ⚠ Skipped (no data)")
        return
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    sns.heatmap(df, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                cbar_kws={'label': 'Mean PnL ($)'}, linewidths=1, ax=ax)
    
    ax.set_title('Top Trading Symbols: Mean PnL across Sentiment Zones', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Sentiment Classification', fontsize=12, fontweight='bold')
    ax.set_ylabel('Trading Symbol', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    filepath = FIGURES_DIR / 'symbol_performance_heatmap.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {filepath}")
    plt.close()


def contrarian_vs_momentum(df):
    """
    Create scatter plot: x=PnL in Fear, y=PnL in Greed, color by trader type.
    Contrarian traders in bottom-right, momentum in top-left.
    
    Args:
        df (pd.DataFrame): Classification df from contrarian_vs_momentum_analysis()
    """
    print("\n📊 Generating Chart 9: Contrarian vs Momentum Traders...")
    
    if df is None or 'type' not in df.columns:
        print("  ⚠ Skipped (no data)")
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Scatter plot
    contrarian = df[df['type'] == 'Contrarian']
    momentum = df[df['type'] == 'Momentum']
    
    ax.scatter(contrarian['fear_pnl'], contrarian['greed_pnl'], 
              label='Contrarian', s=100, alpha=0.6, color='#d62728', edgecolor='black')
    ax.scatter(momentum['fear_pnl'], momentum['greed_pnl'],
              label='Momentum', s=100, alpha=0.6, color='#2ca02c', edgecolor='black')
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    
    ax.set_xlabel('Total PnL in Fear Zones ($)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total PnL in Greed Zones ($)', fontsize=12, fontweight='bold')
    ax.set_title('Trader Classification: Contrarian vs Momentum', fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    filepath = FIGURES_DIR / 'contrarian_vs_momentum.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {filepath}")
    plt.close()


def lag_correlation_chart(df):
    """
    Create line chart: x=lag days (-3 to +3), y=correlation with PnL.
    Highlight the lag with strongest signal.
    
    Args:
        df (pd.DataFrame): Lag analysis results
    """
    print("\n📊 Generating Chart 10: Lag Effect Analysis...")
    
    if df is None:
        print("  ⚠ Skipped (no data)")
        return
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Line plot
    ax.plot(df['lag'], df['correlation'], marker='o', linewidth=2.5, markersize=8, color='#1f77b4')
    
    # Highlight best lag
    best_idx = df['correlation'].abs().idxmax()
    best_lag = df.loc[best_idx]
    ax.scatter(best_lag['lag'], best_lag['correlation'], s=300, color='red', zorder=5, edgecolor='black', linewidth=2)
    ax.annotate(f"Best: Lag {int(best_lag['lag'])} days\n(r={best_lag['correlation']:.3f})",
               xy=(best_lag['lag'], best_lag['correlation']),
               xytext=(best_lag['lag']+0.5, best_lag['correlation']+0.05),
               fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
               arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Lag (days)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Correlation with PnL', fontsize=12, fontweight='bold')
    ax.set_title('Lag Effect: Does Today\'s Sentiment Predict Future PnL?', fontsize=14, fontweight='bold', pad=20)
    ax.grid(alpha=0.3)
    ax.set_xticks(sorted(df['lag'].unique()))
    
    plt.tight_layout()
    filepath = FIGURES_DIR / 'lag_correlation_chart.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {filepath}")
    plt.close()


if __name__ == "__main__":
    print("Visualizer module loaded. Import functions and call with analysis results.")
