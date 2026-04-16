"""
Core analysis module for PrimeTrade.ai sentiment and trading analysis.
Implements 8 quantitative analyses across sentiment zones and trader behavior.
"""

import pandas as pd
import numpy as np
from scipy import stats


def pnl_by_sentiment(df):
    """
    Analyze PnL distribution across sentiment zones.
    
    Returns: mean, median, std, count, and total PnL per sentiment classification,
    plus % contribution of each sentiment zone to total PnL.
    
    Args:
        df (pd.DataFrame): Merged trades + sentiment dataframe
        
    Returns:
        pd.DataFrame: PnL statistics by sentiment
    """
    print("\n" + "="*70)
    print("ANALYSIS 1: PnL BY SENTIMENT")
    print("="*70)
    
    # Remove trades without sentiment label
    df_analysis = df.dropna(subset=['classification'])
    
    result = df_analysis.groupby('classification', observed=True)['closed_pnl'].agg([
        ('mean', 'mean'),
        ('median', 'median'),
        ('std', 'std'),
        ('count', 'count'),
        ('total_pnl', 'sum')
    ]).round(2)
    
    # Calculate % of total PnL
    total_pnl = result['total_pnl'].sum()
    result['pnl_pct'] = (100 * result['total_pnl'] / total_pnl).round(2)
    
    print("\nPnL Statistics by Sentiment Zone:")
    print(result)
    
    print("\n✓ Key Insights:")
    best_sentiment = result['mean'].idxmax()
    worst_sentiment = result['mean'].idxmin()
    print(f"  - Best avg PnL: {best_sentiment} (${result.loc[best_sentiment, 'mean']:.2f})")
    print(f"  - Worst avg PnL: {worst_sentiment} (${result.loc[worst_sentiment, 'mean']:.2f})")
    print(f"  - Total PnL across all trades: ${total_pnl:,.2f}")
    
    return result


def win_rate_by_sentiment(df):
    """
    Calculate win rate and profitability metrics per sentiment zone.
    Win rate = % of trades where closedPnL > 0.
    Also splits by Long/Short side.
    
    Args:
        df (pd.DataFrame): Merged trades + sentiment dataframe
        
    Returns:
        pd.DataFrame: Win rate statistics by sentiment and side
    """
    print("\n" + "="*70)
    print("ANALYSIS 2: WIN RATE BY SENTIMENT")
    print("="*70)
    
    # Remove trades without sentiment label
    df_analysis = df.dropna(subset=['classification']).copy()
    df_analysis['is_win'] = df_analysis['closed_pnl'] > 0
    
    # Overall win rate by sentiment
    win_rates = df_analysis.groupby('classification', observed=True).agg({
        'is_win': ['sum', 'count'],
        'closed_pnl': 'mean'
    }).round(3)
    
    win_rates.columns = ['wins', 'total_trades', 'avg_pnl']
    win_rates['win_rate'] = (100 * win_rates['wins'] / win_rates['total_trades']).round(2)
    
    print("\nOverall Win Rate by Sentiment:")
    print(win_rates[['total_trades', 'wins', 'win_rate', 'avg_pnl']])
    
    # Win rate split by Long/Short
    if 'side' in df_analysis.columns:
        print("\n\nWin Rate by Sentiment and Side (Long/Short):")
        side_analysis = df_analysis.groupby(['classification', 'side'], observed=True).agg({
            'is_win': ['sum', 'count'],
            'closed_pnl': 'mean'
        }).round(3)
        
        side_analysis.columns = ['wins', 'total_trades', 'avg_pnl']
        side_analysis['win_rate'] = (100 * side_analysis['wins'] / side_analysis['total_trades']).round(2)
        print(side_analysis)
    
    print("\n✓ Key Insights:")
    print(f"  - Overall win rate: {(df_analysis['is_win'].sum() / len(df_analysis) * 100):.1f}%")
    best_wr = win_rates['win_rate'].idxmax()
    print(f"  - Best win rate: {best_wr} ({win_rates.loc[best_wr, 'win_rate']:.1f}%)")
    
    return win_rates


def long_short_sentiment_analysis(df):
    """
    Pivot analysis: sentiment × side (Long/Short) on mean PnL.
    Which sentiment is best for longs? Which for shorts?
    
    Args:
        df (pd.DataFrame): Merged trades + sentiment dataframe
        
    Returns:
        pd.DataFrame: Pivot table of mean PnL by sentiment and side
    """
    print("\n" + "="*70)
    print("ANALYSIS 3: LONG vs SHORT BY SENTIMENT")
    print("="*70)
    
    if 'side' not in df.columns:
        print("⚠ 'side' column not found. Skipping analysis.")
        return None
    
    df_analysis = df.dropna(subset=['classification']).copy()
    
    # Pivot: rows=sentiment, cols=side, values=mean PnL
    pivot = df_analysis.pivot_table(
        index='classification',
        columns='side',
        values='closed_pnl',
        aggfunc='mean'
    ).round(2)
    
    print("\nMean PnL by Sentiment and Side:")
    print(pivot)
    
    # Also show counts
    pivot_counts = df_analysis.pivot_table(
        index='classification',
        columns='side',
        values='closed_pnl',
        aggfunc='count'
    )
    print("\nTrade counts by Sentiment and Side:")
    print(pivot_counts)
    
    print("\n✓ Key Insights:")
    if 'Long' in pivot.columns:
        best_long_sentiment = pivot['Long'].idxmax()
        print(f"  - Best sentiment for LONG positions: {best_long_sentiment} (${pivot.loc[best_long_sentiment, 'Long']:.2f})")
    if 'Short' in pivot.columns:
        best_short_sentiment = pivot['Short'].idxmax()
        print(f"  - Best sentiment for SHORT positions: {best_short_sentiment} (${pivot.loc[best_short_sentiment, 'Short']:.2f})")
    
    return pivot


def top_trader_analysis(df):
    """
    Identify top 10 traders by total closedPnL.
    For each: total PnL, win rate, avg leverage, preferred sentiment zone.
    Create heatmap-ready data: top traders × sentiment zones.
    
    Args:
        df (pd.DataFrame): Merged trades + sentiment dataframe
        
    Returns:
        tuple: (top_traders df, heatmap_data)
    """
    print("\n" + "="*70)
    print("ANALYSIS 4: TOP TRADER PROFILES")
    print("="*70)
    
    if 'account' not in df.columns:
        print("⚠ 'account' column not found. Skipping analysis.")
        return None, None
    
    df_analysis = df.copy()
    
    # Top 10 traders by total PnL
    trader_stats = df_analysis.groupby('account').agg({
        'closed_pnl': ['sum', 'count', 'mean', 'std'],
        'leverage': 'mean',
    }).round(2)
    
    trader_stats.columns = ['total_pnl', 'num_trades', 'avg_pnl', 'pnl_std', 'avg_leverage']
    trader_stats['win_rate'] = (df_analysis.groupby('account')['closed_pnl'].apply(lambda x: (x > 0).sum() / len(x))).round(3)
    trader_stats = trader_stats.sort_values('total_pnl', ascending=False).head(10)
    
    print("\nTop 10 Traders by Total PnL:")
    print(trader_stats)
    
    # Heatmap data: top traders × sentiment
    top_traders_list = trader_stats.index.tolist()
    df_top = df_analysis[df_analysis['account'].isin(top_traders_list)]
    
    if 'classification' in df_top.columns:
        heatmap_data = df_top.dropna(subset=['classification']).pivot_table(
            index='account',
            columns='classification',
            values='closed_pnl',
            aggfunc='mean'
        ).round(2)
        
        heatmap_data = heatmap_data.loc[top_traders_list]
        print("\n\nTop Traders × Sentiment Heatmap Data (Mean PnL):")
        print(heatmap_data)
    else:
        heatmap_data = None
    
    print("\n✓ Key Insights:")
    top_trader = trader_stats.index[0]
    print(f"  - Top trader: {top_trader}")
    print(f"    Total PnL: ${trader_stats.loc[top_trader, 'total_pnl']:,.2f}")
    print(f"    Win rate: {trader_stats.loc[top_trader, 'win_rate']*100:.1f}%")
    print(f"    Num trades: {int(trader_stats.loc[top_trader, 'num_trades'])}")
    
    return trader_stats, heatmap_data


def leverage_sentiment_analysis(df):
    """
    Analyze leverage behavior across sentiment zones.
    - Avg leverage per sentiment
    - Correlation: leverage vs closedPnL
    - Do traders take more risk during greed?
    
    Args:
        df (pd.DataFrame): Merged trades + sentiment dataframe
        
    Returns:
        pd.DataFrame: Leverage statistics by sentiment
    """
    print("\n" + "="*70)
    print("ANALYSIS 5: LEVERAGE BY SENTIMENT")
    print("="*70)
    
    if 'leverage' not in df.columns:
        print("⚠ 'leverage' column not found. Skipping analysis.")
        return None
    
    df_analysis = df.dropna(subset=['classification']).copy()
    
    leverage_stats = df_analysis.groupby('classification', observed=True)['leverage'].agg([
        ('mean', 'mean'),
        ('median', 'median'),
        ('std', 'std'),
        ('min', 'min'),
        ('max', 'max')
    ]).round(2)
    
    print("\nLeverage Statistics by Sentiment:")
    print(leverage_stats)
    
    # Correlation: leverage vs PnL
    correlation = df_analysis[['leverage', 'closed_pnl']].corr().iloc[0, 1]
    print(f"\n✓ Leverage-PnL Correlation: {correlation:.3f}")
    
    # Do traders use more leverage during greed?
    greed_leverage = df_analysis[df_analysis['classification'].isin(['Greed', 'Extreme Greed'])]['leverage'].mean()
    fear_leverage = df_analysis[df_analysis['classification'].isin(['Fear', 'Extreme Fear'])]['leverage'].mean()
    
    print(f"\nLeverage Behavior:")
    print(f"  - Avg leverage in Greed zones: {greed_leverage:.2f}x")
    print(f"  - Avg leverage in Fear zones: {fear_leverage:.2f}x")
    if greed_leverage > fear_leverage:
        print(f"  - ✓ Traders DO use MORE leverage during greed (+{(greed_leverage/fear_leverage - 1)*100:.1f}%)")
    else:
        print(f"  - Traders use LESS leverage during greed ({(1 - greed_leverage/fear_leverage)*100:.1f}% lower)")
    
    return leverage_stats


def symbol_sentiment_analysis(df):
    """
    Identify which trading pairs (symbols) perform best under which sentiment.
    Top 5 symbols per sentiment zone by mean PnL.
    
    Args:
        df (pd.DataFrame): Merged trades + sentiment dataframe
        
    Returns:
        pd.DataFrame: Heatmap data (top symbols × sentiment)
    """
    print("\n" + "="*70)
    print("ANALYSIS 6: SYMBOL PERFORMANCE BY SENTIMENT")
    print("="*70)
    
    if 'symbol' not in df.columns:
        print("⚠ 'symbol' column not found. Skipping analysis.")
        return None
    
    df_analysis = df.dropna(subset=['classification']).copy()
    
    # Top 5 symbols overall by trade volume
    top_symbols = df_analysis['symbol'].value_counts().head(10).index.tolist()
    
    # Heatmap: top symbols × sentiment
    heatmap_data = df_analysis[df_analysis['symbol'].isin(top_symbols)].pivot_table(
        index='symbol',
        columns='classification',
        values='closed_pnl',
        aggfunc='mean'
    ).round(2)
    
    print("\nTop Symbols × Sentiment Heatmap Data (Mean PnL):")
    print(heatmap_data)
    
    # Stats per sentiment
    print("\n\nTop 5 Symbols per Sentiment Zone:")
    for sentiment in ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']:
        sentiment_data = df_analysis[df_analysis['classification'] == sentiment].groupby('symbol')['closed_pnl'].agg(['mean', 'count']).sort_values('mean', ascending=False).head(5)
        if len(sentiment_data) > 0:
            print(f"\n{sentiment}:")
            print(sentiment_data.round(2))
    
    print("\n✓ Key Insights:")
    print(f"  - Analyzed {len(top_symbols)} top trading pairs across sentiment zones")
    
    return heatmap_data


def contrarian_vs_momentum_analysis(df):
    """
    Classify traders as contrarian or momentum:
    - Contrarian: profits more in Fear zones
    - Momentum: profits more in Greed zones
    
    Report: count by type, which has higher total PnL?
    
    Args:
        df (pd.DataFrame): Merged trades + sentiment dataframe
        
    Returns:
        pd.DataFrame: Trader classification (account, type, fear_pnl, greed_pnl, total_pnl)
    """
    print("\n" + "="*70)
    print("ANALYSIS 7: CONTRARIAN vs MOMENTUM TRADERS")
    print("="*70)
    
    if 'account' not in df.columns or 'classification' not in df.columns:
        print("⚠ Required columns not found. Skipping analysis.")
        return None
    
    df_analysis = df.dropna(subset=['classification']).copy()
    
    # Per trader: PnL in Fear zones vs Greed zones
    fear_trades = df_analysis[df_analysis['classification'].isin(['Fear', 'Extreme Fear'])].groupby('account')['closed_pnl'].sum()
    greed_trades = df_analysis[df_analysis['classification'].isin(['Greed', 'Extreme Greed'])].groupby('account')['closed_pnl'].sum()
    neutral_trades = df_analysis[df_analysis['classification'] == 'Neutral'].groupby('account')['closed_pnl'].sum()
    
    classifier = pd.DataFrame({
        'fear_pnl': fear_trades,
        'greed_pnl': greed_trades,
        'neutral_pnl': neutral_trades
    }).fillna(0)
    
    classifier['total_pnl'] = classifier['fear_pnl'] + classifier['greed_pnl'] + classifier['neutral_pnl']
    
    # Classify: contrarian if fear_pnl > greed_pnl, else momentum
    classifier['type'] = classifier.apply(
        lambda row: 'Contrarian' if row['fear_pnl'] > row['greed_pnl'] else 'Momentum',
        axis=1
    )
    
    classifier = classifier.sort_values('total_pnl', ascending=False)
    
    print("\nTrader Classification (Top 20):")
    print(classifier.head(20).round(2))
    
    # Summary stats
    contrarian_traders = (classifier['type'] == 'Contrarian').sum()
    momentum_traders = (classifier['type'] == 'Momentum').sum()
    
    contrarian_total = classifier[classifier['type'] == 'Contrarian']['total_pnl'].sum()
    momentum_total = classifier[classifier['type'] == 'Momentum']['total_pnl'].sum()
    
    print(f"\n✓ Trader Type Distribution:")
    print(f"  - Contrarian traders: {contrarian_traders}")
    print(f"  - Momentum traders: {momentum_traders}")
    print(f"\n✓ Total PnL by Type:")
    print(f"  - Contrarian total PnL: ${contrarian_total:,.2f}")
    print(f"  - Momentum total PnL: ${momentum_total:,.2f}")
    
    if contrarian_total > momentum_total:
        print(f"\n  ✓ CONTRARIAN traders have {(contrarian_total/momentum_total - 1)*100:.1f}% higher total PnL")
    else:
        print(f"\n  ✓ MOMENTUM traders have {(momentum_total/contrarian_total - 1)*100:.1f}% higher total PnL")
    
    return classifier


def lag_effect_analysis(df):
    """
    Does sentiment TODAY predict PnL TOMORROW?
    
    Create sentiment score (numeric), shift by 1/2/3 days,
    calculate correlation with next-day PnL.
    Report which lag has strongest signal.
    
    Args:
        df (pd.DataFrame): Merged trades + sentiment dataframe
        
    Returns:
        pd.DataFrame: Correlation by lag
    """
    print("\n" + "="*70)
    print("ANALYSIS 8: LAG EFFECT - SENTIMENT PREDICTING FUTURE PnL")
    print("="*70)
    
    if 'classification' not in df.columns or 'date' not in df.columns:
        print("⚠ Required columns not found. Skipping analysis.")
        return None
    
    df_analysis = df.dropna(subset=['classification']).copy()
    df_analysis['date'] = pd.to_datetime(df_analysis['date'])
    
    # Convert sentiment to numeric score
    sentiment_map = {
        'Extreme Fear': 1,
        'Fear': 2,
        'Neutral': 3,
        'Greed': 4,
        'Extreme Greed': 5
    }
    df_analysis['sentiment_score'] = df_analysis['classification'].map(sentiment_map)
    
    # Daily aggregation: avg sentiment score and PnL per date
    daily = df_analysis.groupby('date').agg({
        'sentiment_score': 'mean',
        'closed_pnl': ['sum', 'mean']
    }).reset_index()
    
    daily.columns = ['date', 'sentiment_score', 'daily_pnl_sum', 'daily_pnl_mean']
    daily = daily.sort_values('date').reset_index(drop=True)
    
    # Calculate lagged correlations
    lag_results = {}
    for lag in range(-3, 4):
        if lag < 0:
            # Negative lag: past PnL vs current sentiment
            x = daily['sentiment_score'].shift(abs(lag)).dropna()
            y = daily['daily_pnl_mean'].iloc[abs(lag):].values
        elif lag > 0:
            # Positive lag: current sentiment vs future PnL
            x = daily['sentiment_score'].iloc[:-lag].values
            y = daily['daily_pnl_mean'].shift(-lag).dropna()
        else:
            # No lag
            x = daily['sentiment_score']
            y = daily['daily_pnl_mean']
        
        if len(x) > 3 and len(y) > 3:
            corr = np.corrcoef(x, y)[0, 1]
            lag_results[lag] = corr
    
    lag_df = pd.DataFrame(list(lag_results.items()), columns=['lag', 'correlation']).sort_values('lag')
    
    print("\nLag Effect Analysis (Sentiment → Future PnL):")
    print(lag_df.round(3))
    
    best_lag = lag_df.loc[lag_df['correlation'].abs().idxmax()]
    print(f"\n✓ Key Insights:")
    print(f"  - Strongest signal: Lag {int(best_lag['lag'])} days (corr: {best_lag['correlation']:.3f})")
    
    if best_lag['lag'] > 0:
        print(f"  - Interpretation: Sentiment {int(best_lag['lag'])} days ago predicts future PnL")
    elif best_lag['lag'] < 0:
        print(f"  - Interpretation: Past PnL {int(abs(best_lag['lag']))} days ago correlates with current sentiment")
    else:
        print(f"  - Interpretation: Current sentiment has immediate PnL correlation")
    
    return lag_df


if __name__ == "__main__":
    # Example usage (requires merged dataframe)
    print("Analysis module loaded. Import functions and call with merged dataframe.")
