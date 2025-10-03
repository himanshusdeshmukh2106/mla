"""Analyze backtest results to identify issues"""
import pandas as pd
import numpy as np

df = pd.read_csv('backtest_results.csv')
df['entry_time'] = pd.to_datetime(df['entry_time'])
df['hour'] = df['entry_time'].dt.hour
df['minute'] = df['entry_time'].dt.minute

print("="*70)
print("BACKTEST ANALYSIS - IDENTIFYING ISSUES")
print("="*70)

# Issue 1: Stop Loss Hit Rate
print("\n🔴 ISSUE 1: TOO MANY STOP LOSSES")
print(f"Stop losses: {(df['exit_reason']=='stop_loss').sum()} ({(df['exit_reason']=='stop_loss').mean()*100:.1f}%)")
print(f"These are hitting stop loss VERY quickly:")
sl_trades = df[df['exit_reason']=='stop_loss']
print(f"  Avg candles before SL: {sl_trades['candles_held'].mean():.1f}")
print(f"  Median candles before SL: {sl_trades['candles_held'].median():.0f}")

# Issue 2: Quick Reversals
print("\n🔴 ISSUE 2: QUICK REVERSALS (<=3 candles)")
quick_exits = df[df['candles_held'] <= 3]
print(f"Quick exits: {len(quick_exits)} ({len(quick_exits)/len(df)*100:.1f}%)")
print(f"Win rate: {quick_exits['win'].mean()*100:.1f}%")
print(f"Exit reasons:")
print(quick_exits['exit_reason'].value_counts())

# Issue 3: Timing Analysis
print("\n🔴 ISSUE 3: ENTRY TIMING")
print("Win rate by hour:")
hourly = df.groupby('hour').agg({
    'win': ['sum', 'count', 'mean']
}).round(3)
hourly.columns = ['wins', 'total', 'win_rate']
hourly['win_rate'] = hourly['win_rate'] * 100
print(hourly)

# Issue 4: First candle entries
print("\n🔴 ISSUE 4: 9:15 AM ENTRIES (Market Open)")
first_candle = df[(df['hour']==9) & (df['minute']==15)]
print(f"Trades at 9:15 AM: {len(first_candle)} ({len(first_candle)/len(df)*100:.1f}%)")
print(f"Win rate: {first_candle['win'].mean()*100:.1f}%")
print(f"Avg P&L: {first_candle['pnl_pct'].mean():.3f}%")

# Issue 5: Risk/Reward Imbalance
print("\n🔴 ISSUE 5: RISK/REWARD RATIO")
print(f"Current setup:")
print(f"  Profit Target: 0.5%")
print(f"  Stop Loss: 0.3%")
print(f"  Risk/Reward: 1.67:1")
print(f"\nWith 39% win rate, need R:R of at least 1.56:1 to break even")
print(f"Current R:R is good, but win rate is too low!")

# Issue 6: Max Holding Exits
print("\n🔴 ISSUE 6: MAX HOLDING PERIOD EXITS")
max_hold = df[df['exit_reason']=='max_holding']
print(f"Max holding exits: {len(max_hold)} ({len(max_hold)/len(df)*100:.1f}%)")
print(f"Win rate: {max_hold['win'].mean()*100:.1f}%")
print(f"Avg P&L: {max_hold['pnl_pct'].mean():.3f}%")
print(f"These trades are going nowhere - model is wrong")

# Summary
print("\n" + "="*70)
print("🎯 ROOT CAUSES:")
print("="*70)
print("1. Model generates too many FALSE signals (61% failure rate)")
print("2. Signals reverse IMMEDIATELY (median 7 candles to stop loss)")
print("3. 9:15 AM entries are terrible (market open volatility)")
print("4. Need BETTER signal filtering, not better risk management")
print("\n" + "="*70)
print("💡 SOLUTIONS:")
print("="*70)
print("1. ✅ Increase confidence threshold (0.55 -> 0.70+)")
print("2. ✅ Skip first 15 minutes (avoid 9:15-9:30 entries)")
print("3. ✅ Add confirmation candle (wait 1 candle after signal)")
print("4. ✅ Widen stop loss slightly (0.3% -> 0.4%)")
print("5. ✅ Add trend filter (only trade with ADX > 25)")
print("="*70)
