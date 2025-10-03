"""Compare original vs improved backtest results"""
import pandas as pd

print("="*70)
print("BACKTEST COMPARISON: ORIGINAL vs IMPROVED")
print("="*70)

# Load both results
original = pd.read_csv('backtest_results.csv')
improved = pd.read_csv('backtest_results_improved.csv')

print("\n📊 TRADE COUNT:")
print(f"Original:  {len(original)} trades")
print(f"Improved:  {len(improved)} trades")
print(f"Reduction: {len(original) - len(improved)} trades ({(1-len(improved)/len(original))*100:.1f}% fewer)")

print("\n🎯 WIN RATE:")
print(f"Original:  {original['win'].mean()*100:.2f}%")
print(f"Improved:  {improved['win'].mean()*100:.2f}%")
print(f"Change:    {(improved['win'].mean() - original['win'].mean())*100:+.2f}%")

print("\n💰 TOTAL P&L:")
print(f"Original:  {original['pnl_pct'].sum():.2f}%")
print(f"Improved:  {improved['pnl_pct'].sum():.2f}%")
print(f"Change:    {improved['pnl_pct'].sum() - original['pnl_pct'].sum():+.2f}%")

print("\n📈 AVERAGE P&L PER TRADE:")
print(f"Original:  {original['pnl_pct'].mean():.3f}%")
print(f"Improved:  {improved['pnl_pct'].mean():.3f}%")
print(f"Change:    {improved['pnl_pct'].mean() - original['pnl_pct'].mean():+.3f}%")

print("\n💵 PROFIT FACTOR:")
orig_wins = original[original['win']==1]['pnl_pct'].sum()
orig_loss = abs(original[original['win']==0]['pnl_pct'].sum())
orig_pf = orig_wins / orig_loss if orig_loss > 0 else 0

imp_wins = improved[improved['win']==1]['pnl_pct'].sum()
imp_loss = abs(improved[improved['win']==0]['pnl_pct'].sum())
imp_pf = imp_wins / imp_loss if imp_loss > 0 else 0

print(f"Original:  {orig_pf:.2f}")
print(f"Improved:  {imp_pf:.2f}")
print(f"Change:    {imp_pf - orig_pf:+.2f}")

print("\n🚪 EXIT REASONS:")
print("\nOriginal:")
print(original['exit_reason'].value_counts())
print("\nImproved:")
print(improved['exit_reason'].value_counts())

print("\n⏱️  HOLDING PERIOD:")
print(f"Original:  {original['candles_held'].mean():.1f} candles ({original['candles_held'].mean()*5:.0f} min)")
print(f"Improved:  {improved['candles_held'].mean():.1f} candles ({improved['candles_held'].mean()*5:.0f} min)")

print("\n" + "="*70)
print("🎯 IMPROVEMENTS MADE:")
print("="*70)
print("✅ Reduced trades by 76% (400 -> 95) - Better quality signals")
print("✅ Win rate improved from 39.25% to 43.16% (+3.91%)")
print("✅ Stop loss hits reduced from 56.5% to 43.2%")
print("✅ ATR-based stops adapt to volatility")
print("✅ Filtered out 9:15 AM entries (market open chaos)")
print("✅ Only trading with ADX > 25 (trending markets)")
print("✅ Higher confidence threshold (0.70 vs 0.55)")

print("\n" + "="*70)
print("⚠️  STILL NEEDS WORK:")
print("="*70)
print("❌ Still losing money (-2.21% total P&L)")
print("❌ Profit factor still < 1.0 (0.88)")
print("❌ Win rate still too low (need 45%+ minimum)")
print("❌ 25% of trades hitting max holding (going nowhere)")

print("\n" + "="*70)
print("💡 NEXT STEPS:")
print("="*70)
print("1. Increase confidence threshold to 0.80+ (take only best signals)")
print("2. Add trailing stop loss (lock in profits)")
print("3. Adjust profit target to 0.6-0.7% (easier to hit)")
print("4. Filter out max_holding trades (add momentum check)")
print("5. Consider only trading specific hours (11 AM - 2 PM best)")
print("="*70)
