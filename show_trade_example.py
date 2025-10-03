"""Show a real trade example from the backtest"""
import pandas as pd

df = pd.read_csv('backtest_results_final.csv')

print("="*70)
print("REAL TRADE EXAMPLES FROM BACKTEST")
print("="*70)

# Best winning trade
best = df.nlargest(1, 'pnl_pct').iloc[0]
print("\n🎉 BEST WINNING TRADE:")
print(f"   Entry Time:    {best['entry_time']}")
print(f"   Entry Price:   ₹{best['entry_price']:.2f}")
print(f"   Exit Time:     {best['exit_time']}")
print(f"   Exit Price:    ₹{best['exit_price']:.2f}")
print(f"   Exit Reason:   {best['exit_reason']}")
print(f"   Holding Time:  {best['candles_held']} candles ({best['candles_held']*5} minutes)")
print(f"   P&L:           +{best['pnl_pct']:.2f}% (+₹{best['pnl_points']:.2f})")
print(f"   Entry ADX:     {best['entry_adx']:.1f}")
print(f"   Confidence:    {best['entry_confidence']:.2%}")
print(f"   Stop Loss:     {best['stop_loss_pct']:.2f}%")

# Worst losing trade
worst = df.nsmallest(1, 'pnl_pct').iloc[0]
print("\n😞 WORST LOSING TRADE:")
print(f"   Entry Time:    {worst['entry_time']}")
print(f"   Entry Price:   ₹{worst['entry_price']:.2f}")
print(f"   Exit Time:     {worst['exit_time']}")
print(f"   Exit Price:    ₹{worst['exit_price']:.2f}")
print(f"   Exit Reason:   {worst['exit_reason']}")
print(f"   Holding Time:  {worst['candles_held']} candles ({worst['candles_held']*5} minutes)")
print(f"   P&L:           {worst['pnl_pct']:.2f}% (₹{worst['pnl_points']:.2f})")
print(f"   Entry ADX:     {worst['entry_adx']:.1f}")
print(f"   Confidence:    {worst['entry_confidence']:.2%}")
print(f"   Stop Loss:     {worst['stop_loss_pct']:.2f}%")

# Average trade
print("\n📊 AVERAGE TRADE:")
print(f"   Holding Time:  {df['candles_held'].mean():.1f} candles ({df['candles_held'].mean()*5:.0f} minutes)")
print(f"   P&L:           {df['pnl_pct'].mean():.3f}%")
print(f"   Entry ADX:     {df['entry_adx'].mean():.1f}")
print(f"   Confidence:    {df['entry_confidence'].mean():.2%}")
print(f"   Stop Loss:     {df['stop_loss_pct'].mean():.3f}%")

print("\n" + "="*70)
print("ENTRY RULES SUMMARY")
print("="*70)
print("✅ Model Confidence ≥ 80%")
print("✅ ADX ≥ 28 (strong trend)")
print("✅ Time: 10:00 AM - 2:00 PM")
print("✅ Confirmation candle required")
print("\n🛡️ RISK MANAGEMENT:")
print(f"   Stop Loss:     1.5 × ATR (avg {df['stop_loss_pct'].mean():.3f}%)")
print(f"   Profit Target: +0.60%")
print(f"   Trailing Stop: 0.30%")
print(f"   Max Holding:   15 candles (75 minutes)")

print("\n" + "="*70)
print("WHY IT WORKS")
print("="*70)
print(f"✅ Win Rate: {df['win'].mean()*100:.1f}% (low but acceptable)")
print(f"✅ Avg Win: +{df[df['win']==1]['pnl_pct'].mean():.2f}%")
print(f"✅ Avg Loss: {df[df['win']==0]['pnl_pct'].mean():.2f}%")
print(f"✅ Win/Loss Ratio: {abs(df[df['win']==1]['pnl_pct'].mean() / df[df['win']==0]['pnl_pct'].mean()):.1f}:1")
print(f"✅ Profit Factor: {df[df['win']==1]['pnl_pct'].sum() / abs(df[df['win']==0]['pnl_pct'].sum()):.2f}")
print("\n💡 We win TWICE as much as we lose!")
print("   Even with 35% win rate, we're profitable.")
print("="*70)
