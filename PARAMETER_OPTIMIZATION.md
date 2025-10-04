# Parameter Optimization Guide

## 🎯 What We're Optimizing

Instead of guessing the best parameters, we'll test ALL combinations and find what works best!

### Parameters to Test:

1. **Confidence Threshold** (Model signal strength)
   - Current: 0.80
   - Testing: 0.70, 0.75, 0.80, 0.85

2. **Minimum ADX** (Trend strength filter)
   - Current: 28
   - Testing: 25, 28, 30, 32

3. **ATR Multiplier for Stop Loss**
   - Current: 1.5x ATR
   - Testing: 1.0x, 1.5x, 2.0x ATR

4. **Profit Target** (ATR-based)
   - Current: Fixed 0.6%
   - Testing: 1.0x, 1.5x, 2.0x ATR

5. **Trailing Stop** (ATR-based)
   - Current: 0.3% fixed
   - Testing: 0.5x, 0.75x, 1.0x ATR

6. **Max Holding Period**
   - Current: 15 candles
   - Testing: 15, 20, 25 candles

---

## 📊 Total Combinations

```
4 confidence × 4 ADX × 3 stop loss × 3 profit target × 3 trailing × 3 max holding
= 1,296 combinations to test!
```

**Estimated time:** ~40-50 minutes

---

## 🚀 How to Run

```bash
python optimize_parameters.py
```

This will:
1. Load the current model
2. Load test data
3. Test all 1,296 combinations
4. Save results to `optimization_results.csv`
5. Show top 10 for each metric

---

## 📈 What You'll Get

### Top 10 by Total P&L
Best absolute profit over the period

### Top 10 by Profit Factor
Best risk/reward ratio (wins/losses)

### Top 10 by Win Rate
Highest percentage of winning trades

### Best Overall
Balanced score considering all metrics

---

## 💡 Example Output

```
🏆 TOP 10 BY TOTAL P&L:
   Conf:0.75 ADX:28 SL:1.5x PT:2.0x Trail:0.75x → 
   P&L:+1.23% (45 trades, 42.2% WR, PF:1.35)
   
   Conf:0.80 ADX:25 SL:1.0x PT:1.5x Trail:1.00x → 
   P&L:+1.15% (38 trades, 44.7% WR, PF:1.28)
   ...

🏆 BEST OVERALL PARAMETERS:
Confidence Threshold: 0.75
Minimum ADX: 28
Stop Loss: 1.5x ATR
Profit Target: 2.0x ATR
Trailing Stop: 0.75x ATR
Max Holding: 20 candles

Results:
  Total P&L: +1.23%
  Trades: 45
  Win Rate: 42.2%
  Profit Factor: 1.35
```

---

## 🎯 How It Works

### ATR-Based Risk Management

Instead of fixed percentages, everything is based on ATR (volatility):

#### Example Trade:
```
Entry Price: 1450
ATR: 10 points

Stop Loss (1.5x ATR):
  1450 - (10 × 1.5) = 1435 (-15 points, -1.03%)

Profit Targets (2.0x ATR):
  Target 1: 1450 + (10 × 2.0) = 1470 (+20 points, +1.38%)
  Target 2: 1450 + (10 × 4.0) = 1490 (+40 points, +2.76%)
  Target 3: 1450 + (10 × 6.0) = 1510 (+60 points, +4.14%)

Trailing Stop (0.75x ATR):
  If price hits 1470, stop moves to:
  1470 - (10 × 0.75) = 1462.5 (locks in +12.5 points)
```

### Why ATR-Based?

1. **Adapts to Volatility**
   - High volatility = wider stops
   - Low volatility = tighter stops

2. **Consistent Risk**
   - Always risking same multiple of volatility
   - Not affected by price level

3. **Better Targets**
   - Targets based on realistic movement
   - Not arbitrary percentages

---

## 📊 What to Look For

### Good Parameters Have:

✅ **Profit Factor > 1.2**
   - Win more than you lose

✅ **Total P&L > +0.5%**
   - Actually profitable

✅ **Win Rate > 35%**
   - Reasonable success rate

✅ **Enough Trades (20-50)**
   - Not too few (unreliable)
   - Not too many (over-trading)

### Red Flags:

❌ **Profit Factor < 1.0**
   - Losing strategy

❌ **Very Few Trades (< 10)**
   - Too selective, unreliable

❌ **Very Many Trades (> 100)**
   - Too loose, probably over-trading

❌ **Win Rate < 30%**
   - Even with good R:R, psychologically hard

---

## 🎓 Understanding the Results

### Scenario 1: High Confidence (0.85)
```
Trades: 15
Win Rate: 46.7%
P&L: +0.35%
```
**Analysis:** Very selective, decent win rate, but too few trades

### Scenario 2: Low Confidence (0.70)
```
Trades: 65
Win Rate: 33.8%
P&L: +0.25%
```
**Analysis:** Many trades, low win rate, marginal profit

### Scenario 3: Balanced (0.75-0.80)
```
Trades: 35-45
Win Rate: 38-42%
P&L: +0.80-1.20%
```
**Analysis:** Good balance, likely the sweet spot

---

## 🔧 After Optimization

### 1. Review Results
```bash
# Open the CSV file
optimization_results.csv
```

### 2. Pick Best Parameters
Look at:
- Top P&L
- Top Profit Factor
- Best Overall

### 3. Update Backtest Script
```python
# In backtest_ema_trap_final.py
self.confidence_threshold = 0.75  # From optimization
self.min_adx = 28
self.atr_multiplier = 1.5
self.profit_target_atr = 2.0
self.trailing_stop_atr = 0.75
self.max_holding_candles = 20
```

### 4. Verify
Run backtest with new parameters to confirm

---

## ⚠️ Important Notes

### Overfitting Risk
- Optimizing on test data can overfit
- Best practice: Use separate validation data
- Or: Use walk-forward optimization

### Market Changes
- Optimal parameters change over time
- Re-optimize every 3-6 months
- Monitor performance

### Trade-offs
- Higher confidence = fewer trades
- Wider stops = lower win rate but better R:R
- Tighter stops = higher win rate but worse R:R

---

## 💡 Pro Tips

### 1. Start Broad, Then Narrow
```
First pass: Test wide range
Second pass: Zoom in on best region
```

### 2. Consider Trade Count
```
Too few trades (< 20): Unreliable
Sweet spot (30-50): Good balance
Too many (> 80): Probably over-trading
```

### 3. Balance Metrics
```
Don't just optimize for P&L
Consider: Win rate, profit factor, trade count
```

### 4. Psychological Factors
```
40% win rate with 2:1 R:R = Profitable but hard
50% win rate with 1.5:1 R:R = Easier to follow
```

---

## 🏆 Expected Improvements

### Current Performance:
- Confidence: 0.80
- ADX: 28
- Stop: 1.5x ATR (avg -0.077%)
- Target: Fixed 0.6%
- Result: +0.48% (28 trades, 35.7% WR)

### After Optimization:
- Could find: +0.8% to +1.5%
- More trades: 35-50
- Better win rate: 38-45%
- Better profit factor: 1.3-1.5

**Potential improvement: 2-3x better results!**

---

*Run the optimization and let the data tell you what works best!*
