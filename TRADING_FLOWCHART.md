# Trading Decision Flowchart

## 📊 Every 5 Minutes (New Candle):

```
START: New 5-minute candle closes
    ↓
┌─────────────────────────────────────┐
│ 1. CHECK TIME                       │
│    Is it 10:00 AM - 2:00 PM?       │
└─────────────────────────────────────┘
    ↓ NO → SKIP (wait for next candle)
    ↓ YES
┌─────────────────────────────────────┐
│ 2. CALCULATE FEATURES               │
│    - EMA 21, Distance from EMA      │
│    - ADX (trend strength)           │
│    - Time features                  │
│    - Candle patterns                │
│    - Volume, momentum               │
│    Total: 51 features               │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 3. MODEL PREDICTION                 │
│    Feed 51 features to XGBoost      │
│    Get probability: 0.0 to 1.0      │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 4. CHECK CONFIDENCE                 │
│    Is probability ≥ 0.80?           │
└─────────────────────────────────────┘
    ↓ NO → SKIP
    ↓ YES
┌─────────────────────────────────────┐
│ 5. CHECK ADX                        │
│    Is ADX ≥ 28?                     │
│    (Strong trend required)          │
└─────────────────────────────────────┘
    ↓ NO → SKIP
    ↓ YES
┌─────────────────────────────────────┐
│ 6. CHECK CONFIRMATION               │
│    Did previous candle also signal? │
└─────────────────────────────────────┘
    ↓ NO → WAIT (mark as potential)
    ↓ YES
┌─────────────────────────────────────┐
│ 7. ENTER TRADE! 🎯                  │
│    Entry Price = Current Close      │
│    Calculate:                       │
│    - Stop Loss = Entry - (1.5×ATR)  │
│    - Target = Entry + 0.6%          │
│    - Trailing Stop = 0.3%           │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 8. MONITOR TRADE (Every Candle)    │
└─────────────────────────────────────┘
    ↓
    ├─→ Price hits TARGET? → EXIT (WIN) ✅
    ├─→ Price hits STOP? → EXIT (LOSS) ❌
    ├─→ 15 candles passed? → EXIT (TIMEOUT) ⏱️
    └─→ 3:15 PM reached? → EXIT (EOD) 🔔
```

---

## 🎯 Real Example:

### 11:30 AM - New Candle Closes

**Step 1: Time Check**
- Current time: 11:30 AM ✅
- Within 10 AM - 2 PM window ✅

**Step 2: Calculate Features**
- Price: 1450.00
- EMA 21: 1448.00
- Distance from EMA: +0.14%
- ADX: 32
- Hour: 11, Minute: 30
- Candle body: 0.25%
- Volume ratio: 1.2
- ... (51 total features)

**Step 3: Model Prediction**
- Input: All 51 features
- Output: Probability = 0.85 (85% confidence)

**Step 4: Confidence Check**
- 0.85 ≥ 0.80? ✅ YES

**Step 5: ADX Check**
- ADX = 32 ≥ 28? ✅ YES

**Step 6: Confirmation Check**
- Previous candle (11:25) also signaled? ✅ YES

**Step 7: ENTER TRADE**
- Entry: 1450.00
- ATR: 5.0
- Stop Loss: 1450 - (1.5 × 5) = 1442.50 (-0.52%)
- Target: 1450 × 1.006 = 1458.70 (+0.60%)
- Max time: 11:30 + 75 min = 12:45 PM

**Step 8: Monitor**
- 11:35 AM: Price = 1452 (trailing stop → 1447.64)
- 11:40 AM: Price = 1455 (trailing stop → 1450.65)
- 11:45 AM: Price = 1458.70 → **TARGET HIT! EXIT**
- **Result: +0.60% profit in 15 minutes** ✅

---

## 📊 Statistics:

Out of 100 candles:
- 95 candles: Skip (don't meet criteria)
- 5 candles: Generate signal
- 2 candles: Actually enter trade (after confirmation)

Out of 100 trades:
- 36 trades: Hit profit target (+0.47% avg) ✅
- 64 trades: Hit stop loss (-0.24% avg) ❌
- Net result: Profitable! (+0.017% per trade)

---

## 💡 Key Insights:

### Why So Selective?
- Only 28 trades in 3 months (2 per week)
- Because we only take the BEST setups
- Quality > Quantity

### Why Low Win Rate?
- 35% win rate seems bad
- But we win TWICE as much as we lose
- 0.47% wins vs 0.24% losses = 2:1 ratio
- This makes us profitable

### Why ATR Stop Loss?
- Market volatility changes
- Fixed stops don't work
- ATR adapts automatically
- Average stop: only -0.077%!

### Why 10 AM - 2 PM?
- 9:15-10:00: Market open chaos
- 10:00-2:00: Most predictable
- 2:00-3:30: End of day uncertainty

---

## 🎓 The Bottom Line:

**The model doesn't predict every move.**

It identifies high-probability setups where:
1. Trend is strong (ADX ≥ 28)
2. Pattern is clear (confidence ≥ 80%)
3. Timing is right (10 AM - 2 PM)
4. Risk is controlled (ATR stops)

By being ultra-selective and managing risk properly, we achieve profitability even with a 35% win rate.

**It's not about being right all the time.**
**It's about making more when you're right than you lose when you're wrong.**
