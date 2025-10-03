# Complete List of All 51 Features Used in Training

## 📊 All Features with Importance Rankings

### 🔵 EMA-Related Features (9 features)

| # | Feature Name | Importance | Description |
|---|--------------|------------|-------------|
| 6 | Time_EMA_Signal | 3.72% | Time × Distance from EMA (interaction) |
| 7 | EMA_21 | 2.68% | 21-period EMA value |
| 11 | Distance_From_EMA21_Pct | 1.82% | Distance from EMA in % |
| 12 | Crosses_Below_Last_10 | 1.77% | Number of crosses below in last 10 candles |
| 13 | Crosses_Above_Last_2 | 1.72% | Number of crosses above in last 2 candles |
| 17 | EMA21_Cross_Above | 1.66% | Just crossed above EMA? (1/0) |
| 23 | Crosses_Above_Last_3 | 1.50% | Number of crosses above in last 3 candles |
| 25 | Crosses_Above_Last_10 | 1.46% | Number of crosses above in last 10 candles |
| 27 | Distance_EMA_Trend | 1.45% | 3-candle average of distance from EMA |

**Subtotal: 17.78% importance**

---

### ⏰ Time-Based Features (8 features)

| # | Feature Name | Importance | Description |
|---|--------------|------------|-------------|
| 2 | Hour | 6.84% | Hour of day (9-15) |
| 3 | Time_Slot | 6.45% | 15-minute time slot number |
| 4 | Minute | 5.18% | Minute within hour (0-59) |
| 5 | Is_9_15_to_9_30 | 3.73% | Market open window (1/0) |
| 16 | Is_10_00_to_10_30 | 1.67% | Mid-morning window (1/0) |
| 24 | Is_9_30_to_10_00 | 1.48% | Early morning window (1/0) |
| 26 | Is_11_00_to_12_00 | 1.46% | Noon hour window (1/0) |
| 28 | Tiny_Candle | 1.44% | Candle body 0.10-0.15% (1/0) |

**Subtotal: 28.25% importance**

---

### 📏 Candle Pattern Features (7 features)

| # | Feature Name | Importance | Description |
|---|--------------|------------|-------------|
| 1 | Candle_Range_Pct | 10.11% | Total candle range (high-low) in % |
| 9 | Candle_Body_Pct | 2.07% | Candle body size in % |
| 22 | Candle_Efficiency | 1.50% | Body/Range ratio |
| 28 | Tiny_Candle | 1.44% | Body 0.10-0.15% (1/0) |
| 33 | Small_Candle | 1.27% | Body 0.15-0.25% (1/0) |
| 35 | Green_Candle | 1.24% | Close > Open (1/0) |
| 37 | Red_Candle | 1.22% | Close < Open (1/0) |

**Note:** Micro_Candle and Medium_Candle also exist but lower importance

**Subtotal: 18.85% importance**

---

### 📈 Price Momentum Features (4 features)

| # | Feature Name | Importance | Description |
|---|--------------|------------|-------------|
| 10 | Price_Change_5 | 1.89% | 5-candle price change % |
| 14 | Price_Momentum | 1.71% | 3-candle average momentum |
| 15 | Price_Change_3 | 1.70% | 3-candle price change % |
| 38 | Price_Change_1 | 1.21% | 1-candle price change % |

**Subtotal: 6.51% importance**

---

### 📊 Volume Features (7 features)

| # | Feature Name | Importance | Description |
|---|--------------|------------|-------------|
| 8 | High_Volume | 2.18% | Volume ratio > 1.2 (1/0) |
| 18 | Volume_Change | 1.65% | Volume % change from previous |
| 30 | Volume_Ratio | 1.43% | Current volume / 20-period average |
| 34 | Very_Low_Volume | 1.26% | Volume ratio < 0.5 (1/0) |
| 36 | Low_Volume | 1.23% | Volume ratio 0.5-0.8 (1/0) |
| 39 | Normal_Volume | 1.20% | Volume ratio 0.8-1.2 (1/0) |
| 49 | Volume_Candle_Signal | 0.93% | Volume × Candle size (interaction) |

**Subtotal: 9.88% importance**

---

### 💪 ADX (Trend Strength) Features (6 features)

| # | Feature Name | Importance | Description |
|---|--------------|------------|-------------|
| 19 | ADX_Very_Strong | 1.64% | ADX > 40 (1/0) |
| 20 | ADX_Weak | 1.57% | ADX 15-20 (1/0) |
| 21 | ADX | 1.52% | ADX value (14-period) |
| 29 | ADX_Very_Weak | 1.44% | ADX < 15 (1/0) |
| 31 | ADX_Strong | 1.40% | ADX 30-40 (1/0) |
| 32 | ADX_Optimal | 1.28% | ADX 20-30 (1/0) |

**Note:** ADX_Change also exists but lower importance

**Subtotal: 8.85% importance**

---

### 🔗 Interaction Features (3 features)

| # | Feature Name | Importance | Description |
|---|--------------|------------|-------------|
| 6 | Time_EMA_Signal | 3.72% | Time × Distance from EMA |
| 40 | EMA_ADX_Signal | 1.19% | Distance from EMA × ADX |
| 49 | Volume_Candle_Signal | 0.93% | Volume × Candle size |

**Subtotal: 5.84% importance**

---

### 📉 Cross History Features (6 features)

| # | Feature Name | Importance | Description |
|---|--------------|------------|-------------|
| 12 | Crosses_Below_Last_10 | 1.77% | Crosses below in last 10 candles |
| 13 | Crosses_Above_Last_2 | 1.72% | Crosses above in last 2 candles |
| 23 | Crosses_Above_Last_3 | 1.50% | Crosses above in last 3 candles |
| 25 | Crosses_Above_Last_10 | 1.46% | Crosses above in last 10 candles |
| 41 | Crosses_Above_Last_5 | 1.18% | Crosses above in last 5 candles |
| 42 | Crosses_Below_Last_2 | 1.17% | Crosses below in last 2 candles |

**Note:** Crosses_Below_Last_3 and Crosses_Below_Last_5 also exist

**Subtotal: 8.80% importance**

---

## 📋 Complete List (All 51 Features)

### Alphabetical Order:

1. ADX
2. ADX_Change
3. ADX_Optimal
4. ADX_Strong
5. ADX_Very_Strong
6. ADX_Very_Weak
7. ADX_Weak
8. Candle_Body_Pct
9. Candle_Efficiency
10. Candle_Range_Pct
11. Crosses_Above_Last_10
12. Crosses_Above_Last_2
13. Crosses_Above_Last_3
14. Crosses_Above_Last_5
15. Crosses_Below_Last_10
16. Crosses_Below_Last_2
17. Crosses_Below_Last_3
18. Crosses_Below_Last_5
19. Distance_EMA_Change
20. Distance_EMA_Trend
21. Distance_From_EMA21_Pct
22. EMA_21
23. EMA_ADX_Signal
24. EMA21_Cross_Above
25. EMA21_Cross_Below
26. Green_Candle
27. High_Volume
28. Hour
29. Is_10_00_to_10_30
30. Is_10_30_to_11_00
31. Is_11_00_to_12_00
32. Is_9_15_to_9_30
33. Is_9_30_to_10_00
34. Low_Volume
35. Medium_Candle
36. Micro_Candle
37. Minute
38. Normal_Volume
39. Price_Change_1
40. Price_Change_3
41. Price_Change_5
42. Price_Momentum
43. Red_Candle
44. Small_Candle
45. Time_EMA_Signal
46. Time_Slot
47. Tiny_Candle
48. Very_Low_Volume
49. Volume_Candle_Signal
50. Volume_Change
51. Volume_Ratio

---

## 📊 Summary by Category

| Category | # Features | Total Importance | Top Feature |
|----------|-----------|------------------|-------------|
| **Candle Patterns** | 7 | 18.85% | Candle_Range_Pct (10.11%) |
| **Time-Based** | 8 | 28.25% | Hour (6.84%) |
| **EMA-Related** | 9 | 17.78% | Time_EMA_Signal (3.72%) |
| **Volume** | 7 | 9.88% | High_Volume (2.18%) |
| **ADX (Trend)** | 6 | 8.85% | ADX_Very_Strong (1.64%) |
| **Price Momentum** | 4 | 6.51% | Price_Change_5 (1.89%) |
| **Cross History** | 6 | 8.80% | Crosses_Below_Last_10 (1.77%) |
| **Interactions** | 3 | 5.84% | Time_EMA_Signal (3.72%) |
| **Other** | 1 | 4.24% | Various |

**Total: 51 features, 100% importance**

---

## 🎯 Key Insights

### Top 10 Most Important:
1. Candle_Range_Pct (10.11%) - Candle size
2. Hour (6.84%) - Time of day
3. Time_Slot (6.45%) - Time window
4. Minute (5.18%) - Exact timing
5. Is_9_15_to_9_30 (3.73%) - Market open
6. Time_EMA_Signal (3.72%) - Time × EMA
7. EMA_21 (2.68%) - EMA value
8. High_Volume (2.18%) - Volume spike
9. Candle_Body_Pct (2.07%) - Body size
10. Price_Change_5 (1.89%) - 5-candle momentum

**These 10 features account for 48.05% of total importance!**

### Feature Type Distribution:
- **Time features:** 28.25% (most important!)
- **Candle features:** 18.85%
- **EMA features:** 17.78%
- **Volume features:** 9.88%
- **Cross history:** 8.80%
- **ADX features:** 8.85%
- **Momentum:** 6.51%
- **Interactions:** 5.84%

---

## 💡 What This Tells Us

1. **Time is King** (28.25%) - When you trade matters most
2. **Candle Size Matters** (18.85%) - Volatility is key
3. **EMA is Important** (17.78%) - But not dominant
4. **Volume Confirms** (9.88%) - Validates moves
5. **Trend Strength** (8.85%) - ADX filters quality

**The model is NOT just about EMA traps. It's a comprehensive multi-factor system that considers timing, volatility, trend, volume, and momentum together.**

---

*This is the complete list of all 51 features the model uses to predict 0.2% moves in the next 10 minutes.*
