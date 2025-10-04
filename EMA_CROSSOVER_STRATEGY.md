# EMA Crossover + Retest Strategy

## 🎯 Strategy Logic

### Entry Type 1: EMA Crossover
**Long Entry:**
- 8 EMA crosses above 30 EMA
- 8 EMA > 30 EMA
- Stop Loss: Most recent swing low
- Targets: 1:1, 1:2 R:R with trailing

**Short Entry:**
- 8 EMA crosses below 30 EMA  
- 8 EMA < 30 EMA
- Stop Loss: Most recent swing high
- Targets: 1:1, 1:2 R:R with trailing

### Entry Type 2: EMA Retest
**Short Entry (Price above both EMAs):**
1. Price closes below 8 EMA
2. Price closes below 30 EMA
3. Price retests 30 EMA (resistance)
4. Strong bearish candle forms
5. Stop Loss: Recent swing high

**Long Entry (Price below both EMAs):**
1. Price breaks below 8 EMA
2. Price retests 30 EMA (support)
3. Strong bullish candle forms
4. Stop Loss: Recent swing low

## 📊 Feature Categories

### Category 1: Core EMA Dynamic Features
- EMA Spread: (8_EMA - 30_EMA) / Close_Price
- EMA Spread Rate of Change
- 8 EMA Slope (3-5 bars)
- 30 EMA Slope

### Category 2: Price-to-EMA Relationship
- Price Distance from 8 EMA: (Close - 8_EMA) / Close
- Price Distance from 30 EMA: (Close - 30_EMA) / Close
- Price Position Flag (0=below both, 1=between, 2=above both)

### Category 3: Price Action & Volatility
- Candle Body Size: abs(Open - Close) / Close
- Candle Range: (High - Low) / Close
- Wick-to-Body Ratio
- Rolling Price Volatility (20 periods)

### Category 4: Volume & Conviction
- Volume Spike: Current_Volume / MA_Volume(20)
- Volume Rate of Change

### Category 5: Historical Context
- Lagged EMA Spread (1, 2, 3 candles ago)
- Lagged Price Distance from 30 EMA

## 🎯 Target Definition
- **Success**: Price moves favorably by 1:1 or 1:2 R:R before hitting stop
- **Failure**: Stop loss hit before target