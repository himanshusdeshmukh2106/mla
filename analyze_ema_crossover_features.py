"""
Analyze EMA Crossover Features
Shows which features are most important and which can be removed
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def analyze_features(data_path='ema_crossover_with_targets.csv'):
    """Analyze feature importance and correlations"""
    
    print("="*80)
    print("📊 EMA CROSSOVER FEATURE ANALYSIS")
    print("="*80)
    
    # Load data
    print("\n1️⃣ Loading data...")
    df = pd.read_csv(data_path)
    
    # Prepare features
    target_cols = [col for col in df.columns if 'target_' in col]
    exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'datetime', 'timestamp']
    exclude_cols.extend(target_cols)
    
    feature_names = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_names].fillna(0).values
    y = df['target_combined_1.0R'].values if 'target_combined_1.0R' in df.columns else df[target_cols[0]].values
    
    print(f"   Total features: {len(feature_names)}")
    print(f"   Total samples: {len(X):,}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train quick model for feature importance
    print("\n2️⃣ Training XGBoost for feature importance...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Get feature importance
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # Categorize features
    print("\n3️⃣ Categorizing features...")
    
    categories = {
        'Core EMA': ['EMA_8', 'EMA_30', 'EMA_Spread', 'EMA_8_Above_30'],
        'EMA Dynamics': [f for f in feature_names if 'Slope' in f or 'ROC' in f and 'EMA' in f],
        'Crossovers': [f for f in feature_names if 'Cross' in f],
        'Price-EMA Distance': [f for f in feature_names if 'Distance' in f or 'Position' in f],
        'Price Action': [f for f in feature_names if 'Candle' in f or 'Wick' in f or 'Green' in f or 'Red' in f],
        'Volume': [f for f in feature_names if 'Volume' in f],
        'Volatility': [f for f in feature_names if 'Volatility' in f or 'Returns' in f],
        'Swing Levels': [f for f in feature_names if 'Swing' in f],
        'Lag Features': [f for f in feature_names if 'Lag' in f],
        'Time': [f for f in feature_names if 'Hour' in f or 'Minute' in f or 'Time' in f or 'Session' in f],
        'Composite Signals': [f for f in feature_names if 'Signal' in f]
    }
    
    # Calculate importance by category
    category_importance = {}
    for cat, feats in categories.items():
        cat_feats = [f for f in feats if f in feature_names]
        if cat_feats:
            cat_importance = feature_importance[feature_importance['feature'].isin(cat_feats)]['importance'].sum()
            category_importance[cat] = {
                'total_importance': cat_importance,
                'num_features': len(cat_feats),
                'avg_importance': cat_importance / len(cat_feats) if len(cat_feats) > 0 else 0
            }
    
    # Print results
    print("\n" + "="*80)
    print("📈 FEATURE IMPORTANCE BY CATEGORY")
    print("="*80)
    
    sorted_categories = sorted(category_importance.items(), key=lambda x: x[1]['total_importance'], reverse=True)
    
    for cat, stats in sorted_categories:
        print(f"\n{cat}:")
        print(f"   Features: {stats['num_features']}")
        print(f"   Total Importance: {stats['total_importance']:.4f}")
        print(f"   Avg Importance: {stats['avg_importance']:.4f}")
    
    # Top features
    print("\n" + "="*80)
    print("🏆 TOP 30 MOST IMPORTANT FEATURES")
    print("="*80)
    
    for i, row in feature_importance.head(30).iterrows():
        print(f"{i+1:2d}. {row['feature']:40s} {row['importance']:.4f}")
    
    # Bottom features (candidates for removal)
    print("\n" + "="*80)
    print("❌ BOTTOM 20 LEAST IMPORTANT FEATURES (Consider removing)")
    print("="*80)
    
    for i, row in feature_importance.tail(20).iterrows():
        print(f"{len(feature_importance)-i:2d}. {row['feature']:40s} {row['importance']:.4f}")
    
    # Correlation analysis
    print("\n4️⃣ Analyzing feature correlations...")
    
    df_features = pd.DataFrame(X, columns=feature_names)
    corr_matrix = df_features.corr().abs()
    
    # Find highly correlated features
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > 0.9:  # Correlation > 0.9
                high_corr_pairs.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': corr_matrix.iloc[i, j]
                })
    
    if high_corr_pairs:
        print(f"\n⚠️  Found {len(high_corr_pairs)} highly correlated feature pairs (>0.9):")
        for pair in high_corr_pairs[:10]:  # Show top 10
            print(f"   {pair['feature1']:30s} <-> {pair['feature2']:30s} ({pair['correlation']:.3f})")
        if len(high_corr_pairs) > 10:
            print(f"   ... and {len(high_corr_pairs) - 10} more")
    
    # Recommendations
    print("\n" + "="*80)
    print("💡 RECOMMENDATIONS")
    print("="*80)
    
    # Calculate cumulative importance
    feature_importance['cumulative_importance'] = feature_importance['importance'].cumsum() / feature_importance['importance'].sum()
    
    # Find how many features for 90% importance
    features_90 = len(feature_importance[feature_importance['cumulative_importance'] <= 0.90])
    features_95 = len(feature_importance[feature_importance['cumulative_importance'] <= 0.95])
    
    print(f"\n📊 Cumulative Importance:")
    print(f"   Top {features_90} features = 90% of total importance")
    print(f"   Top {features_95} features = 95% of total importance")
    print(f"   Current: {len(feature_names)} features")
    
    print(f"\n✅ Suggested Actions:")
    print(f"   1. Keep top {features_90} features (90% importance)")
    print(f"   2. Remove bottom {len(feature_names) - features_90} features")
    print(f"   3. Remove one feature from each highly correlated pair")
    print(f"   4. Target: ~30-35 features (from {len(feature_names)})")
    
    print(f"\n🎯 Priority Categories to Keep:")
    for i, (cat, stats) in enumerate(sorted_categories[:5], 1):
        print(f"   {i}. {cat} (Avg importance: {stats['avg_importance']:.4f})")
    
    print(f"\n❌ Categories to Reduce:")
    for i, (cat, stats) in enumerate(sorted_categories[-3:], 1):
        print(f"   {i}. {cat} (Avg importance: {stats['avg_importance']:.4f})")
    
    # Save results
    feature_importance.to_csv('ema_crossover_feature_importance.csv', index=False)
    print(f"\n💾 Saved detailed analysis to: ema_crossover_feature_importance.csv")
    
    return feature_importance, category_importance


if __name__ == "__main__":
    try:
        feature_importance, category_importance = analyze_features()
        
        print("\n" + "="*80)
        print("✅ ANALYSIS COMPLETE!")
        print("="*80)
        print("\nNext steps:")
        print("1. Review the feature importance rankings")
        print("2. Run: python train_ema_crossover_optimized.py")
        print("   (This will automatically select the best ~30 features)")
        
    except FileNotFoundError:
        print("\n❌ Error: ema_crossover_with_targets.csv not found!")
        print("   Please run create_ema_crossover_targets.py first")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
