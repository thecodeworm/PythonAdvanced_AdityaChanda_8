"""
Test script for Severity Classifier
Trains and evaluates the winter severity classification model.
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from data.load_data import load_data
from data.preprocess import preprocess_data, create_feature_columns
from models.classifier import (
    train_severity_classifier,
    predict_severity,
    predict_severity_from_snowfall,
    get_severity_description
)


def main():
    print("="*70)
    print("SEVERITY CLASSIFIER TRAINING AND TESTING")
    print("="*70)
    
    try:
        # Load and preprocess data
        print("\nLoading Pittsburgh winter weather data...")
        df = load_data(
            data_file="data/raw/pittsburgh_winters_10years.csv",
            validate=False,
            filter_station=True,
            winter_only=True
        )
        print(f"Loaded {len(df)} days")
        
        df_processed = preprocess_data(df, add_features=True)
        feature_cols = create_feature_columns(df_processed)
        print(f"Preprocessed with {len(feature_cols)} features")
        
        # Train classifier
        clf, scaler = train_severity_classifier(
            df_processed,
            feature_cols,
            save_path="models/severity_classifier.pkl"
        )
        
        # Test predictions on sample days
        print("\n" + "="*70)
        print("TESTING SEVERITY PREDICTIONS")
        print("="*70)
        
        from models.classifier import create_severity_labels
        severities = create_severity_labels(df_processed)
        severity_names = {0: 'Mild', 1: 'Snowy', 2: 'Severe'}
        
        for severity_level in [0, 1, 2]:
            indices = np.where(severities == severity_level)[0]
            if len(indices) > 0:
                sample_indices = np.random.choice(indices, min(3, len(indices)), replace=False)
                
                print(f"{'='*70}")
                print(f"{severity_names[severity_level].upper()} CONDITIONS:")
                print(f"{'='*70}")
                
                for idx in sample_indices:
                    row = df_processed.iloc[idx]
                    print(f"Date: {row['DATE'].strftime('%Y-%m-%d')}")
                    print(f"Snowfall: {row['SNOW']:.1f}\"")
                    print(f"Temperature: {row['TAVG']:.1f}°F")
                    print(f"Wind: {row['AWND']:.1f} mph")
                    print(f"Wind Chill: {row.get('WIND_CHILL', row['TAVG']):.1f}°F")
                    print(f"Classification: {severity_names[severity_level]}")
                    print(f"{get_severity_description(severity_level)}\n")
        
        # Test rule-based predictions
        print("\n" + "="*70)
        print("RULE-BASED PREDICTION EXAMPLES")
        print("="*70)
        
        scenarios = [
            {"name": "Clear Winter Day", "snow": 0.0, "temp": 35, "wind": 5},
            {"name": "Light Dusting", "snow": 0.3, "temp": 30, "wind": 8},
            {"name": "Moderate Snowfall", "snow": 2.0, "temp": 25, "wind": 15},
            {"name": "Heavy Snow", "snow": 5.0, "temp": 20, "wind": 10},
            {"name": "Blizzard Conditions", "snow": 3.0, "temp": 15, "wind": 30},
            {"name": "Extreme Cold", "snow": 0.5, "temp": 5, "wind": 25},
        ]
        
        for scenario in scenarios:
            result = predict_severity_from_snowfall(
                scenario['snow'],
                scenario['temp'],
                scenario['wind']
            )
            print(f"{scenario['name']}:")
            print(f"  Snowfall: {result['snowfall']}\" | Temp: {result['temp']}°F | Wind: {result['wind']} mph")
            print(f"  Wind Chill: {result['wind_chill']:.1f}°F")
            print(f"  {result['name']}: {get_severity_description(result['severity'])}\n")
        
        print("="*70)
        print("Severity Classifier Complete")
        print("="*70)
        print("\nFiles saved:")
        print("  - models/severity_classifier.pkl")
        
        print("\nHow to use:")
        print("  1. For ML predictions: predict_severity(features)")
        print("  2. For quick predictions: predict_severity_from_snowfall(snow, temp, wind)")
        print("  3. In GUI: Use rule-based for instant feedback")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
