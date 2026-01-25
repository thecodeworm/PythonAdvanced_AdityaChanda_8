"""
Test script for preprocessing pipeline with Pittsburgh data
Run this to verify feature engineering and data preparation works correctly.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from data.load_data import load_data
from data.preprocess import (
    preprocess_data,
    train_test_split_temporal,
    prepare_data_for_lstm,
    create_feature_columns
)


def main():
    print("="*70)
    print("TESTING PREPROCESSING PIPELINE - PITTSBURGH DATA")
    print("="*70)
    
    try:
        # Step 1: Load data
        print("\nSTEP 1: Loading Pittsburgh winter weather data...")
        df = load_data(
            data_file="data/raw/pittsburgh_winters_10years.csv",
            validate=False,
            filter_station=True,
            winter_only=True
        )
        print(f"Loaded {len(df)} winter days")
        
        # Step 2: Preprocess
        print("\nSTEP 2: Preprocessing and feature engineering...")
        df_processed = preprocess_data(
            df,
            add_features=True,
            handle_missing=True,
            missing_strategy='forward_fill'
        )
        
        # Show features created
        feature_cols = create_feature_columns(df_processed)
        print(f"Features created ({len(feature_cols)} total):")
        for i, col in enumerate(feature_cols, 1):
            print(f"  {i:2d}. {col}")
        
        # Show sample of processed data
        sample_cols = ['DATE', 'SNOW', 'TAVG', 'WIND_CHILL', 'SNOW_ROLL_3D', 'IS_FREEZING']
        available_cols = [col for col in sample_cols if col in df_processed.columns]
        print("\nSample of processed data:")
        print(df_processed[available_cols].head(10).to_string())
        
        # Step 3: Train/Test Split
        print("\nSTEP 3: Splitting into train/test...")
        train_df, test_df = train_test_split_temporal(df_processed, test_size=0.2)
        
        # Step 4: Prepare for LSTM
        print("\nSTEP 4: Preparing sequences for LSTM...")
        lstm_data = prepare_data_for_lstm(
            train_df,
            test_df,
            sequence_length=7,
            scaler_path="models/scaler.pkl"
        )
        
        # Display summary
        print("\n" + "="*70)
        print("Preprocessing test complete")
        print("="*70)
        print(f"Original data: {len(df)} days")
        print(f"After preprocessing: {len(df_processed)} days")
        print(f"Training sequences: {lstm_data['X_train'].shape[0]}")
        print(f"Test sequences: {lstm_data['X_test'].shape[0]}")
        print(f"Sequence length: {lstm_data['sequence_length']} days")
        print(f"Number of features: {lstm_data['X_train'].shape[2]}")
        
        # Snowfall statistics
        print("\nTraining set snowfall:")
        print(f"  Days with snow: {(lstm_data['y_train'] > 0).sum()} / {len(lstm_data['y_train'])} ({(lstm_data['y_train'] > 0).sum()/len(lstm_data['y_train'])*100:.1f}%)")
        print(f"  Total snowfall: {lstm_data['y_train'].sum():.1f} inches")
        print(f"  Average snowfall: {lstm_data['y_train'].mean():.2f} inches")
        print(f"  Max snowfall: {lstm_data['y_train'].max():.2f} inches")
        
        print("\nTest set snowfall:")
        print(f"  Days with snow: {(lstm_data['y_test'] > 0).sum()} / {len(lstm_data['y_test'])} ({(lstm_data['y_test'] > 0).sum()/len(lstm_data['y_test'])*100:.1f}%)")
        print(f"  Total snowfall: {lstm_data['y_test'].sum():.1f} inches")
        print(f"  Max snowfall: {lstm_data['y_test'].max():.2f} inches")
        
        print("\nAll tests passed. Data ready for LSTM model training.")
        print("="*70)
        
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Make sure your file is at: data/raw/pittsburgh_winters_10years.csv")
        
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
