"""
Data Preprocessing Module for Smart Winter Planner
Handles feature engineering, missing values, and data preparation for ML models.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
from pathlib import Path
import pickle


def calculate_wind_chill(temp_f: float, wind_mph: float) -> float:
    """
    Calculate wind chill using the NWS formula.
    Only valid for temps ≤ 50°F and wind speeds > 3 mph.
    
    Args:
        temp_f: Temperature in Fahrenheit
        wind_mph: Wind speed in miles per hour
        
    Returns:
        Wind chill temperature in Fahrenheit
    """
    if temp_f > 50 or wind_mph <= 3:
        return temp_f
    
    wind_chill = (35.74 + 
                  0.6215 * temp_f - 
                  35.75 * (wind_mph ** 0.16) + 
                  0.4275 * temp_f * (wind_mph ** 0.16))
    
    return round(wind_chill, 1)


def add_wind_chill_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add wind chill as a feature to the dataframe.
    
    Args:
        df: DataFrame with TMIN and AWND columns
        
    Returns:
        DataFrame with added WIND_CHILL column
    """
    df = df.copy()
    
    if 'TMIN' in df.columns and 'AWND' in df.columns:
        df['WIND_CHILL'] = df.apply(
            lambda row: calculate_wind_chill(row['TMIN'], row['AWND']), 
            axis=1
        )
        print("✓ Added WIND_CHILL feature")
    else:
        print("⚠️  Cannot calculate wind chill: missing TMIN or AWND")
    
    return df


def add_snow_specific_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features specifically designed to predict snowfall.
    
    Args:
        df: DataFrame with weather data
        
    Returns:
        DataFrame with added snow prediction features
    """
    df = df.copy()
    
    # Temperature threshold features (snow typically forms when temp < 32°F)
    if 'TAVG' in df.columns:
        df['IS_FREEZING'] = (df['TAVG'] <= 32).astype(int)
        df['TEMP_BELOW_FREEZING'] = np.maximum(0, 32 - df['TAVG'])  # How far below freezing
    
    if 'TMIN' in df.columns:
        df['TMIN_BELOW_FREEZING'] = np.maximum(0, 32 - df['TMIN'])
    
    # Precipitation + freezing = likely snow
    if 'PRCP' in df.columns and 'TAVG' in df.columns:
        df['FREEZING_PRECIP'] = df['PRCP'] * (df['TAVG'] <= 32).astype(int)
    
    # Humidity indicators (if available)
    # High humidity + cold = snow conditions
    
    # Recent snow accumulation (snow on ground increases likelihood of more)
    if 'SNWD' in df.columns:
        df['HAS_SNOW_COVER'] = (df['SNWD'] > 0).astype(int)
    
    # Temperature change (rapid drops often precede snow)
    if 'TAVG' in df.columns:
        df['TEMP_CHANGE_1D'] = df['TAVG'].diff()
        df['TEMP_CHANGE_3D'] = df['TAVG'].diff(3)
    
    # Count consecutive freezing days
    if 'TAVG' in df.columns:
        freezing = (df['TAVG'] <= 32).astype(int)
        df['CONSECUTIVE_FREEZE_DAYS'] = freezing.groupby((freezing != freezing.shift()).cumsum()).cumsum()
    
    print("✓ Added snow-specific features")
    
    return df


def add_rolling_features(df: pd.DataFrame, windows: list = [3, 7]) -> pd.DataFrame:
    """
    Add rolling average features for temperature and snowfall.
    
    Args:
        df: DataFrame with weather data
        windows: List of window sizes for rolling averages
        
    Returns:
        DataFrame with added rolling features
    """
    df = df.copy()
    
    for window in windows:
        # Rolling average temperature
        if 'TAVG' in df.columns:
            df[f'TAVG_ROLL_{window}D'] = df['TAVG'].rolling(window=window, min_periods=1).mean()
        
        # Rolling average snowfall
        if 'SNOW' in df.columns:
            df[f'SNOW_ROLL_{window}D'] = df['SNOW'].rolling(window=window, min_periods=1).mean()
        
        # Rolling average precipitation
        if 'PRCP' in df.columns:
            df[f'PRCP_ROLL_{window}D'] = df['PRCP'].rolling(window=window, min_periods=1).mean()
    
    print(f"✓ Added rolling features for windows: {windows}")
    
    return df


def add_lag_features(df: pd.DataFrame, lags: list = [1, 2, 3]) -> pd.DataFrame:
    """
    Add lag features for snowfall prediction.
    Helps model learn patterns like "it snowed yesterday, will it snow today?"
    
    Args:
        df: DataFrame with weather data
        lags: List of lag days
        
    Returns:
        DataFrame with added lag features
    """
    df = df.copy()
    
    for lag in lags:
        # Lag snowfall
        if 'SNOW' in df.columns:
            df[f'SNOW_LAG_{lag}D'] = df['SNOW'].shift(lag)
        
        # Lag temperature
        if 'TAVG' in df.columns:
            df[f'TAVG_LAG_{lag}D'] = df['TAVG'].shift(lag)
    
    print(f"✓ Added lag features for lags: {lags}")
    
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temporal features using sine/cosine encoding for cyclical patterns.
    
    Args:
        df: DataFrame with DATE column
        
    Returns:
        DataFrame with added temporal features
    """
    df = df.copy()
    
    if 'DATE' not in df.columns:
        print("⚠️  Cannot add temporal features: missing DATE column")
        return df
    
    # Day of year (1-365/366)
    df['DAY_OF_YEAR'] = df['DATE'].dt.dayofyear
    
    # Sine and cosine encoding for cyclical nature of seasons
    # This helps the model understand that Dec 31 and Jan 1 are close
    df['DAY_SIN'] = np.sin(2 * np.pi * df['DAY_OF_YEAR'] / 365.25)
    df['DAY_COS'] = np.cos(2 * np.pi * df['DAY_OF_YEAR'] / 365.25)
    
    # Month as categorical
    df['MONTH'] = df['DATE'].dt.month
    
    # Is it early winter (Dec) vs late winter (Feb)?
    df['IS_DECEMBER'] = (df['MONTH'] == 12).astype(int)
    df['IS_JANUARY'] = (df['MONTH'] == 1).astype(int)
    df['IS_FEBRUARY'] = (df['MONTH'] == 2).astype(int)
    
    print("✓ Added temporal features (day of year, sine/cosine encoding, month indicators)")
    
    return df


def handle_missing_values(df: pd.DataFrame, strategy: str = 'forward_fill') -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Args:
        df: DataFrame with potential missing values
        strategy: 'forward_fill', 'interpolate', or 'drop'
        
    Returns:
        DataFrame with handled missing values
    """
    df = df.copy()
    
    missing_before = df.isnull().sum().sum()
    
    if missing_before == 0:
        print("✓ No missing values to handle")
        return df
    
    print(f"\nHandling {missing_before} missing values using strategy: {strategy}")
    
    if strategy == 'forward_fill':
        # Forward fill then backward fill (for any leading NaNs)
        df = df.fillna(method='ffill').fillna(method='bfill')
        
    elif strategy == 'interpolate':
        # Linear interpolation
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_direction='both')
        
    elif strategy == 'drop':
        # Drop rows with any missing values
        df = df.dropna()
    
    missing_after = df.isnull().sum().sum()
    print(f"✓ Missing values reduced: {missing_before} → {missing_after}")
    
    return df


def create_feature_columns(df: pd.DataFrame) -> list:
    """
    Define which columns to use as features for the model.
    
    Args:
        df: DataFrame with all features
        
    Returns:
        List of feature column names
    """
    # Base weather features
    base_features = ['TAVG', 'TMAX', 'TMIN', 'AWND', 'PRCP', 'SNWD']
    
    # Engineered features
    engineered_features = [col for col in df.columns if any([
        'ROLL' in col,
        'LAG' in col,
        'WIND_CHILL' in col,
        'DAY_SIN' in col,
        'DAY_COS' in col,
        'IS_' in col
    ])]
    
    # Combine and filter to only existing columns
    all_features = base_features + engineered_features
    feature_cols = [col for col in all_features if col in df.columns]
    
    return feature_cols


def preprocess_data(df: pd.DataFrame,
                   add_features: bool = True,
                   handle_missing: bool = True,
                   missing_strategy: str = 'forward_fill') -> pd.DataFrame:
    """
    Main preprocessing pipeline that applies all transformations.
    
    Args:
        df: Raw DataFrame from load_data
        add_features: Whether to add engineered features
        handle_missing: Whether to handle missing values
        missing_strategy: Strategy for handling missing values
        
    Returns:
        Preprocessed DataFrame ready for modeling
    """
    print("\n" + "="*60)
    print("PREPROCESSING DATA")
    print("="*60)
    
    df = df.copy()
    
    # 1. Handle missing values first
    if handle_missing:
        df = handle_missing_values(df, strategy=missing_strategy)
    
    # 2. Add engineered features
    if add_features:
        print("\nAdding engineered features...")
        df = add_wind_chill_feature(df)
        df = add_temporal_features(df)
        df = add_snow_specific_features(df)  # NEW: Snow-specific features
        df = add_rolling_features(df, windows=[3, 7])
        df = add_lag_features(df, lags=[1, 2, 3])
    
    # 3. Handle any NaNs created by rolling/lag features
    # For the first few rows, use backward fill
    df = df.fillna(method='bfill')
    
    print("\n" + "="*60)
    print(f"✓ PREPROCESSING COMPLETE")
    print(f"  Final shape: {df.shape}")
    print(f"  Feature columns: {len(create_feature_columns(df))}")
    print("="*60 + "\n")
    
    return df


def create_sequences(data: np.ndarray, 
                    target: np.ndarray,
                    sequence_length: int = 7) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM training.
    Uses past N days to predict future snowfall.
    
    Args:
        data: Feature array (samples, features)
        target: Target array (samples,)
        sequence_length: Number of past days to use
        
    Returns:
        Tuple of (X_sequences, y_targets)
        X_sequences shape: (samples, sequence_length, features)
        y_targets shape: (samples,)
    """
    X, y = [], []
    
    for i in range(len(data) - sequence_length):
        # Get sequence of past days
        X.append(data[i:i + sequence_length])
        # Get target (next day's snowfall)
        y.append(target[i + sequence_length])
    
    return np.array(X), np.array(y)


def train_test_split_temporal(df: pd.DataFrame, 
                              test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/test sets while preserving temporal order.
    IMPORTANT: For time-series, we don't randomly shuffle!
    
    Args:
        df: Preprocessed DataFrame
        test_size: Fraction of data to use for testing
        
    Returns:
        Tuple of (train_df, test_df)
    """
    # Calculate split point
    split_idx = int(len(df) * (1 - test_size))
    
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    print(f"\n📊 Train/Test Split:")
    print(f"  Training set: {len(train_df)} days ({train_df['DATE'].min()} to {train_df['DATE'].max()})")
    print(f"  Test set: {len(test_df)} days ({test_df['DATE'].min()} to {test_df['DATE'].max()})")
    print(f"  Split ratio: {(1-test_size)*100:.0f}% train / {test_size*100:.0f}% test")
    
    return train_df, test_df


def normalize_features(train_df: pd.DataFrame,
                      test_df: pd.DataFrame,
                      feature_cols: list,
                      scaler_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Normalize features using StandardScaler.
    IMPORTANT: Fit on training data only, then transform both train and test.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        feature_cols: List of feature column names to normalize
        scaler_path: Optional path to save the fitted scaler
        
    Returns:
        Tuple of (train_normalized, test_normalized, scaler)
    """
    # Initialize scaler
    scaler = StandardScaler()
    
    # Fit on training data only
    train_features = train_df[feature_cols].values
    scaler.fit(train_features)
    
    # Transform both train and test
    train_normalized = scaler.transform(train_features)
    test_normalized = scaler.transform(test_df[feature_cols].values)
    
    print(f"\n✓ Normalized {len(feature_cols)} features")
    print(f"  Mean: {train_normalized.mean(axis=0)[:5]}")  # Show first 5
    print(f"  Std: {train_normalized.std(axis=0)[:5]}")
    
    # Save scaler if path provided
    if scaler_path:
        # Create directory if it doesn't exist
        scaler_dir = Path(scaler_path).parent
        scaler_dir.mkdir(parents=True, exist_ok=True)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"✓ Saved scaler to {scaler_path}")
    
    return train_normalized, test_normalized, scaler


def prepare_data_for_lstm(train_df: pd.DataFrame,
                         test_df: pd.DataFrame,
                         sequence_length: int = 7,
                         scaler_path: Optional[str] = None) -> dict:
    """
    Complete data preparation pipeline for LSTM model.
    
    Args:
        train_df: Training DataFrame (preprocessed)
        test_df: Test DataFrame (preprocessed)
        sequence_length: Number of past days to use for prediction
        scaler_path: Optional path to save the fitted scaler
        
    Returns:
        Dictionary containing all prepared data and metadata
    """
    print("\n" + "="*60)
    print("PREPARING DATA FOR LSTM")
    print("="*60)
    
    # Get feature columns
    feature_cols = create_feature_columns(train_df)
    print(f"\nUsing {len(feature_cols)} features:")
    print(f"  {feature_cols[:5]}...")  # Show first 5
    
    # Normalize features
    train_norm, test_norm, scaler = normalize_features(
        train_df, test_df, feature_cols, scaler_path
    )
    
    # Get target variable (SNOW)
    train_target = train_df['SNOW'].values
    test_target = test_df['SNOW'].values
    
    # Create sequences
    print(f"\nCreating sequences (length={sequence_length})...")
    X_train, y_train = create_sequences(train_norm, train_target, sequence_length)
    X_test, y_test = create_sequences(test_norm, test_target, sequence_length)
    
    print(f"✓ Training sequences: {X_train.shape}")
    print(f"✓ Test sequences: {X_test.shape}")
    
    print("="*60 + "\n")
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'sequence_length': sequence_length,
        'train_dates': train_df['DATE'].iloc[sequence_length:].values,
        'test_dates': test_df['DATE'].iloc[sequence_length:].values
    }


if __name__ == "__main__":
    # Example usage
    print("Testing preprocessing pipeline...\n")
    
    # This would normally import from load_data
    # from load_data import load_data
    # df = load_data(data_dir="../../data/raw")
    
    print("To use this module:")
    print("1. Load data: df = load_data('data/raw')")
    print("2. Preprocess: df = preprocess_data(df)")
    print("3. Split: train_df, test_df = train_test_split_temporal(df)")
    print("4. Prepare for LSTM: data = prepare_data_for_lstm(train_df, test_df)")