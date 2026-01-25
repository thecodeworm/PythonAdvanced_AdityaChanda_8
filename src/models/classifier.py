"""
Winter Severity Classifier
Categorizes predicted snowfall and weather conditions into severity levels:
- Mild: Light or no snow, manageable conditions
- Snowy: Moderate snow, use caution
- Severe: Heavy snow or dangerous conditions
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path
from typing import Dict, Tuple


def create_severity_labels(df: pd.DataFrame) -> pd.Series:
    """
    Create severity labels based on weather conditions.
    
    Rules:
    - SEVERE: Snowfall > 3" OR (Snowfall > 1.5" AND Wind > 20mph) OR Wind Chill < -10°F
    - SNOWY: Snowfall > 0.5" OR (Snowfall > 0.1" AND Temp < 25°F)
    - MILD: Everything else
    
    Args:
        df: DataFrame with SNOW, TAVG, AWND, WIND_CHILL columns
        
    Returns:
        Series with severity labels (0=Mild, 1=Snowy, 2=Severe)
    """
    severity = pd.Series(0, index=df.index)  # Default: Mild
    
    snow = df.get('SNOW', pd.Series(0, index=df.index))
    temp = df.get('TAVG', pd.Series(32, index=df.index))
    wind = df.get('AWND', pd.Series(0, index=df.index))
    wind_chill = df.get('WIND_CHILL', temp)
    
    # Snowy conditions
    snowy_mask = (snow > 0.5) | ((snow > 0.1) & (temp < 25))
    severity[snowy_mask] = 1
    
    # Severe conditions (overrides Snowy)
    severe_mask = (snow > 3.0) | ((snow > 1.5) & (wind > 20)) | (wind_chill < -10)
    severity[severe_mask] = 2
    
    return severity


def train_severity_classifier(df: pd.DataFrame, 
                              feature_cols: list,
                              save_path: str = "models/severity_classifier.pkl") -> Tuple[RandomForestClassifier, StandardScaler]:
    """
    Train a Random Forest classifier for severity prediction.
    
    Args:
        df: DataFrame with features and SNOW/TAVG/AWND columns
        feature_cols: List of feature column names
        save_path: Path to save the trained model
        
    Returns:
        Tuple of (trained_model, scaler)
    """
    print("Training Severity Classifier")
    
    # Create severity labels
    y = create_severity_labels(df)
    
    # Show distribution
    severity_names = {0: 'Mild', 1: 'Snowy', 2: 'Severe'}
    print("Severity distribution:")
    for level, name in severity_names.items():
        count = (y == level).sum()
        pct = count / len(y) * 100
        print(f"  {name}: {count} days ({pct:.1f}%)")
    
    # Prepare features
    X = df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Random Forest
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'
    )
    clf.fit(X_scaled, y)
    
    # Training accuracy
    train_acc = clf.score(X_scaled, y)
    print(f"Training accuracy: {train_acc*100:.1f}%")
    
    # Feature importance
    importances = clf.feature_importances_
    important_features = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)[:10]
    print("Top 10 important features:")
    for feat, imp in important_features:
        print(f"  {feat}: {imp:.3f}")
    
    # Save model
    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'wb') as f:
        pickle.dump({
            'model': clf,
            'scaler': scaler,
            'feature_cols': feature_cols,
            'severity_names': severity_names
        }, f)
    
    print(f"Model saved to {save_path}\n")
    
    return clf, scaler


def predict_severity(features: np.ndarray,
                     model_path: str = "models/severity_classifier.pkl") -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict severity levels for new data.
    
    Args:
        features: Feature array (samples, features)
        model_path: Path to saved model
        
    Returns:
        Tuple of (predicted_classes, probabilities)
    """
    with open(model_path, 'rb') as f:
        saved_data = pickle.load(f)
    
    clf = saved_data['model']
    scaler = saved_data['scaler']
    
    features_scaled = scaler.transform(features)
    predictions = clf.predict(features_scaled)
    probabilities = clf.predict_proba(features_scaled)
    
    return predictions, probabilities


def predict_severity_from_snowfall(snowfall: float,
                                   temp: float = 30.0,
                                   wind: float = 10.0,
                                   wind_chill: float = None) -> Dict:
    """
    Simple rule-based severity prediction from weather values.
    
    Args:
        snowfall: Predicted snowfall in inches
        temp: Temperature in °F
        wind: Wind speed in mph
        wind_chill: Wind chill in °F (optional)
        
    Returns:
        Dictionary with severity level and name
    """
    if wind_chill is None:
        if temp <= 50 and wind > 3:
            wind_chill = 35.74 + 0.6215*temp - 35.75*(wind**0.16) + 0.4275*temp*(wind**0.16)
        else:
            wind_chill = temp
    
    if (snowfall > 3.0) or ((snowfall > 1.5) and (wind > 20)) or (wind_chill < -10):
        severity = 2
        name = "Severe"
    elif (snowfall > 0.5) or ((snowfall > 0.1) & (temp < 25)):
        severity = 1
        name = "Snowy"
    else:
        severity = 0
        name = "Mild"
    
    return {
        'severity': severity,
        'name': name,
        'snowfall': snowfall,
        'temp': temp,
        'wind': wind,
        'wind_chill': wind_chill
    }


def get_severity_description(severity: int) -> str:
    """
    Get a human-readable description of the severity level.
    """
    descriptions = {
        0: "Mild winter conditions. Light or no snow expected. Normal activities possible.",
        1: "Snowy conditions. Moderate snowfall expected. Use caution when traveling.",
        2: "Severe winter weather. Heavy snow and/or dangerous conditions. Avoid unnecessary travel."
    }
    
    return descriptions.get(severity, "Unknown severity level")


if __name__ == "__main__":
    print("Severity Classifier Module")
    
    test_cases = [
        {"snowfall": 0.0, "temp": 35, "wind": 5, "name": "Clear day"},
        {"snowfall": 0.8, "temp": 28, "wind": 8, "name": "Light snow"},
        {"snowfall": 2.5, "temp": 25, "wind": 12, "name": "Moderate snow"},
        {"snowfall": 4.0, "temp": 20, "wind": 25, "name": "Heavy snow + wind"},
        {"snowfall": 0.5, "temp": 10, "wind": 20, "name": "Light snow, very cold"},
    ]
    
    for case in test_cases:
        result = predict_severity_from_snowfall(
            case['snowfall'], 
            case['temp'], 
            case['wind']
        )
        print(f"{case['name']}:")
        print(f"  Snowfall: {result['snowfall']}\"")
        print(f"  Temp: {result['temp']}°F")
        print(f"  Wind: {result['wind']} mph")
        print(f"  Wind Chill: {result['wind_chill']:.1f}°F")
        print(f"  Severity: {result['name']}")
        print(f"  {get_severity_description(result['severity'])}\n")

