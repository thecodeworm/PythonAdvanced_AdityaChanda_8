"""
Generate and display confusion matrix for severity classifier
"""

import numpy as np
import pickle
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data, create_feature_columns
from src.models.classifier import create_severity_labels
from src.evaluation.plots import ClassificationPlots
from src.evaluation.metrics import ClassificationMetrics

def main():
    print("="*70)
    print("SEVERITY CLASSIFIER CONFUSION MATRIX")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    df = load_data("data/raw/pittsburgh_winters_10years.csv", 
                   validate=False, filter_station=True, winter_only=True)
    df_processed = preprocess_data(df, add_features=True)
    
    # Create severity labels
    print("Creating severity labels...")
    y_true = create_severity_labels(df_processed)
    
    # Get features
    feature_cols = create_feature_columns(df_processed)
    X = df_processed[feature_cols].values
    
    # Load trained classifier
    print("Loading trained classifier...")
    try:
        with open("models/severity_classifier.pkl", 'rb') as f:
            saved_data = pickle.load(f)
        
        clf = saved_data['model']
        scaler = saved_data['scaler']
        
        # Make predictions
        print("Generating predictions...")
        X_scaled = scaler.transform(X)
        y_pred = clf.predict(X_scaled)
        
        # Calculate metrics
        print("\n" + "="*70)
        print("CLASSIFICATION METRICS")
        print("="*70)
        
        class_names = ['Mild', 'Snowy', 'Severe']
        metrics = ClassificationMetrics.calculate_all_metrics(
            y_true, y_pred, class_names=class_names
        )
        
        print(ClassificationMetrics.format_metrics_report(metrics))
        
        # Generate confusion matrix plot
        print("\nGenerating confusion matrix plot...")
        ClassificationPlots.plot_confusion_matrix(
            metrics['confusion_matrix'],
            class_names=class_names,
            title="Severity Classification Confusion Matrix",
            save_path="reports/figures/severity_confusion_matrix.png"
        )
        
        print("\n✓ Confusion matrix saved to reports/figures/severity_confusion_matrix.png")
        
        # Generate classification metrics plot
        print("\nGenerating classification metrics plot...")
        ClassificationPlots.plot_classification_report(
            metrics,
            save_path="reports/figures/severity_metrics.png"
        )
        
        print("✓ Metrics plot saved to reports/figures/severity_metrics.png")
        
        print("\n" + "="*70)
        print("COMPLETE!")
        print("="*70)
        
    except FileNotFoundError:
        print("\n✗ Error: Classifier not found!")
        print("Please train it first:")
        print("  python test_severity_classifier.py")

if __name__ == "__main__":
    main()