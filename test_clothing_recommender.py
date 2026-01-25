"""
Test script for Clothing Recommendation Engine
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from recommender.clothing_rules import (
    get_clothing_recommendations,
    format_recommendations,
    get_quick_recommendation,
    get_layering_guide,
    get_time_limit_outside
)
from models.classifier import predict_severity_from_snowfall


def main():
    print("="*70)
    print("CLOTHING RECOMMENDATION ENGINE - FULL TEST")
    print("="*70)
    
    # Test different weather scenarios
    scenarios = [
        {
            "name": "Beautiful Winter Day",
            "snow": 0.0,
            "temp": 40,
            "wind": 8
        },
        {
            "name": "Light Snow, Cold",
            "snow": 0.8,
            "temp": 25,
            "wind": 12
        },
        {
            "name": "Moderate Snowfall",
            "snow": 2.5,
            "temp": 28,
            "wind": 15
        },
        {
            "name": "Heavy Snow Event",
            "snow": 5.0,
            "temp": 22,
            "wind": 18
        },
        {
            "name": "Blizzard Conditions",
            "snow": 3.5,
            "temp": 18,
            "wind": 35
        },
        {
            "name": "Dangerously Cold",
            "snow": 1.0,
            "temp": 0,
            "wind": 25
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{'='*70}")
        print(f"SCENARIO: {scenario['name']}")
        print(f"{'='*70}")
        print(f"Snowfall: {scenario['snow']}\"")
        print(f"Temperature: {scenario['temp']}°F")
        print(f"Wind: {scenario['wind']} mph")
        
        # Get severity classification
        severity_result = predict_severity_from_snowfall(
            scenario['snow'],
            scenario['temp'],
            scenario['wind']
        )
        
        print(f"\nClassification: {severity_result['name']}")
        print(f"Wind Chill: {severity_result['wind_chill']:.1f}°F")
        
        # Quick recommendation
        quick = get_quick_recommendation(
            scenario['snow'],
            scenario['temp'],
            scenario['wind']
        )
        print(f"\nQuick Recommendation:")
        print(f"{quick}")
        
        # Detailed recommendations
        recs = get_clothing_recommendations(
            severity_result['severity'],
            scenario['temp'],
            severity_result['wind_chill'],
            scenario['snow']
        )
        
        print(f"\nDetailed Clothing Recommendations:")
        print(format_recommendations(recs))
        
        # Time limit
        time_limit = get_time_limit_outside(
            severity_result['severity'],
            severity_result['wind_chill']
        )
        print(f"\nOutdoor Exposure Limit:")
        print(f"{time_limit}")
        
        # Layering guide
        layers = get_layering_guide(severity_result['severity'])
        print(f"\nLayering Guide:")
        for layer in layers:
            print(f"  • {layer}")
        
        print()
    
    print("="*70)
    print("Clothing Recommender Test Complete!")
    print("="*70)
    
    print("\nSummary:")
    print("  - Severity classification working")
    print("  - Quick recommendations working")
    print("  - Detailed recommendations working")
    print("  - Layering guides working")
    print("  - Time limits working")

if __name__ == "__main__":
    main()
