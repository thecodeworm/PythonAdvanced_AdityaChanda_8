"""
Test script for load_data.py
Run this to verify your data files are loading correctly.
"""

import sys
from pathlib import Path

# Add src to path so we can import our modules
sys.path.append(str(Path(__file__).parent / 'src'))

from data.load_data import load_data

def main():
    print("="*70)
    print("TESTING DATA LOADER")
    print("="*70)
    
    try:
        # Load the data
        df = load_data(data_dir="data/raw", validate=True)
        
        print("\n" + "="*70)
        print("✓ DATA LOADED SUCCESSFULLY!")
        print("="*70)
        
        print(f"\n📊 Dataset Info:")
        print(f"  - Total rows: {len(df)}")
        print(f"  - Columns: {list(df.columns)}")
        
        print(f"\n📅 Sample Data (first 5 rows):")
        print(df.head().to_string())
        
        print(f"\n📅 Sample Data (last 5 rows):")
        print(df.tail().to_string())
        
        # Check for snow days
        if 'SNOW' in df.columns:
            snow_days = df[df['SNOW'] > 0]
            print(f"\n❄️  Snow Statistics:")
            print(f"  - Total snow days: {len(snow_days)}")
            print(f"  - Biggest snowfall: {df['SNOW'].max():.1f} inches")
            print(f"\n  Top 5 snowiest days:")
            print(snow_days.nlargest(5, 'SNOW')[['DATE', 'SNOW', 'TMIN', 'TMAX']].to_string())
        
        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED!")
        print("="*70)
        print("\nYou're ready to move on to preprocessing! 🚀")
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\n📁 Make sure your files are organized like this:")
        print("   winter-smart-planner/")
        print("   ├── data/")
        print("   │   └── raw/")
        print("   │       ├── winter_2021_2022.csv")
        print("   │       ├── winter_2022_2023.csv")
        print("   │       ├── winter_2023_2024.csv")
        print("   │       └── winter_2024_2025.csv")
        print("   └── src/")
        print("       └── data/")
        print("           └── load_data.py")
        
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()