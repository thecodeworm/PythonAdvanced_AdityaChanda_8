"""
Test script for Pittsburgh winter weather data loading
Run this to verify your data file loads correctly.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from data.load_data import load_data

def main():
    print("="*70)
    print("TESTING PITTSBURGH DATA LOADER")
    print("="*70)
    
    try:
        # Load the data
        print("\nStep 1: Loading Pittsburgh winter weather data")
        
        # Update this path to match your file name
        df = load_data(
            data_file="data/raw/pittsburgh_winters_10years.csv",
            validate=True,
            filter_station=True,
            winter_only=True
        )
        
        print("\n" + "="*70)
        print("DATA LOADED SUCCESSFULLY!")
        print("="*70)
        
        print(f"\nDataset Info:")
        print(f"  - Total days: {len(df)}")
        print(f"  - Columns: {list(df.columns)}")
        
        print(f"\nFirst 10 rows:")
        print(df.head(10).to_string())
        
        print(f"\nLast 10 rows:")
        print(df.tail(10).to_string())
        
        # Check for snow days
        if 'SNOW' in df.columns:
            snow_days = df[df['SNOW'] > 0]
            print(f"\nSnow Statistics:")
            print(f"  - Total snow days: {len(snow_days)}")
            print(f"  - Percentage: {len(snow_days)/len(df)*100:.1f}%")
            print(f"  - Total snowfall: {df['SNOW'].sum():.1f} inches")
            print(f"  - Biggest snowfall: {df['SNOW'].max():.1f} inches")
            print("\nTop 10 snowiest days:")
            top_snow = snow_days.nlargest(10, 'SNOW')[['DATE', 'SNOW', 'TMIN', 'TMAX', 'TAVG']]
            print(top_snow.to_string())
        
        # Temperature stats
        if 'TAVG' in df.columns:
            print(f"\nTemperature Statistics:")
            print(f"  - Average: {df['TAVG'].mean():.1f}°F")
            print(f"  - Coldest: {df['TMIN'].min():.1f}°F")
            print(f"  - Warmest: {df['TMAX'].max():.1f}°F")
        
        print("\n" + "="*70)
        print("ALL TESTS PASSED!")
        print("="*70)
        print("\nYou're ready to preprocess and train!")
        print("Next step: python test_preprocessing.py")
        
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("\nExpected file location: data/raw/pittsburgh_winters_10years.csv")
        print("Tips:")
        print("  1. Make sure you downloaded the file from NOAA")
        print("  2. Save it with this exact name (or update the filename in the script)")
        print("  3. Put it in the data/raw/ folder")
        print("File should contain:")
        print("  - Station: USW00094823 (Pittsburgh International)")
        print("  - Date range: Dec 2014 - Feb 2025 (or your chosen range)")
        print("  - Variables: TMAX, TMIN, TAVG, SNOW, PRCP, AWND, etc.")
        
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        print("Common issues:")
        print("  - File format incorrect (should be CSV)")
        print("  - Missing required columns")
        print("  - Date format issues")

if __name__ == "__main__":
    main()
