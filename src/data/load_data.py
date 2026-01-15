"""
Data Loading Module for Smart Winter Planner
Loads and combines NOAA winter weather data from multiple CSV files.
"""

import pandas as pd
from pathlib import Path


def load_single_winter_file(filepath: str) -> pd.DataFrame:
    """
    Load a single winter CSV file from NOAA.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame with loaded data
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded {filepath}: {len(df)} rows")
        return df
    except FileNotFoundError:
        print(f"✗ File not found: {filepath}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading {filepath}: {str(e)}")
        return pd.DataFrame()


def load_all_winter_data(data_dir: str = "data/raw") -> pd.DataFrame:
    """
    Load and combine all winter CSV files from the data directory.
    
    Args:
        data_dir: Directory containing winter_YYYY_YYYY.csv files
        
    Returns:
        Combined DataFrame with all winter data
    """
    # Get all CSV files that match the winter pattern
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Find all winter CSV files
    winter_files = sorted(data_path.glob("winter_*.csv"))
    
    if not winter_files:
        raise FileNotFoundError(f"No winter_*.csv files found in {data_dir}")
    
    print(f"\nFound {len(winter_files)} winter data files:")
    for f in winter_files:
        print(f"  - {f.name}")
    
    # Load all files
    print("\nLoading files...")
    dfs = []
    for filepath in winter_files:
        df = load_single_winter_file(filepath)
        if not df.empty:
            dfs.append(df)
    
    if not dfs:
        raise ValueError("No data was successfully loaded!")
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    print(f"\n✓ Combined dataset: {len(combined_df)} total rows")
    print(f"  Date range: {combined_df['DATE'].min()} to {combined_df['DATE'].max()}")
    
    return combined_df


def filter_ohare_station(df: pd.DataFrame, station_id: str = "USW00094846") -> pd.DataFrame:
    """
    Filter data to only include O'Hare International Airport station.
    
    Args:
        df: DataFrame with weather data
        station_id: Station ID to filter (default: O'Hare)
        
    Returns:
        Filtered DataFrame
    """
    if 'STATION' not in df.columns:
        print("Warning: No STATION column found. Skipping filter.")
        return df
    
    original_count = len(df)
    filtered_df = df[df['STATION'] == station_id].copy()
    
    print(f"\nFiltered to station {station_id}:")
    print(f"  Kept {len(filtered_df)} rows (removed {original_count - len(filtered_df)})")
    
    if filtered_df.empty:
        print(f"Warning: No data found for station {station_id}")
    
    return filtered_df


def validate_data(df: pd.DataFrame) -> None:
    """
    Validate the loaded data and print summary statistics.
    
    Args:
        df: DataFrame to validate
    """
    print("\n" + "="*60)
    print("DATA VALIDATION SUMMARY")
    print("="*60)
    
    # Basic info
    print(f"\nDataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nDate Range: {df['DATE'].min()} to {df['DATE'].max()}")
    
    # Check for required columns
    required_cols = ['DATE', 'TMAX', 'TMIN', 'SNOW', 'PRCP']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"\nMissing required columns: {missing_cols}")
    else:
        print(f"\n✓ All required columns present")
    
    # Missing values summary
    print(f"\nMissing Values:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    
    for col in df.columns:
        if missing[col] > 0:
            print(f"  - {col}: {missing[col]} ({missing_pct[col]}%)")
    
    if missing.sum() == 0:
        print("  ✓ No missing values!")
    
    # Basic statistics for key columns
    print(f"\nKey Statistics:")
    if 'SNOW' in df.columns:
        snow_days = (df['SNOW'] > 0).sum()
        print(f"  - Days with snow: {snow_days} ({snow_days/len(df)*100:.1f}%)")
        print(f"  - Max snowfall: {df['SNOW'].max():.1f} inches")
    
    if 'TMIN' in df.columns and 'TMAX' in df.columns:
        print(f"  - Temperature range: {df['TMIN'].min():.1f}°F to {df['TMAX'].max():.1f}°F")
    
    print("="*60 + "\n")


def load_data(data_dir: str = "data/raw", 
              validate: bool = True,
              filter_station: bool = True) -> pd.DataFrame:
    """
    Main function to load, combine, and validate winter weather data.
    
    Args:
        data_dir: Directory containing CSV files
        validate: Whether to print validation summary
        filter_station: Whether to filter to O'Hare station only
        
    Returns:
        Clean DataFrame ready for preprocessing
    """
    print("\n" + "="*60)
    print("LOADING WINTER WEATHER DATA")
    print("="*60)
    
    # Load all CSV files
    df = load_all_winter_data(data_dir)
    
    # Filter to O'Hare if needed
    if filter_station:
        df = filter_ohare_station(df)
    
    # Convert DATE to datetime
    df['DATE'] = pd.to_datetime(df['DATE'])
    
    # Sort by date
    df = df.sort_values('DATE').reset_index(drop=True)
    
    # Validate if requested
    if validate:
        validate_data(df)
    
    return df


if __name__ == "__main__":
    # Example usage
    print("Testing data loading...\n")
    
    # Try to load data
    try:
        df = load_data(data_dir="../../data/raw")
        print(f"✓ Success! Loaded {len(df)} rows of data.")
        print(f"\nFirst few rows:")
        print(df.head())
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        print("\nMake sure your data files are in the correct location:")
        print("  data/raw/winter_2021_2022.csv")
        print("  data/raw/winter_2022_2023.csv")
        print("  etc.")