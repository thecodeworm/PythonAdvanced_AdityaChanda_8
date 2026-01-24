"""
Data Loading Module for Smart Winter Planner
Loads Pittsburgh winter weather data and filters to winter months (Dec-Feb).
"""

import pandas as pd
from pathlib import Path


def filter_to_winter_months(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter data to only winter months (December, January, February).

    Args:
        df: DataFrame with DATE column

    Returns:
        DataFrame filtered to winter months only
    """
    if not pd.api.types.is_datetime64_any_dtype(df['DATE']):
        df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')

    df['MONTH'] = df['DATE'].dt.month
    winter_df = df[df['MONTH'].isin([12, 1, 2])].copy()
    winter_df = winter_df.drop('MONTH', axis=1)

    print(f"Filtered to winter months: {len(winter_df)} of {len(df)} total rows")
    return winter_df


def load_pittsburgh_data(filepath: str) -> pd.DataFrame:
    """
    Load Pittsburgh weather data from CSV.

    Args:
        filepath: Path to the CSV file

    Returns:
        DataFrame with loaded data
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} rows from {filepath}")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    except Exception as e:
        raise Exception(f"Error loading {filepath}: {str(e)}")


def filter_pittsburgh_station(df: pd.DataFrame, station_id: str = "USW00094823") -> pd.DataFrame:
    """
    Filter data to Pittsburgh International Airport station.

    Args:
        df: DataFrame with weather data
        station_id: Station ID to filter

    Returns:
        Filtered DataFrame
    """
    if 'STATION' not in df.columns:
        print("No STATION column found. Using all data as-is.")
        return df

    filtered_df = df[df['STATION'] == station_id].copy()
    print(f"Filtered to station {station_id}: {len(filtered_df)} rows retained")
    if filtered_df.empty:
        print(f"Warning: No data found for station {station_id}")
    return filtered_df


def validate_data(df: pd.DataFrame) -> None:
    """
    Validate the loaded data and print summary statistics.

    Args:
        df: DataFrame to validate
    """
    print("Data Validation Summary")
    print(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")

    if 'DATE' in df.columns:
        try:
            print(f"Date range: {df['DATE'].min()} to {df['DATE'].max()}")
            df_temp = df.copy()
            df_temp['YEAR'] = df_temp['DATE'].dt.year
            df_temp['MONTH'] = df_temp['DATE'].dt.month
            df_temp['SEASON'] = df_temp.apply(
                lambda row: f"{row['YEAR']}-{row['YEAR']+1}" if row['MONTH'] == 12 
                else f"{row['YEAR']-1}-{row['YEAR']}", axis=1
            )
            season_counts = df_temp['SEASON'].value_counts().sort_index()
            print("Winter seasons:")
            for season, count in season_counts.items():
                print(f"  {season}: {count} days")
        except:
            print("DATE column exists but format may vary.")

    required_cols = ['DATE', 'TMAX', 'TMIN', 'SNOW', 'PRCP']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Missing required columns: {missing_cols}")
    else:
        print("All required columns present.")

    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    for col in df.columns:
        if missing[col] > 0:
            print(f"{col}: {missing[col]} missing ({missing_pct[col]}%)")

    if 'SNOW' in df.columns:
        snow_days = (df['SNOW'] > 0).sum()
        print(f"Days with snow: {snow_days} ({snow_days/len(df)*100:.1f}%)")
        print(f"Max snowfall: {df['SNOW'].max():.1f} inches")
        print(f"Total snowfall: {df['SNOW'].sum():.1f} inches")
        if snow_days > 0:
            print(f"Average per snow day: {df[df['SNOW'] > 0]['SNOW'].mean():.2f} inches")

    if 'TMIN' in df.columns and 'TMAX' in df.columns:
        print(f"Temperature range: {df['TMIN'].min():.1f}°F to {df['TMAX'].max():.1f}°F")
        if 'TAVG' in df.columns:
            print(f"Average winter temp: {df['TAVG'].mean():.1f}°F")


def load_data(data_file: str = "data/raw/pittsburgh_winters_10years.csv",
              validate: bool = True,
              filter_station: bool = True,
              winter_only: bool = True) -> pd.DataFrame:
    """
    Load, filter, and optionally validate Pittsburgh winter weather data.

    Args:
        data_file: Path to CSV
        validate: Whether to print validation summary
        filter_station: Filter to Pittsburgh International station
        winter_only: Filter to winter months only

    Returns:
        Clean DataFrame ready for preprocessing
    """
    print("Loading Pittsburgh winter weather data...")

    df = load_pittsburgh_data(data_file)

    if filter_station:
        df = filter_pittsburgh_station(df)

    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')

    if winter_only:
        df = filter_to_winter_months(df)

    df = df.sort_values('DATE').reset_index(drop=True)

    if validate:
        validate_data(df)

    return df


if __name__ == "__main__":
    # Example usage
    try:
        df = load_data(data_file="data/raw/pittsburgh_winters_10years.csv")
        print(f"Loaded {len(df)} winter weather days.")
        print(df.head(10))
        print("Columns:", df.columns.tolist())
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Ensure the CSV exists at the specified path.")
