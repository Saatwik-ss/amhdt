"""
Data Harmonization Script: Prepare NIFTY 1-day OHLCV data to match Dow 30 format.

Dow 30 format (ml/data/dow30/1d_data.csv):
- Tab-delimited (\\t)
- Columns: DateTime, Open, High, Low, Close, Volume, TickVolume
- DateTime format: YYYY.MM.DD 00:00:00
- Pure 1-day OHLCV data

This script:
1. Reads all Nifty daily stock records from ml/data/nifty/NIFTY50_all.csv.
2. Computes the daily combined composite 1-day OHLCV bar for each date across the Nifty constituent universe.
3. Formats and writes ml/data/nifty/1d_data.csv matching Dow 30 schema, column names, and delimiter.
4. Also cleans and sorts Dow 30 data for consistent chronological order.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def prepare_nifty_1d_data(
    input_csv: Path = Path("ml/data/nifty/NIFTY50_all.csv"),
    output_csv: Path = Path("ml/data/nifty/1d_data.csv"),
) -> pd.DataFrame:
    """
    Combine NIFTY daily stock data into a single 1-day OHLCV dataset
    matching Dow 30 format.
    """
    print(f"Loading NIFTY data from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # Parse dates
    df["Date"] = pd.to_datetime(df["Date"])
    
    # Filter EQ series (equity)
    if "Series" in df.columns:
        df = df[df["Series"] == "EQ"]
        
    print(f"Loaded {len(df):,} rows across {df['Symbol'].nunique()} symbols and {df['Date'].nunique()} dates.")
    
    # Group by Date to compute composite 1-day OHLCV
    grouped = df.groupby("Date")
    
    daily_open = grouped["Open"].mean()
    daily_high = grouped["High"].mean()
    daily_low = grouped["Low"].mean()
    daily_close = grouped["Close"].mean()
    daily_volume = grouped["Volume"].sum()
    
    # TickVolume: use Trades if available and not all NaN, otherwise Volume
    if "Trades" in df.columns and df["Trades"].notna().sum() > 0:
        daily_tick_vol = grouped["Trades"].sum().fillna(0).astype(np.int64)
        daily_tick_vol = np.where(daily_tick_vol == 0, daily_volume, daily_tick_vol)
    else:
        daily_tick_vol = daily_volume
        
    nifty_1d = pd.DataFrame({
        "DateTime": daily_open.index.strftime("%Y.%m.%d 00:00:00"),
        "Open": daily_open.round(2),
        "High": daily_high.round(2),
        "Low": daily_low.round(2),
        "Close": daily_close.round(2),
        "Volume": daily_volume.astype(np.int64),
        "TickVolume": daily_tick_vol.astype(np.int64),
    }).reset_index(drop=True)
    
    # Ensure chronological order (oldest to newest)
    nifty_1d = nifty_1d.sort_values(by="DateTime").reset_index(drop=True)
    
    # Save as tab-separated CSV matching Dow 30 format
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    nifty_1d.to_csv(output_csv, sep="\t", index=False)
    print(f"✓ Saved harmonized NIFTY 1d OHLCV data to {output_csv} ({len(nifty_1d):,} trading days)")
    print(f"  Date range: {nifty_1d['DateTime'].iloc[0]} to {nifty_1d['DateTime'].iloc[-1]}")
    print(f"  Columns: {nifty_1d.columns.tolist()}")
    print("  Sample rows:")
    print(nifty_1d.head(3).to_string())
    
    return nifty_1d


def clean_dow30_1d_data(
    input_csv: Path = Path("ml/data/dow30/1d_data.csv"),
    output_csv: Path = Path("ml/data/dow30/1d_data_clean.csv"),
) -> pd.DataFrame:
    """
    Clean Dow 30 1d data: sort chronologically (oldest to newest)
    and ensure Volume reflects TickVolume if Volume is all 0.
    """
    print(f"\nProcessing DOW 30 data from {input_csv}...")
    df = pd.read_csv(input_csv, sep="\t")
    
    # Parse dates and sort oldest first
    df["dt"] = pd.to_datetime(df["DateTime"], format="%Y.%m.%d %H:%M:%S")
    df = df.sort_values(by="dt").reset_index(drop=True)
    df = df.drop(columns=["dt"])
    
    # If Volume is all 0 and TickVolume exists, populate Volume
    if df["Volume"].fillna(0).abs().sum() == 0 and "TickVolume" in df.columns:
        df["Volume"] = df["TickVolume"]
        
    df.to_csv(output_csv, sep="\t", index=False)
    print(f"✓ Saved cleaned DOW 30 1d data to {output_csv} ({len(df):,} trading days)")
    print(f"  Date range: {df['DateTime'].iloc[0]} to {df['DateTime'].iloc[-1]}")
    return df


if __name__ == "__main__":
    prepare_nifty_1d_data()
    clean_dow30_1d_data()
