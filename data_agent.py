"""
HoneyQuant Data Processing Agent
Production-grade data ingestion, validation, and export system
"""

import logging
import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd
from binance.client import Client
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BinanceConnector:
    """Connector for fetching market data from Binance API"""
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        """
        Initialize Binance connector
        
        Args:
            api_key: Binance API key (optional, can use env var)
            api_secret: Binance API secret (optional, can use env var)
        """
        self.api_key = api_key or os.getenv('BINANCE_API_KEY', '')
        self.api_secret = api_secret or os.getenv('BINANCE_API_SECRET', '')
        
        if not self.api_key or not self.api_secret:
            logger.warning("Binance credentials not provided, using public client")
        
        self.client = Client(self.api_key, self.api_secret) if self.api_key else Client()
        logger.info("BinanceConnector initialized")
    
    def fetch_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500
    ) -> pd.DataFrame:
        """
        Fetch klines/candles from Binance
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            interval: Kline interval (e.g., '1m', '5m', '1h')
            limit: Number of klines to fetch (default: 500, max: 1000)
        
        Returns:
            DataFrame with columns: ["timestamp", "price", "volume"]
        """
        logger.info(f"Fetching {limit} klines for {symbol} at interval {interval}")
        
        try:
            klines = self.client.get_klines(symbol=symbol, interval=interval, limit=limit)
            
            if not klines:
                logger.warning(f"No klines returned for {symbol}")
                return pd.DataFrame(columns=["timestamp", "price", "volume"])
            
            # Extract timestamp, close price (price), and volume
            data = []
            for kline in klines:
                data.append({
                    "timestamp": pd.to_datetime(int(kline[0]), unit='ms', utc=True),
                    "price": float(kline[4]),  # Close price
                    "volume": float(kline[5])   # Volume
                })
            
            df = pd.DataFrame(data)
            logger.info(f"Successfully fetched {len(df)} klines")
            return df
        
        except Exception as e:
            logger.error(f"Error fetching klines: {e}")
            raise


class DataAgent:
    """Production-grade data processing agent with validation and export capabilities"""
    
    def __init__(self, groq_api_key: Optional[str] = None):
        """
        Initialize DataAgent
        
        Args:
            groq_api_key: Groq API key (optional, can use env var)
        """
        self.groq_api_key = groq_api_key or os.getenv('GROQ_API_KEY', '')
        
        if not self.groq_api_key:
            logger.warning("Groq API key not provided, LLM validation will be skipped")
            self.groq_client = None
        else:
            self.groq_client = Groq(api_key=self.groq_api_key)
            logger.info("DataAgent initialized with Groq client")
    
    def canonicalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Canonicalize dataframe: convert timestamps to UTC, sort, remove duplicates
        
        Args:
            df: Input dataframe with timestamp, price, volume columns
        
        Returns:
            Canonicalized dataframe
        """
        logger.info("Canonicalizing dataframe")
        
        if df.empty:
            logger.warning("Empty dataframe provided for canonicalization")
            return df
        
        # Ensure timestamp column exists and is datetime
        if 'timestamp' not in df.columns:
            raise ValueError("DataFrame must contain 'timestamp' column")
        
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        
        # Drop rows with missing timestamps
        initial_count = len(df)
        df = df.dropna(subset=['timestamp'])
        if len(df) < initial_count:
            logger.info(f"Dropped {initial_count - len(df)} rows with missing timestamps")
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Remove duplicates based on timestamp
        initial_count = len(df)
        df = df.drop_duplicates(subset=['timestamp'], keep='last').reset_index(drop=True)
        if len(df) < initial_count:
            logger.info(f"Removed {initial_count - len(df)} duplicate timestamps")
        
        logger.info(f"Canonicalization complete: {len(df)} rows remaining")
        return df
    
    def validate_rules(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate dataframe using rule-based checks
        
        Args:
            df: Input dataframe
        
        Returns:
            Tuple of (is_valid, list_of_anomalies)
        """
        logger.info("Running rule-based validation")
        
        anomalies = []
        
        if df.empty:
            anomalies.append("DataFrame is empty")
            return False, anomalies
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.any():
            for col, count in missing_values.items():
                if count > 0:
                    anomalies.append(f"Missing values in {col}: {count} rows")
        
        # Check for non-positive prices
        if 'price' in df.columns:
            non_positive_prices = (df['price'] <= 0).sum()
            if non_positive_prices > 0:
                anomalies.append(f"Non-positive prices: {non_positive_prices} rows")
        
        # Check for extreme jumps (>25% price change)
        if 'price' in df.columns and len(df) > 1:
            df_sorted = df.sort_values('timestamp')
            pct_change = df_sorted['price'].pct_change().abs()
            extreme_jumps = (pct_change > 0.25).sum()
            if extreme_jumps > 0:
                anomalies.append(f"Extreme price jumps (>25%): {extreme_jumps} occurrences")
        
        # Check for volume resets (volume=0 or sudden drops >90%)
        if 'volume' in df.columns and len(df) > 1:
            zero_volume = (df['volume'] == 0).sum()
            if zero_volume > 0:
                anomalies.append(f"Zero volume entries: {zero_volume} rows")
            
            df_sorted = df.sort_values('timestamp')
            volume_change = df_sorted['volume'].pct_change()
            sudden_drops = ((volume_change < -0.9) & (df_sorted['volume'].shift(1) > 0)).sum()
            if sudden_drops > 0:
                anomalies.append(f"Sudden volume drops (>90%): {sudden_drops} occurrences")
        
        is_valid = len(anomalies) == 0
        if is_valid:
            logger.info("Rule-based validation passed")
        else:
            logger.warning(f"Rule-based validation found {len(anomalies)} anomalies")
        
        return is_valid, anomalies
    
    def validate_llm(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Validate dataframe using Groq LLM for anomaly detection
        
        Args:
            df: Input dataframe
        
        Returns:
            Tuple of (is_valid, anomaly_report)
        """
        logger.info("Running LLM-based validation")
        
        if self.groq_client is None:
            logger.warning("Groq client not available, skipping LLM validation")
            return True, "LLM validation skipped (no API key)"
        
        if df.empty:
            return True, "Empty dataframe, no anomalies to detect"
        
        # Prepare sample data (first 50 rows)
        sample_df = df.head(50).copy()
        sample_df['timestamp'] = sample_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S UTC')
        
        # Convert to JSON
        sample_json = sample_df.to_json(orient='records', indent=2)
        
        # Create prompt for LLM
        prompt = f"""Analyze the following market data sample (first 50 rows) and detect any anomalies:

{sample_json}

Please check for:
- Timestamp anomalies (missing, duplicates, out of order)
- Data corruption (invalid values, nulls in critical fields)
- Fake candles (suspicious patterns, manipulated data)
- Liquidation wicks (extreme price spikes/drops)
- Unrealistic volume spikes (sudden massive volume changes)
- Any other weird behavior or suspicious patterns

Provide a bullet list of any anomalies found. If no anomalies are detected, respond with "No anomalies detected"."""

        try:
            response = self.groq_client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a quantitative data analyst expert at detecting anomalies in financial market data. Be thorough and specific."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=1024
            )
            
            anomaly_report = response.choices[0].message.content
            is_valid = "No anomalies detected" in anomaly_report or "no anomalies" in anomaly_report.lower()
            
            logger.info(f"LLM validation complete: {'passed' if is_valid else 'anomalies found'}")
            return is_valid, anomaly_report
        
        except Exception as e:
            logger.error(f"Error in LLM validation: {e}")
            return True, f"LLM validation error: {str(e)}"
    
    def build_resolutions(
        self,
        df: pd.DataFrame,
        resolutions: List[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Build OHLCV bars for multiple time resolutions
        
        Args:
            df: Input dataframe with timestamp, price, volume columns
            resolutions: List of resolution strings (default: ["1s", "1m", "5m"])
        
        Returns:
            Dictionary mapping resolution to OHLCV dataframe
        """
        if resolutions is None:
            resolutions = ["1s", "1m", "5m"]
        
        logger.info(f"Building OHLCV resolutions: {resolutions}")
        
        if df.empty:
            logger.warning("Empty dataframe provided for resolution building")
            return {res: pd.DataFrame() for res in resolutions}
        
        # Ensure timestamp is datetime and set as index
        df_work = df.copy()
        df_work['timestamp'] = pd.to_datetime(df_work['timestamp'], utc=True)
        df_work = df_work.set_index('timestamp').sort_index()
        
        result = {}
        
        for resolution in resolutions:
            try:
                # Resample to target resolution
                resampled = df_work.resample(resolution)
                
                # Build OHLCV
                ohlcv = pd.DataFrame({
                    'open': resampled['price'].first(),
                    'high': resampled['price'].max(),
                    'low': resampled['price'].min(),
                    'close': resampled['price'].last(),
                    'volume': resampled['volume'].sum()
                })
                
                # Remove rows with all NaN (empty time buckets)
                ohlcv = ohlcv.dropna(subset=['close'])
                ohlcv = ohlcv.reset_index()
                
                result[resolution] = ohlcv
                logger.info(f"Built {len(ohlcv)} bars for resolution {resolution}")
            
            except Exception as e:
                logger.error(f"Error building resolution {resolution}: {e}")
                result[resolution] = pd.DataFrame()
        
        return result
    
    def snapshot_hash(self, df: pd.DataFrame) -> str:
        """
        Compute deterministic SHA256 hash of dataframe
        
        Args:
            df: Input dataframe
        
        Returns:
            SHA256 hash string
        """
        logger.info("Computing snapshot hash")
        
        if df.empty:
            logger.warning("Empty dataframe provided for hashing")
            return hashlib.sha256(b'').hexdigest()
        
        # Create deterministic representation
        df_work = df.copy()
        
        # Ensure timestamp is in consistent format
        if 'timestamp' in df_work.columns:
            df_work['timestamp'] = pd.to_datetime(df_work['timestamp'], utc=True)
            df_work = df_work.sort_values('timestamp').reset_index(drop=True)
            # Convert to ISO format string for deterministic hashing
            df_work['timestamp'] = df_work['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        
        # Convert to CSV string with fixed format
        csv_string = df_work.to_csv(index=False, float_format='%.8f')
        
        # Compute hash
        hash_value = hashlib.sha256(csv_string.encode('utf-8')).hexdigest()
        logger.info(f"Snapshot hash computed: {hash_value[:16]}...")
        
        return hash_value
    
    def export_snapshot(
        self,
        df: pd.DataFrame,
        resolutions: Dict[str, pd.DataFrame],
        metadata: Dict,
        output_dir: str = "snapshots"
    ) -> str:
        """
        Export dataframe and resolutions to parquet files with manifest
        
        Args:
            df: Raw dataframe
            resolutions: Dictionary of resolution dataframes
            metadata: Metadata dictionary with symbol, etc.
            output_dir: Output directory path
        
        Returns:
            Path to manifest file
        """
        logger.info(f"Exporting snapshot to {output_dir}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Export raw data
        raw_path = os.path.join(output_dir, "data_raw.parquet")
        df.to_parquet(raw_path, index=False, engine='pyarrow')
        logger.info(f"Exported raw data: {raw_path}")
        
        # Export resolution data
        for resolution, res_df in resolutions.items():
            if not res_df.empty:
                res_path = os.path.join(output_dir, f"data_{resolution}.parquet")
                res_df.to_parquet(res_path, index=False, engine='pyarrow')
                logger.info(f"Exported {resolution} data: {res_path}")
        
        # Create manifest
        manifest = {
            "symbol": metadata.get("symbol", "UNKNOWN"),
            "exchange": metadata.get("exchange", "binance"),
            "snap_timestamp": datetime.now(timezone.utc).isoformat(),
            "row_count": len(df),
            "sha256": metadata.get("sha256", ""),
            "resolutions": list(resolutions.keys()),
            "version": "1.0"
        }
        
        manifest_path = os.path.join(output_dir, "manifest.json")
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Manifest created: {manifest_path}")
        return manifest_path


def main():
    """Main function demonstrating DataAgent workflow"""
    logger.info("Starting HoneyQuant Data Agent workflow")
    
    # Initialize components
    binance = BinanceConnector()
    agent = DataAgent()
    
    # Configuration
    symbol = "BTCUSDT"
    interval = "1m"
    limit = 500
    
    try:
        # Step 1: Fetch Binance data
        logger.info("=" * 50)
        logger.info("Step 1: Fetching Binance data")
        logger.info("=" * 50)
        df = binance.fetch_klines(symbol=symbol, interval=interval, limit=limit)
        
        # Step 2: Clean data
        logger.info("=" * 50)
        logger.info("Step 2: Cleaning data")
        logger.info("=" * 50)
        df_clean = agent.canonicalize(df)
        
        # Step 3: Rule-based validation
        logger.info("=" * 50)
        logger.info("Step 3: Rule-based validation")
        logger.info("=" * 50)
        is_valid_rules, anomalies = agent.validate_rules(df_clean)
        logger.info(f"Validation result: {'PASSED' if is_valid_rules else 'FAILED'}")
        if anomalies:
            for anomaly in anomalies:
                logger.warning(f"  - {anomaly}")
        
        # Step 4: LLM validation
        logger.info("=" * 50)
        logger.info("Step 4: LLM-based validation")
        logger.info("=" * 50)
        is_valid_llm, llm_report = agent.validate_llm(df_clean)
        logger.info(f"LLM validation result: {'PASSED' if is_valid_llm else 'ANOMALIES FOUND'}")
        logger.info(f"LLM Report:\n{llm_report}")
        
        # Step 5: Build resolutions
        logger.info("=" * 50)
        logger.info("Step 5: Building multi-resolution bars")
        logger.info("=" * 50)
        resolutions = agent.build_resolutions(df_clean)
        
        # Step 6: Compute hash
        logger.info("=" * 50)
        logger.info("Step 6: Computing snapshot hash")
        logger.info("=" * 50)
        snapshot_hash = agent.snapshot_hash(df_clean)
        logger.info(f"Snapshot hash: {snapshot_hash}")
        
        # Step 7: Export snapshot
        logger.info("=" * 50)
        logger.info("Step 7: Exporting snapshot")
        logger.info("=" * 50)
        metadata = {
            "symbol": symbol,
            "exchange": "binance",
            "sha256": snapshot_hash
        }
        manifest_path = agent.export_snapshot(df_clean, resolutions, metadata)
        
        logger.info("=" * 50)
        logger.info("Workflow complete!")
        logger.info(f"Manifest: {manifest_path}")
        logger.info("=" * 50)
    
    except Exception as e:
        logger.error(f"Error in main workflow: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

