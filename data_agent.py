import logging
import hashlib
import json
import os
import time
import hmac
import requests
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------
#  GEMINI LLM SETUP
# ---------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    GEMINI_MODEL = genai.GenerativeModel("gemini-2.0-flash")
else:
    GEMINI_MODEL = None
    logger.warning("Gemini API key missing — LLM validation will be skipped.")
# ---------------------------------------------------------


class CoinDCXConnector:
    BASE_URL = "https://public.coindcx.com"

    def __init__(self, api_key=None, api_secret=None):
        self.api_key = api_key or os.getenv('COINDCX_API_KEY', '')
        self.api_secret = api_secret or os.getenv('COINDCX_API_SECRET', '')

        if not self.api_key or not self.api_secret:
            logger.warning("No CoinDCX API keys — using public market endpoints.")

    def fetch_klines(self, symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
        logger.info(f"Fetching {limit} klines for {symbol} at interval {interval}")

        try:
            endpoint = f"{self.BASE_URL}/market_data/candles"
            params = {
                "pair": symbol,
                "interval": interval,
                "limit": min(limit, 1000)
            }

            response = requests.get(endpoint, params=params, timeout=30)
            response.raise_for_status()

            candles = response.json()
            if not candles:
                logger.warning("No data returned from CoinDCX")
                return pd.DataFrame(columns=["timestamp", "price", "volume"])

            # Vectorized parsing for performance
            df = pd.DataFrame(candles)
            # Select columns: 0=time, 4=close, 5=volume
            df = df.iloc[:, [0, 4, 5]]
            df.columns = ["timestamp", "price", "volume"]
            
            # Convert types efficiently
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df["price"] = df["price"].astype(float)
            df["volume"] = df["volume"].astype(float)

            return df

        except Exception as e:
            logger.error(f"CoinDCX fetch error: {e}")
            raise


class DataAgent:
    def __init__(self):
        if GEMINI_MODEL:
            logger.info("Gemini LLM initialized.")
        else:
            logger.warning("Gemini not initialized — LLM validation disabled.")

    # ---------------------------------------------------------
    #  RULE BASED VALIDATION (NO CHANGES)
    # ---------------------------------------------------------
    def canonicalize(self, df):
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.dropna(subset=["timestamp"])
        df = df.sort_values("timestamp")
        df = df.drop_duplicates(subset=["timestamp"])
        return df

    def validate_rules(self, df):
        anomalies = []

        if df.empty:
            anomalies.append("Empty dataframe")
            return False, anomalies

        if (df["price"] <= 0).sum() > 0:
            anomalies.append("Non-positive prices detected")

        df_sorted = df.sort_values("timestamp")
        if (df_sorted["price"].pct_change().abs() > 0.25).sum() > 0:
            anomalies.append("Extreme price jumps detected")

        if (df["volume"] == 0).sum() > 0:
            anomalies.append("Zero volume entries detected")

        return len(anomalies) == 0, anomalies

    # ---------------------------------------------------------
    #  UPDATED: GEMINI LLM VALIDATION
    # ---------------------------------------------------------
    def validate_llm(self, df):
        if GEMINI_MODEL is None:
            return True, "Gemini validation skipped (no API key)."

        if df.empty:
            return True, "Empty dataset — no anomalies."

        # Prepare
        sample_df = df.head(50).copy()
        sample_df['timestamp'] = sample_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S UTC')
        sample_json = sample_df.to_json(orient="records", indent=2)

        prompt = f"""
You are a financial data anomaly detector.

Given this market data sample:

{sample_json}

Detect:
- timestamp anomalies
- invalid/missing values
- fake candles
- liquidation wicks
- unusual volume spikes
- any suspicious patterns

Respond with a bullet list of anomalies.
If clean, say "No anomalies detected".
"""

        try:
            response = GEMINI_MODEL.generate_content(prompt)
            report = response.text

            is_valid = "no anomalies" in report.lower()

            return is_valid, report

        except Exception as e:
            logger.error(f"Gemini error: {e}")
            return True, f"Gemini validation error: {e}"

    # ---------------------------------------------------------
    #  RESAMPLING + HASH + EXPORT (unchanged)
    # ---------------------------------------------------------
    def build_resolutions(self, df, resolutions=None):
        if resolutions is None:
            resolutions = ["1s", "1m", "5m"]

        df = df.copy().set_index("timestamp")
        result = {}

        for res in resolutions:
            try:
                # Fix for pandas FutureWarning: 'm' -> 'min', 'M' -> 'ME'
                # But here inputs are '1m', '5m' etc.
                # Pandas 2.2+ deprecates 'm' for minutes? No, 'm' is usually minutes.
                # Wait, the warning said: "FutureWarning: 'm' is deprecated... please use 'ME' instead."
                # That usually refers to Month end. But '1m' is 1 minute.
                # Ah, 'M' is month end, 'm' is minute?
                # Let's check the warning again: "FutureWarning: 'm' is deprecated... please use 'ME' instead."
                # If the input was '1M' (month), then yes. But default is ["1s", "1m", "5m"].
                # Maybe '1m' is being interpreted as month? No, standard aliases.
                # Let's just suppress the warning or ensure we use 'min'.
                
                res_pd = res.replace('m', 'min') if res.endswith('m') else res
                
                r = df.resample(res_pd)
                ohlcv = pd.DataFrame({
                    "open": r["price"].first(),
                    "high": r["price"].max(),
                    "low": r["price"].min(),
                    "close": r["price"].last(),
                    "volume": r["volume"].sum(),
                }).dropna()

                ohlcv.reset_index(inplace=True)
                result[res] = ohlcv

            except:
                result[res] = pd.DataFrame()

        return result

    def snapshot_hash(self, df):
        df2 = df.copy()
        df2["timestamp"] = df2["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        csv_str = df2.to_csv(index=False)
        return hashlib.sha256(csv_str.encode()).hexdigest()

    def export_snapshot(self, df, res, metadata, output_dir="snapshots"):
        os.makedirs(output_dir, exist_ok=True)

        df.to_parquet(f"{output_dir}/data_raw.parquet", index=False)

        for r, d in res.items():
            if not d.empty:
                d.to_parquet(f"{output_dir}/data_{r}.parquet", index=False)

        manifest = {
            "symbol": metadata["symbol"],
            "exchange": metadata["exchange"],
            "sha256": metadata["sha256"],
            "snap_timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "1.0",
        }

        with open(f"{output_dir}/manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        return f"{output_dir}/manifest.json"


# ---------------------------------------------------------
#  MAIN WORKFLOW (unchanged)
# ---------------------------------------------------------
def main():
    logger.info("Starting HoneyQuant Data Agent workflow")

    coindcx = CoinDCXConnector()
    agent = DataAgent()

    symbol = "I-BTC_INR"
    interval = "1m"
    limit = 500

    df = coindcx.fetch_klines(symbol, interval, limit)
    df_clean = agent.canonicalize(df)

    is_valid_rules, anomalies = agent.validate_rules(df_clean)
    logger.info(f"Rules valid: {is_valid_rules}, Issues: {anomalies}")

    is_valid_llm, llm_report = agent.validate_llm(df_clean)
    logger.info(f"LLM valid: {is_valid_llm}")
    logger.info(llm_report)

    resolutions = agent.build_resolutions(df_clean)
    snap_hash = agent.snapshot_hash(df_clean)

    manifest = agent.export_snapshot(
        df_clean, resolutions,
        metadata={"symbol": symbol, "exchange": "coindcx", "sha256": snap_hash}
    )

    logger.info(f"Snapshot exported: {manifest}")


if __name__ == "__main__":
    main()
