import sys
import os
# sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
# print(os.getcwd())
import requests
import pandas as pd
from datetime import datetime
import pytz
from configs.history_data_crawlers_config import metatrader_number_of_days
from utils.df_utils import ffill_df_to_true_time_steps
from utils.logging_tools import default_logger
from dotenv import load_dotenv
from pathlib import Path
from configs.history_data_crawlers_config import (
    data_folder,
    symbols_dict,
)
import os

load_dotenv()
broker_path = os.environ.get("BROKER_PATH")
timezone = pytz.timezone("Etc/UTC")

def crawl_OHLCV_data_binance(    
        feature_config: dict, logger=default_logger, number_of_days: int =metatrader_number_of_days, \
            forward_fill :bool =False,
) :
    logger.info(f"= " * 25)
    logger.info(f"--> start crawl_OHLCV_data_binance func:")

    data_source = "binance"
    folder_name = f"{data_folder}/{data_source}/"
    Path(folder_name).mkdir(parents=True, exist_ok=True)
    interval = "5m"

    # Define the endpoint and parameters
    base_url = "https://api.binance.com"
    endpoint = "/api/v3/klines"
    # symbol = "BTCUSDT"
    limit = 1000  # Binance API max limit per request is 1000
    total_samples = 100000000 # Total number of samples you want



    for symbol in list(feature_config.keys()):
        all_data = []
        logger.info(f"--> symbol: {symbol}")
        file_name = f"{folder_name}/{symbol}_{data_source}.parquet"
        symbol = symbols_dict[symbol]["binance_id"]
        # Start fetching data from now backwards
        current_time = None

        while len(all_data) < total_samples:
            # Construct the full URL with optional start time for pagination
            if current_time:
                url = f"{base_url}{endpoint}?symbol={symbol}&interval={interval}&limit={limit}&endTime={current_time}"
            else:
                url = f"{base_url}{endpoint}?symbol={symbol}&interval={interval}&limit={limit}"

            # Fetch the data
            response = requests.get(url)
            data = response.json()

            if len(data) == 0:
                break  # Stop if there's no more data

            # Append to all_data
            all_data += data

            # Update the current_time to be the open time of the oldest candle minus 1 ms
            current_time = data[0][0] - 1

        # Trim the data to only the first 10,000 samples
        all_data = all_data[:total_samples]

        # Define column names for the DataFrame
        columns = [
            '_time','open','high','low','close','tick_volume','1','2','3','4','5','6']

        # Convert the data to a DataFrame
        df = pd.DataFrame(all_data, columns=columns)

        # Clean the DataFrame (drop unnecessary columns)
        df.drop(columns=['1','2','3','4','5','6'], inplace=True)


        # Convert timestamp to a readable format
        df["_time"] = pd.to_datetime(df["_time"], unit='ms')

        # Convert numeric columns to appropriate types
        df['open'] = pd.to_numeric(df['open'], errors='coerce')
        df['high'] = pd.to_numeric(df['high'], errors='coerce')
        df['low'] = pd.to_numeric(df['low'], errors='coerce')
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df['tick_volume'] = pd.to_numeric(df['tick_volume'], errors='coerce')

        # Drop the last row (optional, based on your original code)
        df = df.drop(df.index[-1])


        # Save to CSV
        # df.to_csv('last10000h.csv', index=False)
        if df.shape[0] == 0:
            logger.info(f"!!! no data for {symbol} | skip this item")
            continue

        df = df.sort_values("_time").reset_index(drop=True)
        df["data_source"] = data_source
        df["symbol"] = symbol
        df.to_parquet(file_name, index=False)

    logger.info(f"--> crawl_OHLCV_data_binance run successfully.")
