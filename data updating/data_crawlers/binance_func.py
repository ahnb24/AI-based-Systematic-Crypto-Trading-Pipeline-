import requests
import pandas as pd
import pytz
from utils.df_utils import (
    ffill_df_to_true_time_steps,
)

from utils.logging_tools import default_logger
import time
import os
from dotenv import load_dotenv
load_dotenv()
timezone = pytz.timezone("Etc/UTC")

def crawl_data_from_binance(
    symbol, timeframe, number_of_days, forward_fill=False,logger=default_logger
):
    """
    crawl data to pandas dataframe.
    """
    all_data = []
    base_url = "https://api.binance.com"
    endpoint = "/api/v3/klines"
    interval = timeframe
    limit = 1000  # Binance API max limit per request is 1000
    total_samples = (int(number_of_days*288/1000)+1)*1000 # Total number of samples you want
    
    current_time = None


    while len(all_data) < total_samples:
            # Construct the full URL with optional start time for pagination
            if current_time:
                url = f"{base_url}{endpoint}?symbol={symbol}&interval={interval}&limit={limit}&endTime={current_time}"
            else:
                url = f"{base_url}{endpoint}?symbol={symbol}&interval={interval}&limit={limit}"

            # Fetch the data
            for i in range(10):
                try:
                    time.sleep(1)
                    response = requests.get(url)
                    response.raise_for_status()
                    data = response.json()
                    break
                except Exception as e:
                    logger.exception(e)
                    if i == 9: # the last loop
                        raise

            if len(data) == 0:
                break  # Stop if there's no more data

            # Append to all_data
            all_data += data

            # Update the current_time to be the open time of the oldest candle minus 1 ms
            current_time = data[0][0] - 1

    all_data = all_data[:total_samples]
    # Define column names for the DataFrame
    columns = [
            '_time','open','high','low','close','tick_volume','1','2','3','4','5','6']

    # Convert the data to a DataFrame
    rates_df = pd.DataFrame(all_data, columns=columns)
          # Clean the DataFrame (drop unnecessary columns)
    rates_df.drop(columns=['1','2','3','4','5','6'], inplace=True)



    if rates_df.shape[0] == 0:
        logger.warning(f"!!! Binance: {symbol} no new data to crawl.")
        rates_df["_time"] = None
        return rates_df
    

    rates_df['open'] = pd.to_numeric(rates_df['open'], errors='coerce')
    rates_df['high'] = pd.to_numeric(rates_df['high'], errors='coerce')
    rates_df['low'] = pd.to_numeric(rates_df['low'], errors='coerce')
    rates_df['close'] = pd.to_numeric(rates_df['close'], errors='coerce')
    rates_df['tick_volume'] = pd.to_numeric(rates_df['tick_volume'], errors='coerce')
    rates_df["_time"] = pd.to_datetime(rates_df["_time"], unit="ms")
    # print(f'time:  {rates_df["_time"]}')
    rates_df = rates_df.sort_values("_time").reset_index(drop=True)

    
    if forward_fill and rates_df.shape[0] > 0:
        rates_df = ffill_df_to_true_time_steps(rates_df)
    
    rates_df = rates_df.drop(rates_df.index[-1])
    if number_of_days != 0:
        rates_df = rates_df.iloc[total_samples - (number_of_days*288)-1:].reset_index(drop=True)
    return rates_df


def crawl_OHLCV_data_binance_one_symbol(
    symbol, data_size_in_days_ohlcv, forward_fill=False
):
    """
    get data for realtime loop.

    """
    timeframe = '5m'
    if data_size_in_days_ohlcv==0:
        df = (
        crawl_data_from_binance(
            symbol, timeframe, data_size_in_days_ohlcv, forward_fill=forward_fill
        )
        .sort_values("_time")
        .reset_index(drop=True)
    ).iloc[-1]
        df = pd.DataFrame([df])
    else:
        df = (
            crawl_data_from_binance(
                symbol, timeframe, data_size_in_days_ohlcv, forward_fill=forward_fill
            )
            .sort_values("_time")
            .reset_index(drop=True)
        )
    return df

