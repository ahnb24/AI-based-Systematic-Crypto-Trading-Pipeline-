import requests
import pandas as pd
import pytz
from utils.df_utils import (
    ffill_df_to_true_time_steps,
)

from utils.logging_tools import default_logger

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

    base_url = "https://api.binance.com"
    endpoint = "/api/v3/klines"
    interval = timeframe
    limit = number_of_days*288
    url = f"{base_url}{endpoint}?symbol={symbol}&interval={interval}&limit={limit}"

    # Fetch the data
    response = requests.get(url)
    data = response.json()
    # Define column names for the DataFrame
    columns = [
            '_time','open','high','low','close','tick_volume','1','2','3','4','5','6']

    # Convert the data to a DataFrame
    rates_df = pd.DataFrame(data, columns=columns)
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
    rates_df["_time"] = pd.to_datetime(rates_df["_time"], unit="s")

    if forward_fill and rates_df.shape[0] > 0:
        rates_df = ffill_df_to_true_time_steps(rates_df)
    
    rates_df = rates_df.drop(rates_df.index[-1])
    rates_df = rates_df.sort_values("_time").reset_index(drop=True)

    return rates_df


# def get_symbols_info(mt5,logger=default_logger):
#     """
#     Get all financial instruments info from the MetaTrader 5 terminal.

#     """
#     # ? list if vilable symbols:
#     symbols_dict = mt5.symbols_get()
#     symbols_dict[0]

#     symbols_info = {}
#     for item in symbols_dict:
#         temp_dict = {
#             "description": item.description,
#             "name": item.name,
#             "path": item.path,
#             "currency_base": item.currency_base,
#             "currency_profit": item.currency_profit,
#             "currency_margin": item.currency_margin,
#             "digits": item.digits,
#         }
#         symbols_info[item.name] = temp_dict

#     logger.info(f"--> number of all symbols: {len(symbols_info)}")
#     return symbols_info

def crawl_OHLCV_data_binance_one_symbol(
    symbol, data_size_in_days, forward_fill=False
):
    """
    get data for realtime loop.

    """
    timeframe = '5m'

    df = (
        crawl_data_from_binance(
            symbol, timeframe, data_size_in_days, forward_fill=forward_fill
        )
        .sort_values("_time")
        .reset_index(drop=True)
    )
    print(df)
    return df

crawl_OHLCV_data_binance_one_symbol("BCHUSDT", 90,True)
