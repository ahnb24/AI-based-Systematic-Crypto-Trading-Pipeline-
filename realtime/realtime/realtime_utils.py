import numpy as np
import pandas as pd
import polars as pl
import glob
import shutil
from pathlib import Path
import time as tt
import pytz
from polars import DataFrame as pl_DataFrame
from create_dataset.columns_merge_func import add_symbol_to_prefixes
from configs.history_data_crawlers_config import root_path, symbols_dict
from utils.clean_data import remove_weekends
from create_features.indicator_func import (
    cal_RSI_base_func,
    cal_EMA_base_func,
    cal_SMA_base_func,
    cal_ATR_func,
    cal_RSTD_func,
    add_candle_base_indicators_polars,
    add_all_ratio_by_config,
)
from create_features.ohlcv_features import (
    pvm,
    wbr,
    vda,
    ivs,
    cpp,
    vcr,
    vwpd,
    ccts,
    vsi,
    bbpr,
    pei,
    cas,
    trr,
    emi,
    vwvs,
)
from create_features.seventy_eight import (
    add_78
)
from create_features.trade_features import (
    trade_volume_imbalance,
    trade_count_ratio,
    trade_avg_trade_size,
    trade_price_volatility,
    trade_intensity,
    trade_direction_persistence,
    trade_cumulative_delta_volume,
    trade_aggression_speed,
    trade_price_skewness,
    trade_size_cv,
    trade_price_impact_proxy,
    trade_burstiness,
    trade_vwap_deviation,
    trade_realized_vs_trade_vol,
    liquidity_transition,
)

from create_features.create_basic_features_func import (
    add_candle_features,
    add_shifted_candle_features,
)
from utils.df_utils import (
    generate_true_time_df_pandas,
)
from realtime_candle.realtime_candle_func import (
    make_realtime_candle,
)

from data_crawlers.metatrader_func import (
    crawl_OHLCV_data_metatrader_one_symbol,
)

from data_crawlers.binance_func import (
    crawl_OHLCV_data_binance_one_symbol,
)
from data_crawlers.binance_trade_data_func import crawl_aggtrade_data_binance_one_symbol

from create_features.realtime_shift_func import create_shifted_col
from create_features.window_agg_features_func import add_win_fe_base_func
from feature_engine.datetime import DatetimeFeatures
from datetime import datetime, time, timedelta, timezone
from tzlocal import get_localzone
import re
from collections import OrderedDict
import pickle
from xgboost import XGBClassifier
import os

def load_all_pickles_from_dir():
    data_dict = {}
    directory = f"{root_path}/data/models"
    for filename in os.listdir(directory):
        if filename.endswith(".pkl"):
            file_path = os.path.join(directory, filename)
            with open(file_path, "rb") as f:
                try:
                    data = pickle.load(f)
                    data_dict[filename] = data
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
    return data_dict

def load_models(symbol_list, till_date, max_open_positions, bot):
    all_pickles = load_all_pickles_from_dir()
    models = {}
    symbol_dicts = []
    for symbol in symbol_list:
        models[symbol] = {}
        for pklfile in all_pickles.keys():
            if symbol in pklfile:
                match = re.search(r'-(\w+_\d+)_', pklfile)
                model_name = match.group(1).replace("_", "")
                models[symbol][model_name] = {}
                models[symbol][model_name]["model_name"] = model_name
                models[symbol][model_name]["model_path"] = f"{root_path}data/models/xgb_model_till_{till_date}-{match.group(1)}_simple.json"
                models[symbol][model_name]["info_path"] = f"{root_path}data/models/xgb_model_till_{till_date}-{match.group(1)}_simple_info.pkl"

                models[symbol][model_name]["target_details"] = {}
                models[symbol][model_name]["target_details"]['target_symbol'] = all_pickles[pklfile]['target_symbol']
                models[symbol][model_name]["target_details"]['trade_mode'] = all_pickles[pklfile]['trade_mode']
                models[symbol][model_name]["target_details"]['look_ahead'] = all_pickles[pklfile]['look_ahead']
                models[symbol][model_name]["target_details"]['take_profit'] = all_pickles[pklfile]['take_profit']
                models[symbol][model_name]["target_details"]['stop_loss'] = all_pickles[pklfile]['stop_loss']

                models[symbol][model_name]["strategy_details"] = {}
                models[symbol][model_name]["strategy_details"]['target_symbol'] = all_pickles[pklfile]['target_symbol']
                models[symbol][model_name]["strategy_details"]['trade_mode'] = all_pickles[pklfile]['trade_mode']
                models[symbol][model_name]["strategy_details"]['look_ahead'] = all_pickles[pklfile]['look_ahead']
                models[symbol][model_name]["strategy_details"]['take_profit'] = all_pickles[pklfile]['take_profit']
                models[symbol][model_name]["strategy_details"]['stop_loss'] = all_pickles[pklfile]['stop_loss']
                models[symbol][model_name]["strategy_details"]['max_open_positions'] = max_open_positions

        
        # sort keys by the number at the end
        sorted_keys = sorted(models[symbol].keys(), key=lambda x: int(re.search(r'\d+$', x).group()))

        # create a new ordered dict
        models_sorted = OrderedDict()
        for k in sorted_keys:
            models_sorted[k] = models[symbol][k]

        # Now aave_models_sorted keys are in order: AAVEUSDT1, AAVEUSDT2, AAVEUSDT3
        symbol_dicts.append(dict(models_sorted))


    models_list = symbol_dicts
                
    n = 0
    for sym in models_list:
        n+=len(sym)
    bot.send_message(f"No. of models to load: {n}")

    for models_info in models_list:
        for model_info in models_info:
            bot.send_message(f"---> model_id_name: {models_info[model_info]['model_name']}")
            model = XGBClassifier()
            model.load_model(models_info[model_info]["model_path"])
            models_info[model_info]["model_object"] = model
            with open(models_info[model_info]["info_path"], 'rb') as file:
                models_info[model_info]["input_cols"] = pickle.load(file)['input_cols']

    bot.send_message("--> models loaded.")

    return models_list



def get_coinex_time_now():
    # return (datetime.now(get_localzone()))
    return datetime.now(timezone.utc)


def is_forex_market_open():
    now = get_coinex_time_now()  # Get the current broker time
    weekday = now.weekday()  # Monday is 0 and Sunday is 6

    if weekday == 5 or weekday == 6:
        return False  # Saturday or Sunday
    else:
        return True  # Market is open


def sleep_until_next_run(run_every=5, offset_seconds=1, reporter=None):
    sleep_until_market_opens(reporter)
    now = get_coinex_time_now()
    next_run = now + (datetime.min.replace(tzinfo=timezone.utc) - now) % timedelta(minutes=run_every)
    wait_seconds = (next_run - now).total_seconds() + offset_seconds
    reporter.send_message(
        f"--> sleep for {wait_seconds} seconds for next candle"
    )
    # now = get_coinex_time_now()
    # next_run = now + (datetime.min - now) % timedelta(minutes=run_every)
    # wait_seconds = (next_run - now).total_seconds() + offset_seconds
    tt.sleep(wait_seconds)

    now = get_coinex_time_now()
    assert (
        now.minute % run_every == 0 and now.second == offset_seconds
    ), "!!! Wrong timing."
    now = now.strftime("%Y-%m-%d %H:%M:%S")
    reporter.send_message(f"Function executed at {now}")


def sleep_until_market_opens(reporter):
    if True:
        reporter.send_message("Crypto market is already open.")
        return

    now = get_coinex_time_now()
    weekday = now.weekday()

    if weekday == 5:  # Saturday
        # Sleep until 00:00 AM on Sunday
        open_time = (now + timedelta(days=2)).replace(
            hour=0, minute=5, second=0, microsecond=0
        )
    elif weekday == 6:  # Sunday before market opens
        # Sleep until 00:00 AM on Sunday
        open_time = (now + timedelta(days=1)).replace(
            hour=0, minute=5, second=0, microsecond=0
        )
    else:
        # The market is open on weekdays, no need to sleep
        return

    # Calculate the time to sleep
    time_to_sleep = (open_time - now).total_seconds() + 60 * 5
    reporter.send_message(
        f"Sleeping for {round(time_to_sleep/60,2)} minuts until the forex market opens."
    )
    reporter.send_message(
        f"market open_time will be: {open_time}."
    )
    tt.sleep(time_to_sleep)
    reporter.send_message("Forex market is now open.")


def merge_mean_dfs(df1, df2, mean_cols):
    merged_df = df1.merge(df2, on=["_time"], how="outer")
    for col in mean_cols:
        merged_df[col] = merged_df[[col + "_x", col + "_y"]].mean(axis=1)

    merged_df = merged_df.drop(
        columns=[col + "_x" for col in mean_cols] + [col + "_y" for col in mean_cols]
    )
    return merged_df


def realtime_stage_one(df):
    # ? remove weekends:
    # df = remove_weekends(df, weekends_day=["Saturday", "Sunday"], convert_tz=False)

    # print(df.isnull().sum())
    # ? check for null and inf:
    # assert df.isnull().sum().sum() == 0, "DataFrame contains null values"
    # assert np.isfinite(df).all().all(), "DataFrame contains infinite values"

    # ? Check order of OHLC makes sense
    assert (df["open"] <= df["high"]).all(), "open higher than high"
    assert (df["open"] >= df["low"]).all(), "open lower than low"
    assert (df["high"] >= df["low"]).all(), "high lower than low"

    # ? Check for outliers in returns
    # returns = df["close"].pct_change().iloc[1:]
    # assert returns.between(-0.35, 0.35).all(), "pct_change outlier returns detected"

    # ? check for big time gap
    time_diffs = df["_time"].diff().iloc[1:]
    assert time_diffs.between(
        pd.Timedelta("1min"),
        pd.Timedelta(days=10),
    ).all(), "Gaps detected in timestamps"

    # ? dtypes
    # assert df["_time"].dtypes == "datetime64[ns, Europe/Istanbul]", "Time column is not datetime"
    df[["open", "high", "low", "close", "tick_volume"]] = df[["open", "high", "low", "close", "tick_volume"]].astype(
        float
    )

    mask = df['tick_volume'] == 0

    # Forward fill the missing values first
    df.replace(0, np.nan, inplace=True)
    df.ffill(inplace=True)

    # Apply random adjustment within ±20% of the previous valid value
    for col in ['open', 'high', 'low', 'close', 'tick_volume']:
        df.loc[mask, col] *= np.random.uniform(0.999, 1.001, size=mask.sum())

    mask = (df['open'] == df['high']) & (df['high'] == df['low']) & (df['low'] == df['close'])

    # Generate random multipliers between 0.999 and 1.001 (±0.1%)
    random_factors = 1 + np.random.uniform(-0.003, 0.003, size=(mask.sum(), 4))

    # Apply the changes only to rows where all values are equal
    values = df.loc[mask, ['open', 'high', 'low', 'close']].values
    df.loc[mask, ['open', 'high', 'low', 'close']] = values * random_factors
        
    return df


def drop_first_day_pandas(df):
    # ? drop first day
    firt_date = df["_time"].dt.date[0]
    df["_date"] = df["_time"].dt.date
    df = df.loc[df["_date"] != firt_date]
    return df


def add_symbol_to_column_name_pandas(df, symbol,mode="no_symbol_name"):
    
    if mode=="no_symbol_name":
        df.rename(
            columns={
                "open": "M5_OPEN",
                "close": "M5_CLOSE",
                "low": "M5_LOW",
                "high": "M5_HIGH",
                "tick_volume": "M5_VOLUME",
            },
            inplace=True,
        )

    elif mode=="with_symbol":

        df.rename(
            columns={
                "open": symbol + "_M5_OPEN",
                "close": symbol + "_M5_CLOSE",
                "low": symbol + "_M5_LOW",
                "high": symbol + "_M5_HIGH",
                "tick_volume": f"{symbol}_M5_VOLUME",
            },
            inplace=True,
        )
    else:
        raise ValueError("!!!")
    
    return df


def generate_realtime_candle_realtime(df, symbol, tf_list=[15, 60, 240, 1440]):
    df = add_symbol_to_column_name_pandas(df, symbol)
    df_pl = pl.from_pandas(df)

    first_row_time = df_pl.row(0, named=True)["_time"]
    if time(0, 0) < first_row_time.time():
        df_pl = df_pl.filter(
            pl.col("_time").dt.date() > first_row_time.date()
        )  # Delete the first day that started from the middle of the day

    df_pl = df_pl.with_columns((pl.col("_time").dt.date()).alias("_date"))
    df_pl = df_pl.with_columns(
        (
            pl.col("_time").dt.minute().cast(pl.Int32, strict=False)
            + (pl.col("_time").dt.hour().cast(pl.Int32, strict=False)) * 60
        ).alias("minutesPassed")
    )


    df_pl = df_pl.with_columns(
        pl.when(pl.col("_time").dt.time().is_in(pl.time(0, 0, 0)))
        .then(1)
        .otherwise(0)
        .alias("isFirst")
    )


    df_pl = df_pl.with_row_index().with_columns(
        pl.col("index").cast(pl.Int32, strict=False).alias("index")
    )  
    df_pl = make_realtime_candle(df_pl, tf_list=tf_list, symbol=symbol)

    return df_pl


##? indicators functions: ------------------------------------------------------------------------
def add_RSI_to_realtime_dataset(dataset, feature_config):
    t0 = tt.time()
    modes = {
        "fe_RSI": {"func": cal_RSI_base_func},
        "fe_EMA": {"func": cal_EMA_base_func},
        "fe_SMA": {"func": cal_SMA_base_func},
        "fe_ATR": {"func": cal_ATR_func},
        "fe_RSTD": {"func": cal_RSTD_func},
    }

    for symbol in feature_config:

        symbol_ratio_dfs = []
        for fe_prefix in modes:
            if fe_prefix not in list(dataset.keys()):
                dataset[fe_prefix] = {}

            if fe_prefix not in list(feature_config[symbol].keys()):
                continue
            features_folder_path = f"{root_path}/data/realtime_cache/{fe_prefix}/"
            shutil.rmtree(features_folder_path, ignore_errors=True)
            Path(features_folder_path).mkdir(parents=True, exist_ok=True)

            try:
                base_cols = feature_config[symbol][fe_prefix]["base_columns"]
            except Exception as e:
                print(e)
                print(symbol)
                print(fe_prefix)
                raise ValueError("!!!")
            opts = {
                "symbol": symbol,
                "candle_timeframe": feature_config[symbol][fe_prefix]["timeframe"],
                "window_size": feature_config[symbol][fe_prefix]["window_size"],
                "features_folder_path": features_folder_path,
            }

            base_features = [
                f"M{tf}_{col}"
                for col in base_cols
                for tf in opts["candle_timeframe"]
            ]
            opts["base_feature"] = base_features
            needed_columns = ["_time", "minutesPassed"] + base_features
            df = dataset["candles"][symbol][needed_columns]

            add_candle_base_indicators_polars(
                df_base=df,
                prefix=fe_prefix,
                base_func=modes[fe_prefix]["func"],
                opts=opts,
            )

            # ? merge
            df = df[["_time"]]
            pathes = glob.glob(
                f"{features_folder_path}/unmerged/{fe_prefix}_**_{symbol}_*.parquet"
            )

            for df_path in pathes:
                df_loaded = pl.read_parquet(df_path)
                df = df.join(df_loaded, on="_time", how="left")

            max_candle_timeframe = max(opts["candle_timeframe"])
            max_window_size = max(opts["window_size"])
            drop_rows = (max_window_size + 1) * (max_candle_timeframe / 5) - 1

            df = df.with_row_count()

            df = (
                df.filter(pl.col("row_nr") >= drop_rows)
                .fill_null(strategy="forward")
                .drop(["row_nr"])
            )

            dataset[fe_prefix][symbol] = df

            # ?? add ratio:
            ratio_prefix = "fe_ratio"
            if ratio_prefix not in list(dataset.keys()):
                dataset[ratio_prefix] = {}
            if ratio_prefix not in list(feature_config[symbol].keys()):
                continue
            if fe_prefix.replace("fe_", "") in list(
                feature_config[symbol][ratio_prefix].keys()
            ):
                ratio_config = feature_config[symbol][ratio_prefix][
                    fe_prefix.replace("fe_", "")
                ]

            symbol_ratio_dfs.append(
                add_all_ratio_by_config(
                    df,
                    symbol,
                    fe_name=fe_prefix.replace("fe_", ""),
                    ratio_config=ratio_config,
                    fe_prefix="fe_ratio",
                )
            )

        # ? merge ratio for one symbol:
        if len(symbol_ratio_dfs) == 0:
            # print("!!! no ratio feature.")
            continue
        elif len(symbol_ratio_dfs) == 1:
            df = symbol_ratio_dfs[0]
        else:
            df = symbol_ratio_dfs[0]
            for i in range(1, len(symbol_ratio_dfs)):
                df = df.join(symbol_ratio_dfs[i], on="_time")

        dataset[ratio_prefix][symbol] = df
        # print(f'--> {fe_prefix}_{symbol} saved.')

    print(f"--> add_RSI_to_realtime_dataset done. time: {(tt.time() - t0):.2f}")

    return dataset


##? fe_shift: ------------------------------------------------------------------------
def add_fe_cndl_shift_fe_realtime_run(dataset, feature_config):
    t0 = tt.time()
    fe_prefix = "fe_cndl_shift"
    dataset[fe_prefix] = {}
    for symbol in feature_config:
        if fe_prefix not in list(feature_config[symbol].keys()):
            continue
        shift_columns = feature_config[symbol][fe_prefix]["columns"]
        shift_configs = feature_config[symbol][fe_prefix]["shift_configs"]

        sh_dfs = []
        df = dataset["candles"][symbol]

        for shift_config in shift_configs:
            timeframe = shift_config["timeframe"]
            shift_sizes = shift_config["shift_sizes"]

            for shift_size in shift_sizes:
                # print(f"symbol:{symbol} , timeframe: {timeframe} , shift_size: {shift_size} :")
                sh_df = create_shifted_col(
                    df, pair_name=symbol, periods=shift_size, time_frame=timeframe
                )
                new_cols = [
                    f"M{timeframe}_{col}_-{shift_size}"
                    for col in shift_columns
                ]
                sh_dfs.append(sh_df[["_time"] + new_cols])

        if len(sh_dfs) == 0:
            raise ValueError("!!! nothing to save.")

        shift_df = sh_dfs[0]
        if len(sh_dfs) > 1:
            for sh_df in sh_dfs[1:]:
                shift_df = shift_df.join(sh_df, on="_time", how="inner")

        dataset[fe_prefix][symbol] = shift_df

    return dataset


##? fe_cndl: ------------------------------------------------------------------------


def add_candle_fe(dataset, feature_config):
    t0 = tt.time()

    # ? -------------------
    fe_prefix = "fe_cndl"
    dataset[fe_prefix] = {}
    for symbol in feature_config:
        tf_list = feature_config[symbol]["fe_cndl"]
        dataset[fe_prefix][symbol] = add_candle_features(
            dataset["candles"][symbol], symbol, tf_list=tf_list, fe_prefix=fe_prefix
        )
        # print(f"--> fe_cndl {symbol} saved | category: {cat}")

    # ? -------------------
    fe_prefix = "fe_cndl_shift"
    # dataset[fe_prefix] = {}
    for symbol in feature_config:
        if fe_prefix not in list(feature_config[symbol].keys()):
            continue
        shift_configs = feature_config[symbol][fe_prefix]["shift_configs"]
        df_sh = dataset[fe_prefix][symbol]
        df_all = df_sh[["_time"]]

        for shift_config in shift_configs:
            timeframe = shift_config["timeframe"]
            shift_sizes = shift_config["shift_sizes"]

            df = add_shifted_candle_features(
                df_sh,
                tf=timeframe,
                shift_sizes=shift_sizes,
                fe_prefix=fe_prefix,
            )
            df_all = df_all.join(df, on="_time", how="left")

        dataset[fe_prefix][symbol] = df_all

    print(f"--> add_candle_fe done. time: {(tt.time() - t0):.2f}")
    return dataset


##? fe_WIN: ------------------------------------------------------------------------
def add_fe_win_realtime_run(dataset, feature_config, round_to=3, fe_prefix="fe_WIN"):
    t0 = tt.time()

    dataset[fe_prefix] = {}

    for symbol in feature_config:
        base_cols = feature_config[symbol][fe_prefix]["base_columns"]
        raw_features = [f"M5_{base_col}" for base_col in base_cols]
        needed_columns = ["_time", "minutesPassed"] + raw_features

        df = dataset["candles"][symbol][needed_columns].to_pandas()
        df.sort_values("_time", inplace=True)

        df = add_win_fe_base_func(
            df,
            symbol,
            raw_features=raw_features,
            timeframes=feature_config[symbol][fe_prefix]["timeframe"],
            window_sizes=feature_config[symbol][fe_prefix]["window_size"],
            round_to=round_to,
            fe_prefix=fe_prefix,
        )

        df.drop(columns=raw_features + ["minutesPassed"], inplace=True)

        dataset[fe_prefix][symbol] = pl.from_pandas(df)

    print(f"--> add_fe_win_realtime_run done. time: {(tt.time() - t0):.2f}")
    return dataset


##? real time candles: ------------------------------------------------------

def add_real_time_candles(dataset, feature_config):
    t0 = tt.time()

    fe_prefix = "candles"
    dataset[fe_prefix] = {}
    for symbol in feature_config:
        tf_list = feature_config[symbol]["base_candle_timeframe"]
        dataset[fe_prefix][symbol] = generate_realtime_candle_realtime(
            dataset["st_one"][symbol].copy(), symbol, tf_list=tf_list
        )

    t0 = tt.time()

    print(f"--> add_real_time_candles done. time: {(tt.time() - t0):.2f}")

    return dataset

def merge_realtime_dataset(dataset, dataset_config): 
    df_list = []
    for feature in dataset_config["features"]:
        match dataset[feature]:
            case pd.DataFrame():
                # print(f"flag 1: {feature}")
                df = dataset[feature]
                df = df.sort_values("_time").drop("symbol",errors='ignore')
                # df = df.rename(add_symbol_to_prefixes(df.columns, symbol))
                df_list.append(df)
            case pl.DataFrame():
                # print(f"flag 2: {feature}")
                df = dataset[feature].to_pandas()
                df = df.sort_values("_time").drop("symbol",errors='ignore')
                # df = df.rename(add_symbol_to_prefixes(df.columns, symbol))
                df_list.append(df)
            case dict():
                # print(f"flag 3: {feature}")
                for symbol in dataset[feature]:
                    # Check if it is a polars dataframe
                    if isinstance(dataset[feature][symbol], pl.DataFrame):
                        # print(feature)
                        df = dataset[feature][symbol].to_pandas()
                        df = df.sort_values("_time").drop("symbol",errors='ignore')
                        df.rename(columns=add_symbol_to_prefixes(df.columns, symbol),inplace=True)
                        # print(f"{feature} | {symbol}")
                        # print(f"df.columns: {df.columns}")
                        df_list.append(df)
                    # Check if it is a pandas dataframe
                    elif isinstance(dataset[feature][symbol], pd.DataFrame):
                        df = dataset[feature][symbol]
                        df = df.sort_values("_time").drop("symbol",errors='ignore')
                        df = df.rename(add_symbol_to_prefixes(df.columns, symbol))
                        df_list.append(df)
                    else:
                        raise ValueError(
                            f"Unsupported data type for {feature} -> {symbol}"
                        )
            case _:
                raise ValueError(f"Unsupported data type for {feature}")

    final_df = df_list[0]
    for df in df_list[1:]:
        final_df = final_df.merge(df, on="_time", how="inner")

    return final_df


def crawl_realtime_data_metatrader(
    mt5, dataset, feature_config, mode="init", forward_fill=True, data_size_in_days_ohlcv=12
):
    prefix = "st_one"
    if mode == "init":
        dataset[prefix] = {}
        for symbol in feature_config:
            symbol_m = symbols_dict[symbol]["metatrader_id"]

            df = crawl_OHLCV_data_metatrader_one_symbol(
                mt5, symbol_m, data_size_in_days_ohlcv, forward_fill=forward_fill
            )
            df.rename(
                columns={
                    "open": "open",
                    "high": "high",
                    "low": "low",
                    "close": "close",
                    "tick_volume": "Volume",
                },
                inplace=True,
            )

            df = realtime_stage_one(df)
            # df = drop_first_day_pandas(df)

            true_time_df = generate_true_time_df_pandas(df)
            df = true_time_df.merge(df, on=["_time"], how="left")
            # print(f"--> {symbol} nulls:",((df.isnull()["open"].sum()/df.shape[0])*100))
            df.sort_values("_time", inplace=True)
            df.drop(
                columns=["real_volume", "spread", "_date"],
                inplace=True,
                errors="ignore",
            )

            dataset[prefix][symbol] = df

    elif mode == "update":
        for symbol in feature_config:
            symbol_m = symbols_dict[symbol]["metatrader_id"]

            df = crawl_OHLCV_data_metatrader_one_symbol(
                mt5, symbol_m, data_size_in_days_ohlcv, forward_fill=forward_fill
            )
            df.rename(
                columns={
                    "open": "open",
                    "high": "high",
                    "low": "low",
                    "close": "close",
                    "tick_volume": "Volume",
                },
                inplace=True,
            )

            df = pd.concat([dataset["st_one"][symbol], df])
            df.drop_duplicates("_time", inplace=True, keep="last")
            df.sort_values("_time", inplace=True)

            
            df.drop(
                columns=["real_volume", "spread", "_date"],
                inplace=True,
                errors="ignore",
            )

            dataset["st_one"][symbol] = df

    return dataset

def crawl_realtime_data_binance(
    dataset, feature_config, mode="init", forward_fill=True, data_size_in_days_ohlcv=60
):
    prefix = "st_one"
    if mode == "init":
        dataset[prefix] = {}
        for symbol in feature_config:
            symbol_m = symbols_dict[symbol]["binance_id"]
            print(symbol_m)
            df = crawl_OHLCV_data_binance_one_symbol(
                symbol_m, data_size_in_days_ohlcv, forward_fill=forward_fill
            )


            df = realtime_stage_one(df)

            true_time_df = generate_true_time_df_pandas(df)
            df = true_time_df.merge(df, on=["_time"], how="left")
            df.sort_values("_time", inplace=True)


            dataset[prefix][symbol] = df

    elif mode == "update":
        for symbol in feature_config:
            symbol_m = symbols_dict[symbol]["binance_id"]

            df = crawl_OHLCV_data_binance_one_symbol(
                symbol_m, data_size_in_days_ohlcv, forward_fill=forward_fill
            )

            df = pd.concat([dataset["st_one"][symbol], df])
            df.drop_duplicates("_time", inplace=True, keep="last")
            df.sort_values("_time", inplace=True)



            dataset["st_one"][symbol] = df

    return dataset


def crawl_realtime_trade_data_binance(
    dataset, feature_config, mode="init", data_size_in_minutes_trade=2*1440
):
    prefix = "st_one_trade"
    if mode == "init":
        dataset[prefix] = {}
        for symbol in feature_config:
            symbol_m = symbols_dict[symbol]["binance_id"]
            print(symbol_m)
            df = crawl_aggtrade_data_binance_one_symbol(
                symbol_m, data_size_in_minutes_trade, mode=mode
            )


            # df = realtime_stage_one(df)

            # true_time_df = generate_true_time_df_pandas(df)
            # df = true_time_df.merge(df, on=["_time"], how="left")
            df.sort_values("_time", inplace=True)


            dataset[prefix][symbol] = df

    elif mode == "update":
        for symbol in feature_config:
            symbol_m = symbols_dict[symbol]["binance_id"]

            df = crawl_aggtrade_data_binance_one_symbol(
                symbol_m, data_size_in_minutes_trade, mode=mode
            )

            df = pd.concat([dataset["st_one_trade"][symbol], df])
            df.drop_duplicates("_time", inplace=True, keep="last")
            df.sort_values("_time", inplace=True)



            dataset["st_one_trade"][symbol] = df

    return dataset

## fe_time features -----------------------------------------------------------
def add_fe_time_realtime_run(dataset, feature_config):
    fe_prefix = "fe_time"
    dataset[fe_prefix] = {}

    df = (
        dataset["st_one"][list(feature_config.keys())[0]][["_time"]]
        .sort_values("_time")
        .reset_index(drop=True)
    )
    df["_time"] = df["_time"] - timedelta(hours=7)

    dtf = DatetimeFeatures(
        features_to_extract=[
            "month",
            "quarter",
            "semester",
            "week",
            "day_of_week",
            "day_of_month",
            "day_of_year",
            "month_start",
            "month_end",
            "quarter_start",
            "quarter_end",
            "year_start",
            "year_end",
            "hour",
            "minute",
        ],
        drop_original=False,
    )

    dtf.fit(df.rename(columns={"_time": fe_prefix}))
    fe_time = dtf.transform(df.rename(columns={"_time": fe_prefix})).rename(
        columns={fe_prefix: "_time"}
    )

    # section 2 -----------------
    markets_trade_times = {
        "New_York": (8, 17),
        "Tokyo": (19, 4),
        "Sydney": (15, 0),
        "London": (3, 11),
    }

    for market_name in markets_trade_times:
        start_time = markets_trade_times[market_name][0]
        stop_time = markets_trade_times[market_name][1]

        col_name = f"{fe_prefix}_isin_{market_name}"
        fe_time[col_name] = 0

        if stop_time > start_time:
            fe_time.loc[
                (fe_time["_time"].dt.hour >= start_time)
                & (fe_time["_time"].dt.hour < stop_time),
                col_name,
            ] = 1

        else:
            fe_time.loc[
                (fe_time["_time"].dt.hour >= start_time)
                | (fe_time["_time"].dt.hour < stop_time),
                col_name,
            ] = 1

    fe_time["_time"] = fe_time["_time"] + timedelta(hours=7)
    dataset[fe_prefix] = fe_time
    return dataset


def add_fe_market_close_realtime_run(dataset, feature_config):
    fe_prefix = "fe_market_close"
    dataset[fe_prefix] = {}
    ##? in EST time zone
    markets_trade_times = {
        "New_York": {
            "hour": 16,
            "minute": 55,
        },
        "Tokyo": {
            "hour": 3,
            "minute": 55,
        },
        "Sydney": {
            "hour": 23,
            "minute": 55,
        },
        "London": {
            "hour": 10,
            "minute": 55,
        },
    }

    for symbol in feature_config:
        df = (
            dataset["st_one"][symbol][["_time", "close"]]
            .sort_values("_time")
            .reset_index(drop=True)
        )
        df["_time"] = df["_time"] - timedelta(hours=7)

        for market in markets_trade_times:
            fiter = (df["_time"].dt.hour == markets_trade_times[market]["hour"]) & (
                df["_time"].dt.minute == markets_trade_times[market]["minute"]
            )

            df["last_close_price"] = None
            df.loc[fiter, "last_close_price"] = df.loc[fiter, "close"]

            df["last_close_time"] = None
            df.loc[fiter, "last_close_time"] = df.loc[fiter, "_time"]
            with pd.option_context("future.no_silent_downcasting", True):
                df = df.ffill(inplace=False).infer_objects(copy=False)
                # df.ffill(inplace=True)

            df[f"{fe_prefix}_{symbol}_{market}"] = (
                df["close"] - df["last_close_price"]
            ) / symbols_dict[symbol]["pip_size"]
            df[f"{fe_prefix}_{symbol}_{market}_time"] = (
                df["_time"] - df["last_close_time"]
            ).dt.seconds // 60

        ##? parquet save:
        df.drop(columns=["last_close_price", "close", "last_close_time"], inplace=True)
        df["_time"] = df["_time"] + timedelta(hours=7)
        df.dropna(inplace=True)
        df.dropna(inplace=True)

        dataset[fe_prefix][symbol] = df.reset_index(drop=True)

    return dataset


# prediction func --------------------
def predict_realtime(
    models_list, predictions, preds_df, crawl_time, final_df, reporter=None
):
    reporter.send_message("run predict_realtime:")
    predictions[crawl_time] = {}

    # for model_dict in models_list:
    #     reporter.send_message("-" * 15)
    #     reporter.send_message(f"model name:{model_dict['model_name']}")
    #     model_feature_names_in_ = model_dict["model_object"].model.feature_names_in_
    #     # model_feature_names_in_ = np.array(model_dict["model_object"].feature_importance["feature_name"])
    #     model_dtypes = model_dict["model_object"].input_cols
    #     x_p = final_df[model_feature_names_in_].iloc[-2:].astype(model_dtypes)
    #     # x_p = final_df[model_feature_names_in_].iloc[-2:]
    #     # x_p = final_df.iloc[-2:]

    #     predictions[crawl_time][model_dict["model_name"]] = {
    #         "model_id_name": model_dict["model_name"],
    #         # "target_name": models_dict[model_id_name]["target"],
    #         "_time": x_p.index[-1],
    #         "crawl_time": crawl_time,
    #         "predict_time": get_coinex_time_now(),
    #         "model_prediction": model_dict["model_object"].model.predict(x_p)[-1],
    #         # "model_prediction": 1,
    #         "strategy_trade_mode": model_dict["strategy_details"]["trade_mode"],
    #         "strategy_target_symbol": model_dict["strategy_details"]["target_symbol"],
    #         "strategy_look_ahead": model_dict["strategy_details"]["look_ahead"],
    #         "strategy_take_profit": model_dict["strategy_details"]["take_profit"],
    #         "strategy_stop_loss": model_dict["strategy_details"]["stop_loss"],
    #         "strategy_max_open_positions": model_dict["strategy_details"]["max_open_positions"],
    #     }

    #     reporter.send_message(
    #         f"model prediction: {predictions[crawl_time][model_dict['model_name']]['model_prediction']} , candle_time: {predictions[crawl_time][model_dict['model_name']]['_time']}"
    #     )
    #     if predictions[crawl_time][model_dict["model_name"]]["model_prediction"] == 1:
    #         reporter.send_message("💎💎💎 SIGNAL")
    
    # for coin in models_list:
    #     for model_dict in coin:
    #         reporter.send_message("-" * 15)
    #         reporter.send_message(f"model name:{coin[model_dict]['model_name']}")
    #         model_feature_names_in_ = coin[model_dict]["model_object"].model.feature_names_in_
    #         # model_feature_names_in_ = np.array(coin[model_dict]["model_object"].feature_importance["feature_name"])
    #         model_dtypes = coin[model_dict]["model_object"].input_cols
    #         x_p = final_df[model_feature_names_in_].iloc[-2:].astype(model_dtypes)
    #         # x_p = final_df[model_feature_names_in_].iloc[-2:]
    #         # x_p = final_df.iloc[-2:]

    #         predictions[crawl_time][coin[model_dict]["model_name"]] = {
    #             "model_id_name": coin[model_dict]["model_name"],
    #             # "target_name": models_dict[model_id_name]["target"],
    #             "_time": x_p.index[-1],
    #             "crawl_time": crawl_time,
    #             "predict_time": get_coinex_time_now(),
    #             "model_prediction": coin[model_dict]["model_object"].model.predict(x_p)[-1],
    #             # "model_prediction": 1,
    #             "strategy_trade_mode": coin[model_dict]["strategy_details"]["trade_mode"],
    #             "strategy_target_symbol": coin[model_dict]["strategy_details"]["target_symbol"],
    #             "strategy_look_ahead": coin[model_dict]["strategy_details"]["look_ahead"],
    #             "strategy_take_profit": coin[model_dict]["strategy_details"]["take_profit"],
    #             "strategy_stop_loss": coin[model_dict]["strategy_details"]["stop_loss"],
    #             "strategy_max_open_positions": coin[model_dict]["strategy_details"]["max_open_positions"],
    #         }

    #         reporter.send_message(
    #             f"model prediction: {predictions[crawl_time][coin[model_dict]['model_name']]['model_prediction']} , candle_time: {predictions[crawl_time][coin[model_dict]['model_name']]['_time']}"
    #         )
    #         if predictions[crawl_time][coin[model_dict]["model_name"]]["model_prediction"] == 1:
    #             reporter.send_message("💎💎💎 SIGNAL")
    #             break
    for coin in models_list:
        for model_dict in coin:
            model_feature_names_in_ = coin[model_dict]["model_object"].feature_names_in_
            model_dtypes = coin[model_dict]["input_cols"]
            x_p = final_df[model_feature_names_in_].iloc[-2:].astype(model_dtypes)
            
            predictions[crawl_time][coin[model_dict]["model_name"]] = {
                "model_id_name": coin[model_dict]["model_name"],
                "_time": x_p.index[-1],
                "crawl_time": crawl_time,
                "predict_time": get_coinex_time_now(),
                "model_prediction": coin[model_dict]["model_object"].predict(x_p)[-1],
                # "model_prediction": 1,
                "strategy_trade_mode": coin[model_dict]["strategy_details"]["trade_mode"],
                "strategy_target_symbol": coin[model_dict]["strategy_details"]["target_symbol"],
                "strategy_look_ahead": coin[model_dict]["strategy_details"]["look_ahead"],
                "strategy_take_profit": coin[model_dict]["strategy_details"]["take_profit"],
                "strategy_stop_loss": coin[model_dict]["strategy_details"]["stop_loss"],
                "strategy_max_open_positions": coin[model_dict]["strategy_details"]["max_open_positions"],
            }
            
            reporter.send_message(
                f"{'-' * 15}\n{coin[model_dict]['model_name']}: {predictions[crawl_time][coin[model_dict]['model_name']]['model_prediction']} \ncandle_time: {predictions[crawl_time][coin[model_dict]['model_name']]['_time']}"
            )
            if predictions[crawl_time][coin[model_dict]["model_name"]]["model_prediction"] == 1:
                reporter.send_message("💎💎💎 SIGNAL")
                break
            


    preds_df = pd.concat(
        [preds_df, pd.DataFrame(predictions[crawl_time]).T]
    ).reset_index(drop=True)
    return predictions, preds_df


# meta labeling
# def predict_realtime(
#     models_list, predictions, preds_df, crawl_time, final_df, reporter=None
# ):
#     reporter.send_message("run predict_realtime:")
#     predictions[crawl_time] = {}

#     for model_dict in models_list:
#         reporter.send_message("-" * 15)
#         reporter.send_message(f"model name:{model_dict['model_name']}")
#         y_preds = {}
#         y_probs = {}
#         Xms = {}
#         models = model_dict["model_object"]
#         n_layers = model_dict["n_layers"]
#         which_layer = n_layers - 1
#         for layer in range(which_layer+1):
#             # print(layer)
#             # predict with the prime model
#             if layer == 0:
#                 model_feature_names_in_ = model_dict["model_object"][f'model{layer}'].model.feature_names_in_
#                 # model_feature_names_in_ = np.array(model_dict["model_object"].feature_importance["feature_name"])
#                 model_dtypes = model_dict["model_object"][f'model{layer}'].input_cols
#                 x_p = final_df[model_feature_names_in_].iloc[-2:].astype(model_dtypes)
#                 # x_p = final_df[model_feature_names_in_].iloc[-2:]
#                 # x_p = final_df.iloc[-2:]
#                 current_dataset = x_p
#             else:
#                 current_dataset = Xms[f'Xm{layer-1}'].copy()

#             features = []
#             for column in current_dataset.columns:
#                 if "y_pred" in column or "y_prob" in column or "_time" in column or "fe" in column:
#                     features.append(column)
#             current_dataset = current_dataset[features]
#             y_preds[f"y_pred_{layer}"] = models[f'model{layer}'].predict(
#                 current_dataset).reshape(-1, 1)
#             # pd.DataFrame(y_preds[f"y_pred_{layer}"]).to_csv("y_pred_0.2.csv")
#             # print(f'model{layer} predict')
#             y_probs[f"y_prob_{layer}"] = models[f'model{layer}'].predict_proba(
#                 current_dataset)[:, 1]

#             # initialize the dataset for training the secondary models
#             Xms[f'Xm{layer}'] = current_dataset
#             Xms[f'Xm{layer}'][f"y_pred_{layer}"] = y_preds[f"y_pred_{layer}"]
#             Xms[f'Xm{layer}'][f"y_prob_{layer}"] = y_probs[f"y_prob_{layer}"]

#             # predict with the secondary models

#             # predict TP model
#             y_predtp = models[f'model{layer}TP'].predict(Xms[f'Xm{layer}']).reshape(-1, 1)
#             proba_predtp = models[f'model{layer}TP'].predict_proba(Xms[f'Xm{layer}'])[:, 1]

#             # predict FP model
#             y_predfp = models[f'model{layer}FP'].predict(Xms[f'Xm{layer}']).reshape(-1, 1)
#             proba_predfp = models[f'model{layer}FP'].predict_proba(Xms[f'Xm{layer}'])[:, 1]

#             # predict TN model
#             y_predtn = models[f'model{layer}TN'].predict(Xms[f'Xm{layer}']).reshape(-1, 1)
#             proba_predtn = models[f'model{layer}TN'].predict_proba(Xms[f'Xm{layer}'])[:, 1]

#             # predict FN model
#             y_predfn = models[f'model{layer}FN'].predict(Xms[f'Xm{layer}']).reshape(-1, 1)
#             proba_predfn = models[f'model{layer}FN'].predict_proba(Xms[f'Xm{layer}'])[:, 1]

#             Xms[f'Xm{layer}'][f"y_pred_{layer}TP"] = y_predtp
#             Xms[f'Xm{layer}'][f"y_prob_{layer}TP"] = proba_predtp

#             Xms[f'Xm{layer}'][f"y_pred_{layer}FP"] = y_predfp
#             Xms[f'Xm{layer}'][f"y_prob_{layer}FP"] = proba_predfp

#             Xms[f'Xm{layer}'][f"y_pred_{layer}TN"] = y_predtn
#             Xms[f'Xm{layer}'][f"y_prob_{layer}TN"] = proba_predtn

#             Xms[f'Xm{layer}'][f"y_pred_{layer}FN"] = y_predfn
#             Xms[f'Xm{layer}'][f"y_prob_{layer}FN"] = proba_predfn
            
#         # if which_layer == 0:
#         #     current_dataset = df.loc[folds[i][set_name]].copy()
#         # else:
#         current_dataset = Xms[f'Xm{which_layer}']

#         features = []
#         for column in current_dataset.columns:
#             if "y_pred" in column or "y_prob" in column or "_time" in column or "fe" in column:
#                 features.append(column)
#         current_dataset = current_dataset[features]
        
#         # predict with the final primary model
#         prediction = models[f'model{which_layer+1}'].predict(
#                 current_dataset).reshape(-1, 1)

        
        

#         predictions[crawl_time][model_dict["model_name"]] = {
#             "model_id_name": model_dict["model_name"],
#             # "target_name": models_dict[model_id_name]["target"],
#             "_time": x_p.index[-1],
#             "crawl_time": crawl_time,
#             "predict_time": get_coinex_time_now(),
#             "model_prediction": prediction[-1],
#             # "model_prediction": 1,
#             "strategy_trade_mode": model_dict["strategy_details"]["trade_mode"],
#             "strategy_target_symbol": model_dict["strategy_details"]["target_symbol"],
#             "strategy_look_ahead": model_dict["strategy_details"]["look_ahead"],
#             "strategy_take_profit": model_dict["strategy_details"]["take_profit"],
#             "strategy_stop_loss": model_dict["strategy_details"]["stop_loss"],
#             "strategy_max_open_positions": model_dict["strategy_details"]["max_open_positions"],
#         }

#         reporter.send_message(
#             f"model prediction: {predictions[crawl_time][model_dict['model_name']]['model_prediction']} , candle_time: {predictions[crawl_time][model_dict['model_name']]['_time']}"
#         )
#         if predictions[crawl_time][model_dict["model_name"]]["model_prediction"] == 1:
#             reporter.send_message("💎💎💎 SIGNAL")

#     # print(predictions[dataset_time][models_dict[model_id_name]["target"]])
#     preds_df = pd.concat(
#         [preds_df, pd.DataFrame(predictions[crawl_time]).T]
#     ).reset_index(drop=True)
#     return predictions, preds_df


# fe_ohlcv funcs

def add_ohlcv_non_w_to_realtime_dataset(dataset, feature_config):
    t0 = tt.time()
    non_w_features = {"fe_pvm": pvm,
                  "fe_wbr": wbr,
                  "fe_vda": vda,
                  "fe_ivs": ivs,
                  "fe_cpp": cpp,}

    for symbol in feature_config:

        for fe_prefix in non_w_features.keys():
            if fe_prefix not in list(dataset.keys()):
                dataset[fe_prefix] = {}

            if fe_prefix not in list(feature_config[symbol].keys()):
                continue
            features_folder_path = f"{root_path}/data/realtime_cache/{fe_prefix}/unmerged/"
            shutil.rmtree(features_folder_path, ignore_errors=True)
            Path(features_folder_path).mkdir(parents=True, exist_ok=True)


            df = dataset["candles"][symbol]
            df_all = df[["_time"]]
            timeframes = feature_config[symbol][fe_prefix]
            function = non_w_features[fe_prefix]
            for tf in timeframes:

                df1 = function(
                    df,
                    symbol=symbol,
                    timeframe=tf,
                    fe_prefix=fe_prefix,
                )
                df1 = df1.with_columns(pl.col("_time").cast(pl.Datetime("ns")))

                df_all = df_all.join(df1, on="_time", how="left", coalesce=True)

            df_all.write_parquet(f"{features_folder_path}/{fe_prefix}_{symbol}.parquet")
            dataset[fe_prefix][symbol] = df_all



    print(f"--> add_ohlcv_non_w_to_realtime_dataset done. time: {(tt.time() - t0):.2f}")

    return dataset


def add_ohlcv_w_to_realtime_dataset(dataset, feature_config):
    t0 = tt.time()
    w_features = {
              "fe_vcr": vcr,
              "fe_vwpd": vwpd,
              "fe_ccts": ccts,
              "fe_vsi": vsi,
              "fe_bbpr": bbpr,
              "fe_pei": pei,
              "fe_cas": cas,
              "fe_trr": trr,
              "fe_emi": emi,
              "fe_vwvs": vwvs,
              }

    for symbol in feature_config:

        for fe_prefix in w_features.keys():
            if fe_prefix not in list(dataset.keys()):
                dataset[fe_prefix] = {}

            if fe_prefix not in list(feature_config[symbol].keys()):
                continue
            features_folder_path = f"{root_path}/data/realtime_cache/{fe_prefix}/unmerged/"
            shutil.rmtree(features_folder_path, ignore_errors=True)
            Path(features_folder_path).mkdir(parents=True, exist_ok=True)


            df = dataset["candles"][symbol]
            df_all = df[["_time"]]
            timeframes = feature_config[symbol][fe_prefix]["timeframe"]
            window_size = feature_config[symbol][fe_prefix]["window_size"]
            function = w_features[fe_prefix]
            for tf in timeframes:
                for w in window_size:
                    df1 = function(
                            df,
                            symbol=symbol,
                            timeframe=tf,
                            window = w,
                            fe_prefix=fe_prefix,
                        )
                    df1 = df1.with_columns(pl.col("_time").cast(pl.Datetime("ns")))

                    df_all = df_all.join(df1, on="_time", how="left", coalesce=True)

            df_all.write_parquet(f"{features_folder_path}/{fe_prefix}_{symbol}.parquet")
            dataset[fe_prefix][symbol] = df_all



    print(f"--> add_ohlcv_w_to_realtime_dataset done. time: {(tt.time() - t0):.2f}")

    return dataset

# fe_78 function

def add_78_to_realtime_dataset(dataset, feature_config):

    t0 = tt.time()
    fe_prefix = 'fe_78'

    for symbol in feature_config:

        if fe_prefix not in list(dataset.keys()):
            dataset[fe_prefix] = {}

        if fe_prefix not in list(feature_config[symbol].keys()):
            continue
        features_folder_path = f"{root_path}/data/realtime_cache/{fe_prefix}/unmerged/"
        shutil.rmtree(features_folder_path, ignore_errors=True)
        Path(features_folder_path).mkdir(parents=True, exist_ok=True)


        df = dataset["candles"][symbol]
        df_all = df[["_time"]]
        timeframes = feature_config[symbol][fe_prefix]
        function = add_78
        for tf in timeframes:
            df1 = add_78(
                    df,
                    symbol=symbol,
                    timeframe=tf,
                    fe_prefix=fe_prefix,
                )
            df1 = df1.with_columns(pl.col("_time").cast(pl.Datetime("ns")))

            df_all = df_all.join(df1, on="_time", how="left", coalesce=True)

        df_all.write_parquet(f"{features_folder_path}/{fe_prefix}_{symbol}.parquet")
        dataset[fe_prefix][symbol] = df_all



    print(f"--> add_78_to_realtime_dataset done. time: {(tt.time() - t0):.2f}")

    return dataset


# fe_trade function

def add_trade_features_to_realtime_dataset(dataset, feature_config):
    t0 = tt.time()
    agg_only_features = {
                    "fe_vol_imb": trade_volume_imbalance,
                    "fe_trade_count_ratio": trade_count_ratio,
                    "fe_avg_trade_size": trade_avg_trade_size,
                    "fe_trade_price_vol": trade_price_volatility,
                    "fe_trade_intensity": trade_intensity,
                    "fe_dir_persist": trade_direction_persistence,
                    "fe_cum_delta": trade_cumulative_delta_volume,
                    "fe_aggr_speed": trade_aggression_speed,
                    "fe_price_skew": trade_price_skewness,
                    "fe_trade_size_cv": trade_size_cv,
                    "fe_trade_price_imp_prox": trade_price_impact_proxy,
                    "fe_trade_burstiness": trade_burstiness,
                    }
    combined_features = {
                    "fe_vwap_dev": trade_vwap_deviation,
                    "fe_vol_ratio": trade_realized_vs_trade_vol,
                    "fe_liq_trans": liquidity_transition,
                    }

    for symbol in feature_config:


        df_candle = dataset["candles"][symbol]
        # df_candle = pl.from_pandas(df_candle)
        df_agg = dataset["st_one_trade"][symbol]
        df_agg = pl.from_pandas(df_agg)

        for fe_prefix in agg_only_features.keys():
            if fe_prefix not in list(dataset.keys()):
                dataset[fe_prefix] = {}

            if fe_prefix not in list(feature_config[symbol].keys()):
                continue
            features_folder_path = f"{root_path}/data/realtime_cache/{fe_prefix}/unmerged/"
            shutil.rmtree(features_folder_path, ignore_errors=True)
            Path(features_folder_path).mkdir(parents=True, exist_ok=True)



            df_all = df_candle[["_time"]]
            timeframes = feature_config[symbol][fe_prefix]
            function = agg_only_features[fe_prefix]
            for tf in timeframes:

                df1 = function(
                    df_agg,
                    symbol=symbol,
                    timeframe=tf,
                    fe_prefix=fe_prefix,
                )
                df1 = df1.with_columns(pl.col("_time").cast(pl.Datetime("ns")))

                df_all = df_all.join(df1, on="_time", how="left", coalesce=True)
            
            # df_all = df_all.with_columns(pl.lit(symbol).alias("symbol"))
            df_all = df_all.with_columns([
                pl.col(col).fill_null(strategy="forward") for col in df_all.columns if col != "_time"])

            df_all.write_parquet(f"{features_folder_path}/{fe_prefix}_{symbol}.parquet")
            dataset[fe_prefix][symbol] = df_all
            del df_all
            del df1  

        for fe_prefix in combined_features.keys():
            if fe_prefix not in list(dataset.keys()):
                dataset[fe_prefix] = {}

            if fe_prefix not in list(feature_config[symbol].keys()):
                continue
            features_folder_path = f"{root_path}/data/realtime_cache/{fe_prefix}/unmerged/"
            shutil.rmtree(features_folder_path, ignore_errors=True)
            Path(features_folder_path).mkdir(parents=True, exist_ok=True)



            df_all = df_candle[["_time"]]
            timeframes = feature_config[symbol][fe_prefix]
            function = combined_features[fe_prefix]
            for tf in timeframes:

                df1 = function(
                        df_agg,
                        df_candle,
                        symbol=symbol,
                        timeframe=tf,
                        fe_prefix=fe_prefix,
                )
                df1 = df1.with_columns(pl.col("_time").cast(pl.Datetime("ns")))

                df_all = df_all.join(df1, on="_time", how="left", coalesce=True)
            
            # df_all = df_all.with_columns(pl.lit(symbol).alias("symbol"))
            df_all = df_all.with_columns([
                pl.col(col).fill_null(strategy="forward") for col in df_all.columns if col != "_time"])

            df_all.write_parquet(f"{features_folder_path}/{fe_prefix}_{symbol}.parquet")
            dataset[fe_prefix][symbol] = df_all
            del df_all
            del df1  
        del df_agg 
        del df_candle 
    print(f"--> add trade_features to realtime_dataset done. time: {(tt.time() - t0):.2f}")

    return dataset