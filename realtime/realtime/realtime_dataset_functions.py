
from realtime.realtime_utils import (

    merge_realtime_dataset,
)
import pytz
from realtime.realtime_utils import crawl_realtime_data_metatrader
from realtime.realtime_utils import crawl_realtime_data_binance, crawl_realtime_trade_data_binance
import time as tt
from datetime import datetime, timedelta, timezone
import pandas as pd
import polars as pl
import pickle

def merge_datasets(dataset1, dataset2, dedup_col="_time"):


    merged_dataset = {}

    for feature in dataset1:
        merged_dataset[feature] = {}
        for symbol in dataset1[feature]:
            df1 = dataset1[feature][symbol]
            df2 = dataset2[feature][symbol]

            if isinstance(df1, pd.DataFrame):
                merged_df = pd.concat([df1, df2]).drop_duplicates(subset=dedup_col).sort_values(by=dedup_col, ascending=False).reset_index(drop=True)
            elif isinstance(df1, pl.DataFrame):
                merged_df = df1.vstack(df2).unique(subset=[dedup_col]).sort(dedup_col, descending=True)
            else:
                raise TypeError(f"Unsupported type for {feature}/{symbol}")

            merged_dataset[feature][symbol] = merged_df

    return merged_dataset



def dataset_gen_realtime_loop(
    mode,
    fe_functions,
    feature_config,
    dataset_config,
    dataset={},
    data_size_in_days_ohlcv=60,
    data_size_in_days_trade=1.5,
    data_size_in_minutes_trade=1.5*1440,
    shape=[1,2,3],
    len_candle=60,
    indate_mins=15,
):
    print("-" * 38)
    t0 = tt.time()
    if mode == 'init':
        print("--> init. dataset.")
        # get data:

        dataset = crawl_realtime_data_binance(
            dataset,
            feature_config,
            mode="init",
            forward_fill=True,
            data_size_in_days_ohlcv=data_size_in_days_ohlcv,
        )
        crawl_time = datetime.now(timezone.utc)
        t15= tt.time()

        print(f"--> crawl time: {(tt.time() - t0):.2f}")

        dataset = crawl_realtime_trade_data_binance(
            dataset,
            feature_config,
            mode='init',
            data_size_in_minutes_trade=data_size_in_minutes_trade)
        print(f"--> trade_crawl time: {(tt.time() - t15):.2f}")
        crawl_time = datetime.now(timezone.utc)
        # create features:
        for func in fe_functions:
            dataset = func(dataset, feature_config)
        # print(dataset)
        t1 = tt.time()
        final_df = merge_realtime_dataset(dataset, dataset_config)
        shape[0] = final_df.shape[0]
        symbol = list(feature_config.keys())[0]
        shape[1] = dataset['st_one'][symbol].shape[0]
        len_candle = data_size_in_days_ohlcv

    elif mode == 'indate':

        print("--> indate dataset.")
        # ? update data:
        dataset = crawl_realtime_data_binance(
            dataset,feature_config,  mode="update", forward_fill=True, data_size_in_days_ohlcv=1
        )
        t15= tt.time()
        crawl_time = datetime.now(timezone.utc)
        print(f"--> crawl time: {(tt.time() - t0):.2f}")

        dataset = crawl_realtime_trade_data_binance(
                dataset,
                feature_config,
                mode='update',
                data_size_in_minutes_trade=indate_mins)

        print(f"--> trade_crawl time: {(tt.time() - t15):.2f}")

        # dataset = merge_datasets(dataset, dataset1)

        crawl_time = datetime.now(timezone.utc)

        now = datetime.now(timezone.utc)
        if now.hour == 0 and now.minute >= 0 and now.minute < 5 :
            for symbol in feature_config.keys():
                dataset['st_one'][symbol] = dataset['st_one'][symbol][-shape[1]:]
                x = 0
                for i in dataset['candles'][symbol]['_time']:
                    if '00:00:00' in str(i):
                        x+=1
                if x > len_candle:
                    dataset['candles'][symbol] = dataset['candles'][symbol][288:].reset_index(drop=True)


            # Ensure the '_time' column is in datetime format for all symbols

                df = dataset['st_one_trade'][symbol]
                df['_time'] = pd.to_datetime(df['_time'])

                # Extract the day from the first timestamp
                first_day = df['_time'].iloc[0].day
                cutoff_index = None
                day_count = 0
                seen_days = set([first_day])

                # Iterate over the DataFrame rows starting from index 1
                for i in range(1, len(df)):
                    current_day = df['_time'].iloc[i].day
                    if current_day not in seen_days:
                        seen_days.add(current_day)
                        day_count += 1
                    if day_count > int(data_size_in_days_trade):
                        cutoff_index = i
                        break
                if cutoff_index is not None:
                    dataset['st_one_trade'][symbol] = df[cutoff_index:].reset_index(drop=True)




        # ? create features:

        for func in fe_functions:
            dataset = func(dataset, feature_config)


        for key in dataset.keys():
            if key=="fe_time":
                dataset[key] = dataset[key].iloc[1:].reset_index(drop=True)
                continue
            for keyy in feature_config.keys():
                if type(dataset[key][keyy]) == pd.DataFrame:
                    dataset[key][keyy] = dataset[key][keyy].iloc[1:].reset_index(drop=True)
                elif type(dataset[key][keyy]) == pl.DataFrame:
                    dataset[key][keyy] = dataset[key][keyy].slice(1)


        t1 = tt.time()
        final_df = merge_realtime_dataset(dataset, dataset_config)
        final_df = final_df[-shape[0]:].reset_index(drop=True)
        # final_df = final_df.iloc[1:].reset_index(drop=True)





    elif mode == 'update':

        print("--> update dataset.")
        # ? update data:
        dataset = crawl_realtime_data_binance(
            dataset,feature_config,  mode="update", forward_fill=True, data_size_in_days_ohlcv=0
        )
        t15= tt.time()
        crawl_time = datetime.now(timezone.utc)
        print(f"--> crawl time: {(tt.time() - t0):.2f}")

        dataset = crawl_realtime_trade_data_binance(
                dataset,
                feature_config,
                mode='update',
                data_size_in_minutes_trade=5)

        print(f"--> trade_crawl time: {(tt.time() - t15):.2f}")

        crawl_time = datetime.now(timezone.utc)

        now = datetime.now(timezone.utc)
        if now.hour == 0 and now.minute >= 0 and now.minute < 30 :
            for symbol in feature_config.keys():
                dataset['st_one'][symbol] = dataset['st_one'][symbol][-shape[1]:]
                x = 0
                for i in dataset['candles'][symbol]['_time']:
                    if '00:00:00' in str(i):
                        x+=1
                if x > len_candle:
                    dataset['candles'][symbol] = dataset['candles'][symbol][288:].reset_index(drop=True)


            # Ensure the '_time' column is in datetime format for all symbols

                df = dataset['st_one_trade'][symbol]
                df['_time'] = pd.to_datetime(df['_time'])

                # Extract the day from the first timestamp
                first_day = df['_time'].iloc[0].day
                cutoff_index = None
                day_count = 0
                seen_days = set([first_day])

                # Iterate over the DataFrame rows starting from index 1
                for i in range(1, len(df)):
                    current_day = df['_time'].iloc[i].day
                    if current_day not in seen_days:
                        seen_days.add(current_day)
                        day_count += 1
                    if day_count > int(data_size_in_days_trade):
                        cutoff_index = i
                        break
                if cutoff_index is not None:
                    dataset['st_one_trade'][symbol] = df[cutoff_index:].reset_index(drop=True)




        # ? create features:

        for func in fe_functions:
            dataset = func(dataset, feature_config)


        for key in dataset.keys():
            if key=="fe_time":
                dataset[key] = dataset[key].iloc[1:].reset_index(drop=True)
                continue
            for keyy in feature_config.keys():
                if type(dataset[key][keyy]) == pd.DataFrame:
                    dataset[key][keyy] = dataset[key][keyy].iloc[1:].reset_index(drop=True)
                elif type(dataset[key][keyy]) == pl.DataFrame:
                    dataset[key][keyy] = dataset[key][keyy].slice(1)


        t1 = tt.time()
        final_df = merge_realtime_dataset(dataset, dataset_config)
        final_df = final_df[-shape[0]:].reset_index(drop=True)
        # final_df = final_df.iloc[1:].reset_index(drop=True)



    final_df.set_index("_time", inplace=True)
    final_df.sort_index(inplace=True)

    print(f"--> merge_columns time: {(tt.time() - t1):.2f}")
    print(f"dataset time: {(tt.time() - t0):.2f}")
    print("-" * 38)

    return dataset, final_df, crawl_time, shape, len_candle

def Error502_handling():
    with open("data/datasetdict.pkl", "rb") as f:
        dataset_dict = pickle.load(f)


    dataset = dataset_dict['dataset']

    symbols = list(dataset['st_one'].keys())

    for feature in dataset.keys():
        if feature != 'fe_time':
            for symbol in symbols:
                dataset[feature][symbol] = dataset[feature][symbol][:-2]
        else:
            dataset[feature] = dataset[feature][:-2]

    final_df = dataset_dict['final_df'][:-2]

    crawl_time = dataset_dict['crawl_time'] - timedelta(minutes=10)

    dataset_dict = {"dataset": dataset,
                    "final_df": final_df,
                    "crawl_time": crawl_time,
                    "shape": dataset_dict['shape'],
                    "len_candle": dataset_dict['len_candle']}

    with open("data/datasetdict.pkl", "wb") as f:
        pickle.dump(dataset_dict, f)