from utils.logging_tools import default_logger
from configs.history_data_crawlers_config import root_path
import os
import pandas as pd
from create_dataset.columns_merge_func import history_columns_merge
import gc

features = [
    'fe_ATR',
    'fe_RSTD',
    'fe_WIN',
    'fe_cndl',
    'fe_EMA',
    'fe_SMA',
    'fe_RSI',
    'fe_cndl_shift',
    'fe_cndl_shift_raw',
    'fe_ratio',
    'fe_market_close',
    'fe_time',
]

def data_manipulation(config_general, prev_dir, logger=default_logger):
    logger.info("= " * 25)
    logger.info("--> start merge_with_previous_data func:")
    logger.info(f"--> prev_dir: {prev_dir}")
    if prev_dir == "pipeline_dir":
        parent_dir = os.path.dirname(os.path.dirname(root_path))
        prev_dir = f"{parent_dir}/pipeline/data"
    elif prev_dir == "local_dir":
        prev_dir = f"{root_path}/old_data"
    else:
        logger.info("wrong prev folder")
    # print(f"prev_dir: {prev_dir}")
    for symbol in list(config_general.keys()):
        # merge old & new stage_one
        logger.info(f"{symbol} - stage_one")
        st_one1_dir = f'{prev_dir}/stage_one_data/{symbol}_stage_one.parquet'
        st_one1 = pd.read_parquet(st_one1_dir)

        st_one2_dir = f"{root_path}/data/stage_one_data/{symbol}_stage_one.parquet"
        st_one2 = pd.read_parquet(st_one2_dir)

        st_one_all = pd.concat([st_one1, st_one2]).drop_duplicates(subset="_time", keep="first").reset_index(drop=True)
        st_one_big = st_one_all[:-288*250]
        st_one_small = st_one_all[-288*250:].reset_index(drop=True)

        dest_file_all1 = f"{root_path}/old_data/dataset/{symbol}/{symbol}_stage_one_all.parquet"
        dest_dir_all1 = os.path.dirname(dest_file_all1)
        os.makedirs(dest_dir_all1, exist_ok=True)
        st_one_all.to_parquet(dest_file_all1)

        dest_file_all2 = f"{root_path}/old_data/stage_one_data/{symbol}_stage_one.parquet"
        dest_dir_all2 = os.path.dirname(dest_file_all2)
        os.makedirs(dest_dir_all2, exist_ok=True)
        st_one_all.to_parquet(dest_file_all2)

        dest_file_big = f"{root_path}/old_data/dataset/{symbol}/{symbol}_stage_one_big.parquet"
        dest_dir_big = os.path.dirname(dest_file_big)
        os.makedirs(dest_dir_big, exist_ok=True)
        st_one_big.to_parquet(dest_file_big)

        dest_file_small = f"{root_path}/old_data/dataset/{symbol}/{symbol}_stage_one_small.parquet"
        dest_dir_small = os.path.dirname(dest_file_small)
        os.makedirs(dest_dir_small, exist_ok=True)
        st_one_small.to_parquet(dest_file_small)

        # merge old & new realtime_candle
        logger.info(f"{symbol} - realtime_candle")
        rc1_dir = f'{prev_dir}/realtime_candle/{symbol}_realtime_candle.parquet'
        rc1 = pd.read_parquet(rc1_dir)

        rc2_dir = f"{root_path}/data/realtime_candle/{symbol}_realtime_candle.parquet"
        rc2 = pd.read_parquet(rc2_dir)

        rc = pd.concat([rc1, rc2]).drop_duplicates(subset="_time", keep="first").reset_index(drop=True)
        dest_file = f"{root_path}/old_data/realtime_candle/{symbol}_realtime_candle.parquet"
        dest_dir = os.path.dirname(dest_file)
        os.makedirs(dest_dir, exist_ok=True)
        rc.to_parquet(dest_file)

    del rc, rc2, rc1, st_one_all, st_one_big, st_one_small, st_one2, st_one1
    gc.collect()

    # merge old & new features
    for symbol in list(config_general.keys()):
        for feature in features:
            logger.info(f"--> {symbol} - {feature}")
            if feature == 'fe_time':
                prev_file = f"{prev_dir}/features/{feature}/{feature}.parquet"
                curr_file = f"{root_path}/data/features/{feature}/{feature}.parquet"
                dest_file = f"{root_path}/old_data/features/{feature}/{feature}.parquet"

            elif feature == 'fe_cndl_shift_raw':
                prev_file = f"{prev_dir}/features/{feature}/fe_cndl_shift_{symbol}.parquet"
                curr_file = f"{root_path}/data/features/{feature}/fe_cndl_shift_{symbol}.parquet"
                dest_file = f"{root_path}/old_data/features/{feature}/fe_cndl_shift_{symbol}.parquet"

            else:
                prev_file = f"{prev_dir}/features/{feature}/{feature}_{symbol}.parquet"
                curr_file = f"{root_path}/data/features/{feature}/{feature}_{symbol}.parquet"
                dest_file = f"{root_path}/old_data/features/{feature}/{feature}_{symbol}.parquet"

            df1 = pd.read_parquet(prev_file)
            df2 = pd.read_parquet(curr_file)
            # feature_df = pd.concat([df1, df2]).drop_duplicates(keep="first").dropna().reset_index(drop=True)
            feature_df = pd.concat([df1, df2]) \
                  .drop_duplicates(subset="_time", keep="first") \
                  .dropna() \
                  .reset_index(drop=True)

            if os.path.exists(dest_file):
                df3 = pd.read_parquet(dest_file)
                if df2['_time'].iloc[-1] == df3['_time'].iloc[-1]:
                    break
            dest_dir = os.path.dirname(dest_file)
            os.makedirs(dest_dir, exist_ok=True)
            feature_df.to_parquet(dest_file)
    del df1, df2, feature_df
    gc.collect()
    
    # merge symbols to create dataset
    # history_columns_merge(config_general,general_mode=True)

    for symbol in list(config_general.keys()):

        # merge old & new datasets
        if symbol != 'BTCUSDT':
            logger.info(f"{symbol} - dataset")
            if "old_data" in prev_dir:
                dataset1_dir = f'{prev_dir}/dataset/{symbol}/dataset_all.parquet'
            else:
                dataset1_dir = f'{prev_dir}/dataset/{symbol}/dataset.parquet'
            
            dataset1 = pd.read_parquet(dataset1_dir)
            columns = dataset1.columns

            dataset2_dir = f"{root_path}/data/dataset/{symbol}/dataset.parquet"
            dataset2 = pd.read_parquet(dataset2_dir)
            dataset2 = dataset2[columns]

            # dataset_all = pd.concat([dataset1, dataset2]).drop_duplicates(keep="first").reset_index(drop=True)
            dataset_big = dataset2[:-250*288]
            dataset_small = dataset2[-250*288:].reset_index(drop=True)

            dest_file_all = f"{root_path}/old_data/dataset/{symbol}/dataset_all.parquet"
            dest_dir_all = os.path.dirname(dest_file_all)
            os.makedirs(dest_dir_all, exist_ok=True)
            dataset2.to_parquet(dest_file_all)

            dest_file_big = f"{root_path}/old_data/dataset/{symbol}/dataset_big.parquet"
            dest_dir_big = os.path.dirname(dest_file_big)
            os.makedirs(dest_dir_big, exist_ok=True)
            dataset_big.to_parquet(dest_file_big)
            
            dest_file_small = f"{root_path}/old_data/dataset/{symbol}/dataset_small.parquet"
            dest_dir_small = os.path.dirname(dest_file_small)
            os.makedirs(dest_dir_small, exist_ok=True)
            dataset_small.to_parquet(dest_file_small)
        

    






