
import pandas as pd
from utils.clean_data import remove_weekends
from utils.datetime_utils import drop_first_day_pandas
from utils.df_utils import ffill_df_to_true_time_steps
from utils.logging_tools import default_logger
from configs.stage_one_data_config import stage_one_data_path
from configs.history_data_crawlers_config import (
  data_folder,
  symbols_dict,
)
import numpy as np
def history_data_stage_one(feature_config, logger=default_logger):
    """
    get data from data_sources[0] first and fill emoty times with other data_sources in order.

    """
    logger.info("= " * 25)
    logger.info("--> start history_data_stage_one func:")
    # data_sources = ["metatrader", "dukascopy"]
    # data_sources = ["binance"]

    for symbol in list(feature_config.keys()):
        logger.info("-" * 25)
        logger.info(f"--> symbol:{symbol}")

        # main_data_source = data_sources[0]
        file_name = f"{data_folder}/{symbol}/{symbol}_klines.parquet"
        columns = ["_time", "open", "high", "low", "close", "tick_volume"]
        df = pd.read_parquet(
            file_name,
            columns=columns,
        )
        # Note: metatrader data naturaly does not have weekends. if the main data source is not metatrader:
        # df = remove_weekends(df, weekends_day=["Saturday", "Sunday"], convert_tz=False)

        # for data_source in data_sources[1:]:
        #     file_name = f"{data_folder}/{data_source}/{symbol}_{data_source}.parquet"
        #     df_temp = pd.read_parquet(file_name,columns=columns)

        #     # Note: only dukascopy that returns data in UTC timezone can be converted using this method. 
        #     df_temp["_time"] = (
        #         df_temp["_time"].dt.tz_localize(None).dt.tz_localize("UTC").dt.tz_convert("US/Eastern")
        #         + pd.offsets.Hour(7)
        #     ).dt.tz_localize(None)

        #     df_temp = remove_weekends(
        #         df_temp, weekends_day=["Saturday", "Sunday"], convert_tz=False
        #     )

        #     main_times = df["_time"]
        #     df = (
        #     pd.concat(
        #         [
        #             df,
        #             df_temp[(~df_temp["_time"].isin(main_times))],
        #         ]
        #     )
        #     .sort_values("_time")
        #     .reset_index(drop=True)
        #     .drop_duplicates("_time")
        #     )

        
        logger.info(f"--> number of nulls before forward fill: {df.isnull().sum().sum()}")
        ##? fill time gaps for trade days in dataset.
        df = ffill_df_to_true_time_steps(df)
        n_nulls=df.isnull().sum().sum()
        # ? check for null and inf:
        # assert n_nulls == 0, f"DataFrame contains null values after ffill. {n_nulls}"

        # ? check for big time gap
        time_diffs = df["_time"].diff().iloc[1:]
        assert time_diffs.between(
            pd.Timedelta("5min"),
            pd.Timedelta(days=100),
        ).all(), "Gaps detected in timestamps"

        ##? delete first incomplete day of the data.
        df = drop_first_day_pandas(df)
        # df["data_source"] = df["data_source"].astype("string")

        # ? Check order of OHLC makes sense
        assert (df["open"] <= df["high"]).all(), "Open higher than high"
        assert (df["open"] >= df["low"]).all(), "Open lower than low"
        assert (df["high"] >= df["low"]).all(), "High lower than low"

        # ? Check for outliers in returns
        returns = df["close"].pct_change().iloc[1:]
        assert returns.between(-0.35, 0.35).all(), "pct_change outlier returns detected"

        df[["open", "high", "low", "close", "tick_volume"]] = df[
            ["open", "high", "low", "close", "tick_volume"]
        ].astype(float)

        
        # mask = df['tick_volume'] == 0
        # df.loc[mask, 'tick_volume'] = np.nan  # Set zeros to NaN
        # df['tick_volume'] = df['tick_volume'].ffill()  # Forward fill

        # # Apply random adjustment only to previously zero values
        # df.loc[mask, 'tick_volume'] *= np.random.uniform(0.8, 1.2, size=mask.sum())

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
        
        logger.info(f'--> min time: {df["_time"].min()}, max time: {df["_time"].max()}')
        stage_one_file_name = f"{stage_one_data_path}/{symbol}_stage_one.parquet"
        df.to_parquet(stage_one_file_name, index=False)
        logger.info(f"--> parquet saved. {stage_one_file_name}") 

    logger.info("--> history_data_stage_one run successfully.")



if __name__ == "__main__":
    from utils.config_utils import read_feature_config
    from configs.feature_configs_general import generate_general_config
    config_general = generate_general_config()
    history_data_stage_one(config_general)
    default_logger.info(f"--> history_data_stage_one DONE.")
