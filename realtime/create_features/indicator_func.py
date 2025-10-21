import polars as pl
from pathlib import Path
from configs.history_data_crawlers_config import symbols_dict
import os, glob
from typing import Callable, Dict, List, Tuple, Union
import re




# ?? indicator ---------------------------------------------------


def cal_RSI_base_func(
    df: pl.DataFrame,
    w: int,
    time_frame: int,
    features: List[str],
    pip_size: float,  # only for compatibility
    prefix: str = "fe_RSI",
    percentage_feature: bool = False,
    add_30_70: bool = True,
) -> pl.DataFrame:
    """
    This function creates RSI feature
    inputs:
    df: dataframe containing the raw feature
    w: window size
    time_frame: time_frame for calculations
    feature: raw feature on which the RSI is calculated
    prefix: prefix of feature name
    percentage_feature: true for percentage features like price-percentage are diff features by nature
    add_30_70: add whether the RSI is above 70 or below 30 !

    To understand the code see the RSI formula
    https://www.wallstreetmojo.com/relative-strength-index/
    pandas version: https://github.com/twopirllc/pandas-ta/blob/main/pandas_ta/momentum/rsi.py

    """
    assert (
        len(features) == 1
    ), f"Only 1 feature should have been passed but {len(features)} received!"
    feature = features[0]

    df = df.sort("_time")
    if percentage_feature:
        # percentage features like price-percentage are diff features by nature
        df = df.with_columns((pl.col(feature)).alias(f"{feature}_diff")).lazy()
    else:
        df = df.with_columns((pl.col(feature).diff()).alias(f"{feature}_diff")).lazy()

    df = df.with_columns(
        ((pl.col(f"{feature}_diff") >= 0) * (pl.col(f"{feature}_diff"))).alias(
            f"{feature}_GAIN"
        )
    ).lazy()
    df = df.with_columns(
        ((pl.col(f"{feature}_diff") < 0) * -1 * (pl.col(f"{feature}_diff"))).alias(
            f"{feature}_LOSS"
        )
    ).lazy()

    df = df.with_columns(
        (
            pl.col(f"{feature}_GAIN").ewm_mean(
                alpha=1.0 / w, min_periods=w, ignore_nulls=True
            )
        ).alias(f"{feature}_Avg_GAIN_{w}")
    ).lazy()
    df = df.with_columns(
        (
            pl.col(f"{feature}_LOSS").ewm_mean(
                alpha=1.0 / w, min_periods=w, ignore_nulls=True
            )
        ).alias(f"{feature}_Avg_LOSS_{w}")
    ).lazy()

    # METHOD I
    df = df.with_columns(
        (
            (pl.col(f"{feature}_Avg_GAIN_{w}")) / ((pl.col(f"{feature}_Avg_LOSS_{w}")))
        ).alias(f"{feature}_RS_{w}")
    ).lazy()
    df = df.with_columns(
        (100 - (100 / (1 + pl.col(f"{feature}_RS_{w}")))).alias(
            f"{prefix}_{feature}_W{w}_cndl_M{time_frame}"
        )
    ).lazy()

    if add_30_70:
        df = df.with_columns(
            ((pl.col(f"{prefix}_{feature}_W{w}_cndl_M{time_frame}")) >= 70).alias(
                f"{prefix}_{feature}_W{w}_gte_70_cndl_M{time_frame}"
            )
        ).lazy()
        df = df.with_columns(
            ((pl.col(f"{prefix}_{feature}_W{w}_cndl_M{time_frame}")) <= 30).alias(
                f"{prefix}_{feature}_W{w}_lte_30_cndl_M{time_frame}"
            )
        ).lazy()

    df = df.drop(
        [
            f"{feature}",
            f"{feature}_diff",
            f"{feature}_GAIN",
            f"{feature}_LOSS",
            f"{feature}_Avg_GAIN_{w}",
            f"{feature}_Avg_LOSS_{w}",
            f"{feature}_RS_{w}",
        ],
    )
    return df.collect()


def cal_EMA_base_func(
    df: pl.DataFrame,
    w: int,
    time_frame: int,
    features: List[str],
    pip_size: float,
    prefix: str = "fe_EMA",
    normalize: bool = True,
) -> pl.DataFrame:
    """
    this function calculates exponantial moving average.
    inputs:
    df: dataframe containing the raw feature
    w: window size
    time_frame: time_frame for calculations
    feature: raw feature on which the RSI is calculated
    pip size: pip size of the pair
    prefix: prefix of feature name
    normalize: if True the function returns pipsize difference between EMA and last close price.

    """
    assert (
        len(features) == 1
    ), f"Only 1 feature should have been passed but {len(features)} received!"
    feature = features[0]

    df = df.sort("_time")

    if normalize:
        df = df.with_columns(
            (
                (
                    (pl.col(feature).ewm_mean(span=w, ignore_nulls=True))
                    - pl.col(feature)
                )*100
                / pl.col(feature)
            ).alias(f"{prefix}_{feature}_W{w}_cndl_M{time_frame}_norm")
        ).lazy()
    else:
        df = df.with_columns(
            (pl.col(feature).ewm_mean(span=w, ignore_nulls=True)).alias(
                f"{prefix}_{feature}_W{w}_cndl_M{time_frame}"
            )
        ).lazy()

    df = df.collect()

    df = df.drop([f"{feature}"])

    return df


def cal_SMA_base_func(
    df: pl.DataFrame,
    w: int,
    time_frame: int,
    features: List[str],
    pip_size: float,
    prefix: str = "fe_SMA",
    normalize: bool = True,
) -> pl.DataFrame:
    """
    this function calculates simple moving average.
    inputs:
    df: dataframe containing the raw feature
    w: window size
    time_frame: time_frame for calculations
    feature: raw feature on which the RSI is calculated
    pip size: pip size of the pair
    prefix: prefix of feature name
    normalize: if True the function returns pipsize difference between EMA and last close price.
    """
    assert (
        len(features) == 1
    ), f"Only 1 feature should have been passed but {len(features)} received!"
    feature = features[0]

    df = df.sort("_time")
    if normalize:
        df = df.with_columns(
            (
                ((pl.col(feature).rolling_mean(window_size=w)) - pl.col(feature))*100
                /  pl.col(feature)
            ).alias(f"{prefix}_{feature}_W{w}_cndl_M{time_frame}_norm")
        ).lazy()

    else:
        df = df.with_columns(
            (pl.col(feature).rolling_mean(window_size=w)).alias(
                f"{prefix}_{feature}_W{w}_cndl_M{time_frame}"
            )
        ).lazy()
    df = df.collect()
    df = df.drop([f"{feature}"])

    return df


def add_candle_base_indicators_polars(
    df_base: pl.DataFrame,
    prefix: str,
    base_func: Callable[..., pl.DataFrame],
    opts: Dict[str, Union[str, List[int]]],
) -> None:
    """
    this function takes an indicator function, apply it and save the resulting parquet
    inputs:
    df_base: base dataframe containing the raw features
    prefix: prefix of feature name
    base_func: the indicator function
    opts: a dictionary of "symbol","base_feature","candle_timeframe","window_size" and "features_folder_path"
    """

    df_base = df_base.sort("_time")
    symbol = opts["symbol"]
    pip_size = symbols_dict[symbol]["pip_size"]
    features_folder_path = opts["features_folder_path"] + "/unmerged/"
    Path(features_folder_path).mkdir(parents=True, exist_ok=True)

    filelist = glob.glob(f"{features_folder_path}/*.parquet", recursive=True)
    for f in filelist:
        os.remove(f)

    features = opts["base_feature"]
    time_frames = opts["candle_timeframe"]
    window_sizes = opts["window_size"]

    for w in window_sizes:
        for time_frame in time_frames:
            df = df_base.filter(
                pl.col("minutesPassed") % time_frame == (time_frame - 5)
            )

            # Create a regex pattern to match 'M' followed by the time_frame number
            pattern = re.compile(rf"M{time_frame}_")

            # Find items where the number after 'M' is not equal to time_frame
            other_tf_features = [f for f in features if not pattern.match(f)]
            df = df.drop(other_tf_features + ["minutesPassed"])
            df = base_func(
                df=df,
                w=w,
                time_frame=time_frame,
                features=list(set(features) - set(other_tf_features)),
                pip_size=pip_size,
                prefix=prefix,
            )

            file_name = (
                features_folder_path + f"/{prefix}_{w}_{symbol}_M{time_frame}.parquet"
            )
            df.write_parquet(file_name)

    return

# ?? ratio  -----------------------------------------------------
def add_ratio_by_columns(
    df: pl.DataFrame, col_name_a: str, col_name_b: str, ratio_col_name
) -> pl.DataFrame:
    """
    this function calculates the ratio of two features
    inputs:
    df: dataframe containing the raw feature
    col_name_a: name of the first feature
    col_name_b: name of the second feature
    ratio_col_name: name of the ratio feature
    """
    df = df.with_columns(
        pl.when(pl.col(col_name_b) == 0)
        .then(0)  # or then(custom_value)
        .otherwise((pl.col(col_name_a) / pl.col(col_name_b)))
        .round(5)
        .alias(ratio_col_name)
    )
    return df


def add_ratio(
    df: pl.DataFrame,
    symbol: str,
    fe_name: str,
    timeframe: int,
    w1: int,
    w2: int,
    fe_prefix: str = "fe_ratio",
) -> pl.DataFrame:
    """
    this function takes whatever needed for defining ratio and then applies add_ratio_by_columns
    """

    if "RSI" in fe_name or "RSTD" in fe_name:
        col_a = f"fe_{fe_name}_M{timeframe}_CLOSE_W{w1}_cndl_M{timeframe}"
        col_b = f"fe_{fe_name}_M{timeframe}_CLOSE_W{w2}_cndl_M{timeframe}"
    elif "ATR" in fe_name:
        col_a = f"fe_{fe_name}_M{timeframe}_W{w1}_cndl_M{timeframe}_norm"
        col_b = f"fe_{fe_name}_M{timeframe}_W{w2}_cndl_M{timeframe}_norm"
    else:
        col_a = f"fe_{fe_name}_M{timeframe}_CLOSE_W{w1}_cndl_M{timeframe}_norm"
        col_b = f"fe_{fe_name}_M{timeframe}_CLOSE_W{w2}_cndl_M{timeframe}_norm"

    if col_a not in df.columns or col_b not in df.columns:
        print(f"!!! {col_a} not in df.columns or {col_b} not in df.columns.")
        return df

    ratio_col_name = (
        f"{fe_prefix}_{fe_name}_M{timeframe}_CLOSE_W{w1}_W{w2}_cndl_M{timeframe}"
    )

    df = add_ratio_by_columns(df, col_a, col_b, ratio_col_name)

    return df


def add_all_ratio_by_config(
    df: pl.DataFrame,
    symbol: str,
    fe_name: str,
    ratio_config: Dict[str, Dict[str, Union[List[int], List[Tuple[int, int]]]]],
    fe_prefix: str = "fe_ratio",
) -> pl.DataFrame:
    """
    this function takes the ratio config and applies add_ratio
    ratio_config: a dictionary of dictionaries containing list of time frames and list of pairs of window sizes needed for ratio
    """

    base_cols = set(df.columns) - set(["_time"])
    for timeframe in ratio_config["timeframe"]:
        for w_set in ratio_config["window_size"]:
            df = add_ratio(
                df, symbol, fe_name, timeframe, w_set[0], w_set[1], fe_prefix
            )

    return df.drop(base_cols)


# ?? volatility
def cal_ATR_func(
    df: pl.DataFrame,
    w: int,
    time_frame: int,
    features: List[str],
    pip_size: float,
    prefix: str = "fe_ATR",
    normalize: bool = True,
) -> pl.DataFrame:
    """
    this function calculates ATR indicator.
    inputs:
    df: dataframe containing the raw feature
    w: window size
    time_frame: time_frame for calculations
    feature: raw feature on which the RSI is calculated
    pip size: pip size of the pair
    prefix: prefix of feature name
    normalize: if True the function returns pipsize difference between EMA and last close price.

    """
    assert (
        len(features) == 3
    ), f"Only 3 feature should have been passed but {len(features)} received!"
    features = sorted(features)

    df = df.sort("_time")

    if normalize:
        df = df.with_columns(
            (
                (
                    (
                        (pl.col(features[1]) - pl.col(features[2])).rolling_mean(
                            window_size=w
                        )
                    )
                    / pl.col(features[0])
                )
                / 0.01
            ).alias(
                f"{prefix}_{features[0].replace('CLOSE','')}W{w}_cndl_M{time_frame}_norm"
            )
        ).lazy()
    else:
        df = df.with_columns(
            (
                (
                    (pl.col(features[1]) - pl.col(features[2])).rolling_mean(
                        window_size=w
                    )
                )
                / pip_size
            ).alias(
                f"{prefix}_{features[0].replace('CLOSE','')}W{w}_cndl_M{time_frame}"
            )
        ).lazy()

    df = df.collect()

    df = df.drop(features)

    return df


def cal_RSTD_func(
    df: pl.DataFrame,
    w: int,
    time_frame: int,
    features: List[str],
    pip_size: float,
    prefix: str = "fe_RSTD",
    normalize: bool = True,
) -> pl.DataFrame:
    """
    this function calculates Standard Deviation of Return.
    inputs:
    df: dataframe containing the raw feature
    w: window size
    time_frame: time_frame for calculations
    feature: raw feature on which the RSI is calculated
    pip size: pip size of the pair
    prefix: prefix of feature name
    normalize: if True the function returns pipsize difference between EMA and last close price.

    """
    assert (
        len(features) == 1
    ), f"Only 1 feature should have been passed but {len(features)} received!"
    feature = features[0]

    df = df.sort("_time")
    if normalize:
        df = df.with_columns(
            (
                (
                    (
                        (
                            pl.col(feature).log() - pl.col(feature).shift(1).log()
                        ).rolling_std(window_size=w)
                    )
                    / pl.col(feature)
                )
                / 0.01
            ).alias(f"{prefix}_{feature}_W{w}_cndl_M{time_frame}")
        ).lazy()

    else:
        df = df.with_columns(
            (
                (
                    (
                        pl.col(feature).log() - pl.col(feature).shift(1).log()
                    ).rolling_std(window_size=w)
                )
                / pip_size
            ).alias(f"{prefix}_{feature}_W{w}_cndl_M{time_frame}")
        ).lazy()
    df = df.collect()
    df = df.drop([f"{feature}"])

    return df
