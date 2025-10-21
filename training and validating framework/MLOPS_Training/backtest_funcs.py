import numpy as np
import pandas as pd
from configs.symbols_info import symbols_dict
import os


def calculate_classification_target_backtest(
    array,
    window_size,
    symbol_decimal_multiply: float = 1,
    take_profit_pct: float = 3,  # Percentage for take profit (e.g., 0.7 means 0.7%)
    stop_loss_pct: float = 1,  # Percentage for stop loss (e.g., 0.3 means 0.3%)
    mode: str = "long",
    trade_fee_pct: float = 0.5  # 0.5% trade fee on open price
):
    """
    This function returns two elements:
    Target: which has 3 different values. 1 means the position reaches the take profit price.
        -1 means the position ended in stoploss. 0 is in between.
    exit_price_diff is in pips.
    """
    swap_days_list = []
    target_list = []
    exit_price_diff_list = []
    n_step_ahead_close_trade_list = []

    if mode == "long":
        for i in range(array.shape[0] - window_size):
            selected_chunk = array[i : i + window_size]

            close_price = selected_chunk[0, 0]  # close price of the current trade

            # Apply the trade fee to close price
            fee_adjusted_close_price = close_price * (1 + trade_fee_pct / 100)

            # Calculate actual take profit and stop loss levels based on the close price
            # take_profit = close_price * (1 + take_profit_pct / 100)
            # stop_loss = close_price * (1 - stop_loss_pct / 100)

            price_diff_high = (selected_chunk[1:, 1] - close_price) / symbol_decimal_multiply
            price_diff_low = (selected_chunk[1:, 2] - close_price) / symbol_decimal_multiply

            # BUY CLASS
            buy_tp_cond = price_diff_high >= take_profit_pct*close_price/100
            buy_sl_cond = price_diff_low <= -stop_loss_pct*close_price/100

            if buy_tp_cond.any():
                arg_buy_tp_cond = np.where((buy_tp_cond))[0][0]
                if buy_sl_cond[0 : arg_buy_tp_cond + 1].any() == False:
                    swap_days = selected_chunk[1 : arg_buy_tp_cond + 1, 3].sum()
                    target = 1
                    exit_price_diff = take_profit_pct - trade_fee_pct
                    n_step_ahead_close_trade = arg_buy_tp_cond

                else:
                    arg_buy_sl_cond = np.where((buy_sl_cond))[0][0]
                    swap_days = selected_chunk[1 : arg_buy_sl_cond + 1, 3].sum()
                    target = -1
                    exit_price_diff = -stop_loss_pct - trade_fee_pct
                    n_step_ahead_close_trade = arg_buy_sl_cond

            elif buy_sl_cond.any():
                arg_buy_sl_cond = np.where((buy_sl_cond))[0][0]
                swap_days = selected_chunk[1 : arg_buy_sl_cond + 1, 3].sum()
                target = -1
                exit_price_diff = -stop_loss_pct - trade_fee_pct
                n_step_ahead_close_trade = arg_buy_sl_cond

            else:
                target = 0
                swap_days = selected_chunk[1:, 3].sum()
                exit_price_diff = ((selected_chunk[-1, 0] - close_price)*100 / (symbol_decimal_multiply*close_price)) - trade_fee_pct
                n_step_ahead_close_trade = window_size

            target_list.append(target)
            swap_days_list.append(swap_days)
            exit_price_diff_list.append(exit_price_diff)
            n_step_ahead_close_trade_list.append(n_step_ahead_close_trade)
    elif mode == "short":
        for i in range(array.shape[0] - window_size):
            selected_chunk = array[i : i + window_size]

            close_price = selected_chunk[0, 0]  # close price of the current trade

            # Apply the trade fee to close price
            fee_adjusted_close_price = close_price * (1 + trade_fee_pct / 100)

            # Calculate actual take profit and stop loss levels based on the close price
            # take_profit = close_price * (1 - take_profit_pct / 100)  # reverse for short
            # stop_loss = close_price * (1 + stop_loss_pct / 100)  # reverse for short

            price_diff_high = (selected_chunk[1:, 1] - close_price) / symbol_decimal_multiply
            price_diff_low = (selected_chunk[1:, 2] - close_price) / symbol_decimal_multiply

            # SELL CLASS
            target = 0
            sell_tp_cond = price_diff_low <= -take_profit_pct*close_price/100
            sell_sl_cond = price_diff_high >= stop_loss_pct*close_price/100

            if sell_tp_cond.any():
                arg_sell_tp_cond = np.where(sell_tp_cond)[0][0]
                if sell_sl_cond[0 : arg_sell_tp_cond + 1].any() == False:
                    swap_days = selected_chunk[1 : arg_sell_tp_cond + 1, 3].sum()
                    target = 1
                    exit_price_diff = take_profit_pct - trade_fee_pct
                    n_step_ahead_close_trade = arg_sell_tp_cond

                else:
                    arg_sell_sl_cond = np.where(sell_sl_cond)[0][0]
                    swap_days = selected_chunk[1 : arg_sell_sl_cond + 1, 3].sum()
                    target = -1
                    exit_price_diff = -stop_loss_pct - trade_fee_pct
                    n_step_ahead_close_trade = arg_sell_sl_cond

            elif sell_sl_cond.any():
                arg_sell_sl_cond = np.where(sell_sl_cond)[0][0]
                swap_days = selected_chunk[1 : arg_sell_sl_cond + 1, 3].sum()
                target = -1
                exit_price_diff = -stop_loss_pct - trade_fee_pct
                n_step_ahead_close_trade = arg_sell_sl_cond

            else:
                target = 0
                swap_days = selected_chunk[1:, 3].sum()
                exit_price_diff = -((selected_chunk[-1, 0] - close_price)*100 / (symbol_decimal_multiply*close_price)) - trade_fee_pct
                n_step_ahead_close_trade = window_size

            target_list.append(target)
            swap_days_list.append(swap_days)
            exit_price_diff_list.append(exit_price_diff)
            n_step_ahead_close_trade_list.append(n_step_ahead_close_trade)
    # Append None for the first 'window_size' entries to match array size
    for _ in range(window_size):
        swap_days_list.append(None)
        target_list.append(None)
        exit_price_diff_list.append(None)
        n_step_ahead_close_trade_list.append(None)

    return target_list, exit_price_diff_list, swap_days_list, n_step_ahead_close_trade_list

def calculate_max_drawdown(balance_series):
    """
    Calculate the maximum drawdown from a balance column in a pandas DataFrame.

    Args:
        df (pandas.DataFrame): Input DataFrame containing the balance column.
        balance_col (str): Name of the column containing the balance values.

    Returns:
        float: Maximum drawdown value.
    """
    # Get the cumulative maximum balance up to each point in time
    cum_max = balance_series.cummax()

    # Calculate the drawdown at each point in time
    drawdowns = (balance_series - cum_max) / cum_max

    # Return the maximum drawdown
    return drawdowns.min() * 100

def cal_backtest_on_raw_cndl(
    df_raw_path: str,
    target_symbol: str,
    look_ahead: int,
    take_profit_pct: int,
    stop_loss_pct: int,
    trade_mode: str,
    trade_fee_pct: float=0.5,
    )-> pd.DataFrame:
    """
    This function is basicaly a pre-backtest fucntion that calculates Backtest on all raw data (all times) based on strategy. 
    This function assumes we trade on each and every time step and calculates the backtest result for each time.
    The result can be merged with actual model signals to reach final backtest 
    """

    base_time_frame = 5
    window_size = int(look_ahead // base_time_frame)
    bt_column_name = (
    f"trg_clf_{trade_mode}_{target_symbol}_M{look_ahead}_TP{take_profit_pct}_SL{stop_loss_pct}_fee{trade_fee_pct}"
    )

    df_raw_backtest = pd.read_parquet(f'{df_raw_path}/{target_symbol}_stage_one.parquet',columns=["_time","open","high","low","close"])
    df_raw_backtest.columns = [
        "_time",
        f"{target_symbol}_M5_OPEN",
        f"{target_symbol}_M5_HIGH",
        f"{target_symbol}_M5_LOW",
        f"{target_symbol}_M5_CLOSE",  
    ]

    df_raw_backtest.sort_values("_time", inplace=True)
    df_raw_backtest['days_diff'] = (df_raw_backtest['_time'].dt.date - df_raw_backtest['_time'].dt.date.shift()).bfill().dt.days
    array = df_raw_backtest[
        [f"{target_symbol}_M5_CLOSE", f"{target_symbol}_M5_HIGH", f"{target_symbol}_M5_LOW", "days_diff"]
    ].to_numpy()

    df_raw_backtest[bt_column_name], df_raw_backtest["pct_diff"], df_raw_backtest["swap_days"], df_raw_backtest["n_step_ahead_close_trade"]\
          = calculate_classification_target_backtest(
        array,
        window_size,
        symbol_decimal_multiply=symbols_dict[target_symbol]["pip_size"],
        take_profit_pct=take_profit_pct,
        stop_loss_pct=stop_loss_pct,
        mode=trade_mode,
        trade_fee_pct=trade_fee_pct
    )
    df_raw_backtest.dropna(inplace=True)
    df_raw_backtest.to_csv(os.getcwd()+ '/df_raw_backtest.csv')

    return df_raw_backtest, bt_column_name

def del_redundant_trades(df) :

    df['_time'] = pd.to_datetime(df['_time'])

    # Compute the threshold times
    df['th'] = df['_time'] + pd.to_timedelta(df["n_step_ahead_close_trade"] * 5, unit='m')

    # Sort the DataFrame by _time for efficient filtering
    df = df.sort_values(by='_time').reset_index(drop=True)

    # Efficient filtering using NumPy
    import numpy as np

    mask = np.ones(len(df), dtype=bool)
    for i in range(len(df) - 1):
        if mask[i]:  # Process only if not previously removed
            th = df.loc[i, 'th']
            mask[i + 1:] &= df.loc[i + 1:, '_time'] > th

    # Apply the mask to keep only the valid rows
    return df[mask].drop(columns=['th']).reset_index(drop=True)


def limit_parallel_trades(df, max_trades):
    # df = df.copy()
    df["take_trade"] = 0  # Default: don't take trade
    active_trades_count = []  # New: track number of active trades per row

    open_trades = []

    for i in range(len(df)):
        # Remove expired trades
        open_trades = [end for end in open_trades if end > i]

        # Record number of currently active trades
        active_trades_count.append(len(open_trades))

        # Check if we can take a new trade
        if df.loc[i, "model_prediction"] == 1 and len(open_trades) < max_trades:
            df.at[i, "take_trade"] = 1
            duration = int(df.loc[i, "n_step_ahead_close_trade"])
            open_trades.append(i + duration)

    df["active_trades"] = active_trades_count  # Assign the new column
    df = df[df["take_trade"]==1]
    df = df.drop("take_trade", axis=1)
    return df.reset_index()




def do_backtest(
    df_model_signal: pd.DataFrame,
    volume: float,
    initial_balance: int,
    df_raw_backtest: pd.DataFrame,
    bt_column_name: str,
    swap_rate: float,
    trade_fee_pct: float,  # Include the trade fee percentage
    leverage = 1
):
    # print(df_model_signal)
    df_model_signal.to_csv(os.getcwd()+'/model_signal1.csv')
    # print(f'coluimns:{df_model_signal.columns} ..../////....////')

    new_trg_df = df_model_signal.merge(df_raw_backtest, on="_time", how="inner")
    new_trg_df.to_csv(os.getcwd()+'/new_trg_df0.5.csv')
    if new_trg_df.shape[0] != 0:
        max_trades = 20
        new_trg_df = limit_parallel_trades(new_trg_df, max_trades)
        new_trg_df.to_csv(os.getcwd()+'/new_trg_df1.csv')
        
        begin_date = df_model_signal.index[0]
        end_date = df_model_signal.index[-1]

        df_total = df_raw_backtest[
            (df_raw_backtest["_time"] >= begin_date) &
            (df_raw_backtest["_time"] <= end_date)
        ].reset_index()

        new_trg_df = pd.merge(df_total, new_trg_df[['_time', 'model_prediction', 'active_trades']], on='_time', how='left')

        new_trg_df["balance"] = initial_balance
        new_trg_df["net_profit"] = 0
        new_trg_df["balance"] = new_trg_df["balance"].astype(float)
        new_trg_df["net_profit"] = new_trg_df["net_profit"].astype(float)
        new_trg_df.to_csv(os.getcwd()+'/new_trg_df1.5.csv')
        index_last = new_trg_df[new_trg_df['model_prediction'] == 1].index[-1]
        print(index_last)

        for i in range( len(new_trg_df)):
            if new_trg_df.loc[i, "model_prediction"] == 1:
                if i<1:
                    take_balance = 1/max_trades
                    new_trg_df.loc[i, "net_profit"] = (new_trg_df.loc[i, "pct_diff"] * initial_balance / 100) * take_balance * leverage
                    step_ahead = new_trg_df.loc[i, "n_step_ahead_close_trade"]
                    index = int(i + step_ahead)

                    if index >= index_last :
                        # pass
                        new_trg_df.loc[index_last:, "balance"] += new_trg_df.loc[i, "net_profit"]

                    else :
                        new_trg_df.loc[index:, "balance"] += new_trg_df.loc[i, "net_profit"]
                else:
                    take_balance = 1/max_trades
                    new_trg_df.loc[i, "net_profit"] = (new_trg_df.loc[i, "pct_diff"] * new_trg_df.loc[i-1, "balance"] / 100) * take_balance * leverage
                    step_ahead = new_trg_df.loc[i, "n_step_ahead_close_trade"]
                    index = int(i + step_ahead)
                    if index >= index_last :
                        # pass
                        new_trg_df.loc[index_last:, "balance"] += new_trg_df.loc[i, "net_profit"]
                    else :
                        new_trg_df.loc[index:, "balance"] += new_trg_df.loc[i, "net_profit"]
        new_trg_df = new_trg_df[new_trg_df["model_prediction"] == 1]
        # # print(new_trg_df)
        # Calculate max drawdown
        max_drawdown = calculate_max_drawdown(new_trg_df["balance"])
        new_trg_df.to_csv(os.getcwd()+'/new_trg_df2.csv')

    else:

        new_trg_df["net_profit"] = 0
        new_trg_df["balance"] = initial_balance

    # Prepare the backtest report
    if new_trg_df.shape[0] == 0:
        backtest_report = {
            "balance_cash": initial_balance,
            "profit_pips": 0,
            "max_draw_down": 0,
            "profit_percent": 0,
            "n_trades" : 0
        }
    else:
        backtest_report = {
            "balance_cash": int(new_trg_df.iloc[-1]["balance"]),
            "profit_pips": int(new_trg_df["net_profit"].sum()),
            "max_draw_down": round(max_drawdown, 2),
            "profit_percent": round(
                ((new_trg_df.iloc[-1]["balance"] - initial_balance) / initial_balance) * 100,
                2,
            ),
            "n_trades" : new_trg_df.shape[0]
        }

    return (
        backtest_report,
        new_trg_df[
            [
                "_time",
                "model_prediction",
                f"{bt_column_name}",
                "pct_diff",
                "net_profit",
                "balance",
            ]
        ],
    )
