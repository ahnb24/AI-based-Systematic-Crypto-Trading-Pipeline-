import numpy as np


def calculate_classification_target_numpy_ver(
    array,
    window_size,
    symbol_decimal_multiply: float = 0.0001,
    take_profit_pct: float = 5,
    stop_loss_pct: float = 2,
    mode: str = "long",
    # trade_fee_pct: float = 0.5
):
    target_list = []

    if mode == "long":
        for i in range(array.shape[0] - window_size):
            selected_chunk = array[i : i + window_size]

            price_diff_high = (
                selected_chunk[1:, 1] - selected_chunk[0, 0]
            ) / symbol_decimal_multiply
            price_diff_low = (
                selected_chunk[1:, 2] - selected_chunk[0, 0]
            ) / symbol_decimal_multiply

            # BUY CLASS
            target = 0

            buy_tp_cond = price_diff_high >= take_profit_pct*selected_chunk[0, 0]/100
            buy_sl_cond = price_diff_low <= -stop_loss_pct*selected_chunk[0, 0]/100

            if buy_tp_cond.any() == True:
                arg_buy_tp_cond = np.where(buy_tp_cond)[0][0]
                if buy_sl_cond[0 : arg_buy_tp_cond + 1].any() == False:
                    target = 1

            target_list.append(target)

    elif mode == "short":
        for i in range(array.shape[0] - window_size):
            selected_chunk = array[i : i + window_size]

            price_diff_high = (
                selected_chunk[1:, 1] - selected_chunk[0, 0]
            ) / symbol_decimal_multiply
            price_diff_low = (
                selected_chunk[1:, 2] - selected_chunk[0, 0]
            ) / symbol_decimal_multiply

            # BUY CLASS
            target = 0
            sell_tp_cond = price_diff_low <= -take_profit_pct*selected_chunk[0, 0]/100
            sell_sl_cond = price_diff_high >= stop_loss_pct*selected_chunk[0, 0]/100

            if sell_tp_cond.any() == True:
                arg_sell_tp_cond = np.where(sell_tp_cond)[0][0]
                if sell_sl_cond[0 : arg_sell_tp_cond + 1].any() == False:
                    target = 1

            target_list.append(target)

    for _ in range(window_size):
        target_list.append(None)

    return target_list


