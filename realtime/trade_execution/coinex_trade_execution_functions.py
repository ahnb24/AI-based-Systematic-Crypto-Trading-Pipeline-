import hashlib
import json
import time
import hmac
from urllib.parse import urlparse, urlencode
import random
from datetime import datetime, timedelta, timezone
import requests
import time
import pandas as pd
from realtime.realtime_utils import (
    get_coinex_time_now,
)
import pickle
import os


class CoinexRequestsClient(object):
    HEADERS = {
        "Content-Type": "application/json; charset=utf-8",
        "Accept": "application/json",
        "X-COINEX-KEY": "",
        "X-COINEX-SIGN": "",
        "X-COINEX-TIMESTAMP": "",
    }

    def __init__(self, access_id, secret_key):
        self.access_id = access_id
        self.secret_key = secret_key
        self.url = "https://api.coinex.com/v2"
        self.headers = self.HEADERS.copy()

    # Generate your signature string
    def gen_sign(self, method, request_path, body, timestamp):
        prepared_str = f"{method}{request_path}{body}{timestamp}"
        signature = hmac.new(
            bytes(self.secret_key, 'latin-1'), 
            msg=bytes(prepared_str, 'latin-1'), 
            digestmod=hashlib.sha256
        ).hexdigest().lower()
        return signature

    def get_common_headers(self, signed_str, timestamp):
        headers = self.HEADERS.copy()
        headers["X-COINEX-KEY"] = self.access_id
        headers["X-COINEX-SIGN"] = signed_str
        headers["X-COINEX-TIMESTAMP"] = timestamp
        headers["Content-Type"] = "application/json; charset=utf-8"
        return headers

    def request(self, method, url, params={}, data=""):
        req = urlparse(url)
        request_path = req.path

        timestamp = str(int(time.time() * 1000))
        if method.upper() == "GET":
            # If params exist, query string needs to be added to the request path
            if params:
                for item in params:
                    if params[item] is None:
                        del params[item]
                        continue
                request_path = request_path + "?" + urlencode(params)

            signed_str = self.gen_sign(
                method, request_path, body="", timestamp=timestamp
            )
            response = requests.get(
                url,
                params=params,
                headers=self.get_common_headers(signed_str, timestamp),
            )

        else:
            signed_str = self.gen_sign(
                method, request_path, body=data, timestamp=timestamp
            )
            response = requests.post(
                url, data, headers=self.get_common_headers(signed_str, timestamp)
            )

        if response.status_code != 200:
            raise ValueError(response.text)
        return response

def get_value(coinex_request, symbol):
    balance = get_spot_balance(coinex_request)
    ccy_list = []
    count_symbol = 0
    for i in balance:
        ccy_list.append(i["ccy"])
    # print(f"balance: {balance}")
    # print(f"ccy_list: {ccy_list}")
    if symbol[:-4] in ccy_list:
        available_symbol = float([item for item in balance if item.get('ccy') == symbol[:-4]][0]['available'])
        frozen_symbol = float([item for item in balance if item.get('ccy') == symbol[:-4]][0]['frozen'])
        count_symbol = available_symbol + frozen_symbol
    available_usdt = float([item for item in balance if item.get('ccy') == "USDT"][0]['available'])
    frozen_usdt = float([item for item in balance if item.get('ccy') == "USDT"][0]['frozen'])
    usdt = available_usdt + frozen_usdt
    price = float(get_symbol_price(coinex_request, symbol))
    value = count_symbol * price + usdt
    return value, usdt

def get_spot_balance(coinex_request):
    request_path = "/assets/spot/balance"
    response = coinex_request.request(
        "GET",
        "{url}{request_path}".format(url=coinex_request.url, request_path=request_path),
    )
    return response.json()['data']

def cancel_order(coinex_request, symbol, order_id, market_type="SPOT"):
    request_path = "/spot/cancel-order"
    data = {
        "market": symbol,
        "market_type": market_type,
        "order_id": order_id,
    }
    data = json.dumps(data)
    response = coinex_request.request(
        "POST",
        "{url}{request_path}".format(url=coinex_request.url, request_path=request_path),
        data=data,
        )
    return response.json()

def cancel_stop_order(coinex_request, symbol, stop_id):
    request_path = "/spot/cancel-stop-order"
    data = {
        "market": symbol,
        "market_type": "SPOT",
        "stop_id": stop_id,
    }
    data = json.dumps(data)
    response = coinex_request.request(
        "POST",
        "{url}{request_path}".format(url=coinex_request.url, request_path=request_path),
        data=data,
        )
    return response.json()

def cal_candle_time_now(time_frame:int=5):
  now = get_coinex_time_now().replace(tzinfo=None)
  return pd.Timestamp(now + (datetime.min - now) % timedelta(minutes=int(time_frame)) - timedelta(minutes=int(2*time_frame)))

# def modify_position(coinex_request, symbol, order_id, amount=0, price=0, market_type="SPOT"):
#     request_path = "/spot/modify-order"

#     if amount == 0 :
#         data = {
#             "market": symbol,
#             "market_type": market_type,
#             "order_id": order_id,
#             "price" : price
#         }
#     elif price == 0 :
#         data = {
#             "market": symbol,
#             "market_type": market_type,
#             "order_id": order_id,
#             "amount" : amount
#         }
#     else:
#         data = {
#             "market": symbol,
#             "market_type": market_type,
#             "order_id": order_id,
#             "amount" : amount,
#             "price" : price
#         }
#     data = json.dumps(data)
#     response = coinex_request.request(
#         "POST",
#         "{url}{request_path}".format(url=coinex_request.url, request_path=request_path),
#         data=data,
#         )
#     return response.json()

def get_open_orders(coinex_request, symbol, side="", market_type='SPOT'):
    request_path = "/spot/pending-order"
    if side != "" :
        params = {
            "market": symbol,
            "side": side,
            "market_type": market_type

        }
    else:
        params = {
            "market": symbol,
            "market_type": market_type
        }
    # params = json.dumps(params)
    response = coinex_request.request(
        "GET",
        "{url}{request_path}".format(url=coinex_request.url, request_path=request_path),
        params=params,
        )
    return response.json()

def get_open_stop_orders(coinex_request, symbol, side=""):
    request_path = "/spot/pending-stop-order"
    if side != "" :
        params = {
            "market": symbol,
            "side": side,
            "market_type": "SPOT"

        }
    else:
        params = {
            "market": symbol,
            "market_type": "SPOT"
        }
    # params = json.dumps(params)
    response = coinex_request.request(
        "GET",
        "{url}{request_path}".format(url=coinex_request.url, request_path=request_path),
        params=params,
        )
    return response.json()

def get_history_orders(coinex_request, symbol, side="buy", market_type="SPOT"):
    request_path = "/spot/finished-order"
    params = {
        "market": symbol,
        "market_type": market_type,
        "side": side
    }
    # params = json.dumps(params)
    response = coinex_request.request(
        "GET",
        "{url}{request_path}".format(url=coinex_request.url, request_path=request_path),
        params=params,
        )
    return response.json()

def get_symbol_price(coinex_request, symbol):
    request_path = "/spot/ticker"
    params = {
        "market": symbol,
    }
    # params = json.dumps(params)
    response = coinex_request.request(
        "GET",
        "{url}{request_path}".format(url=coinex_request.url, request_path=request_path),
        params=params,
        )
    return response.json()["data"][0]["last"]

# def get_open_positions_order_id(coinex_request, symbol):
#     pos_symbol = None
#     ccyy = None
#     balance = get_spot_balance(coinex_request)
#     for i in range(len(balance)-1):
#         ccy = balance[i]['ccy']
#         if ccy != "USDT":
#             ccyy = ccy
#             pos_symbol = f"{ccy}USDT"

#     if pos_symbol == None :
#         return {}

#     else:
#         available = float([item for item in balance if item.get('ccy') == ccyy][0]['available'])
#         price = float(get_symbol_price(coinex_request, symbol))
#         value = available * price
#         if pos_symbol==symbol and value > 1:
#             history_order = get_history_orders(coinex_request, symbol)
#             return history_order.json()['data'][0]['order_id']
#             # order_id = history_order['data'][0]['order_id']

def close_positions_by_id(coinex_request, symbol, order_id, all_trades):
    # orders = get_history_orders(coinex_request, symbol)["data"]
    # order = [item for item in orders if item.get('order_id') == order_id][0]
    amount = all_trades[order_id]["amount"]
    side = all_trades[order_id]["side"]
    if side == "buy":
        order_result = send_order(coinex_request, "close_pos", symbol, amount, "sell", "market" ,100000)
    elif side == "sell":
        order_result = send_order(coinex_request, "close_pos", symbol, amount, "buy", "market" ,100000)
    return order_result

def send_order(coinex_request, order_order, symbol, amount, side, type, price=1, market_type="SPOT"):
    request_path = "/spot/order"
    if order_order == "open_pos":
        if type=="limit":
            data = {
            "market": symbol,
            "market_type": market_type,
            "side": side,
            "type": type,
            "amount": amount,
            "price": price,
            "ccy": symbol[:-4],
            "is_hide": True
            }
        if type=="market":
            data = {
            "market": symbol,
            "market_type": market_type,
            "side": side,
            "type": type,
            "amount": amount,
            "ccy": symbol[:-4],
            "is_hide": True
            }
    elif order_order == "close_pos":
        if type=="limit":
            data = {
            "market": symbol,
            "market_type": market_type,
            "side": side,
            "type": type,
            "amount": amount,
            "price": price*0.999,
            "ccy": symbol[:-4],
            "is_hide": True,
            }
        if type=="market":
            data = {
            "market": symbol,
            "market_type": market_type,
            "side": side,
            "type": type,
            "amount": amount,
            "ccy": symbol[:-4],
            "is_hide": True,
            }

    data = json.dumps(data)
    response = coinex_request.request(
        "POST",
        "{url}{request_path}".format(url=coinex_request.url, request_path=request_path),
        data=data,
        )
    if order_order == "close_pos" and response.json()["code"] == 3109:
        balance = get_spot_balance(coinex_request)
        amount = float(next(item['available'] for item in balance if item['ccy'] == symbol[:-4]))
        if type=="limit":
            data = {
            "market": symbol,
            "market_type": market_type,
            "side": side,
            "type": type,
            "amount": amount,
            "price": price*0.999,
            "ccy": symbol[:-4],
            "is_hide": True,
            }
        if type=="market":
            data = {
            "market": symbol,
            "market_type": market_type,
            "side": side,
            "type": type,
            "amount": amount,
            "ccy": symbol[:-4],
            "is_hide": True,
            }

        data = json.dumps(data)
        response = coinex_request.request(
            "POST",
            "{url}{request_path}".format(url=coinex_request.url, request_path=request_path),
            data=data,
            )

    # if response.status_code > 300:
        # bot.send_message(f"!!! Order failed to open. {response.content}, {response.status_code}")
    return response.json()


def send_stop_order(coinex_request, symbol, amount, tp_or_sl, price, market_type="SPOT"):
    request_path = "/spot/stop-order"
    if tp_or_sl == "sl":
        data = {
                "market": symbol,
                "market_type": market_type,
                "side": "sell",
                "type": "limit",
                "amount": str(amount),
                "ccy": symbol[:-4],
                "price": str(price),
                "trigger_price": str(price*1.0005),
                }
    
    if tp_or_sl == "tp":
        data = {
                "market": symbol,
                "market_type": market_type,
                "side": "sell",
                "type": "limit",
                "amount": str(amount),
                "ccy": symbol[:-4],
                "price": str(price*0.9995),
                "trigger_price": str(price),
                }
    # if side == "buy":
    #     data = {
    #             "market": symbol,
    #             "market_type": market_type,
    #             "side": side,
    #             "type": "limit",
    #             "amount": amount,
    #             "ccy": symbol[:-4],
    #             "is_hide": "true",
    #             "price": price,
    #             "trigger_price": price*0.9995,
    #             }
    

    data = json.dumps(data)
    response = coinex_request.request(
        "POST",
        "{url}{request_path}".format(url=coinex_request.url, request_path=request_path),
        data=data,
        )
    return response.json()

# def get_depth(coinex_request, symbol):
#     request_path = "/spot/depth"
#     params = {
#         "market": symbol,
#         "limit": 5,
#         "interval": 0.0001
#     }
#     # params = json.dumps(params)
#     response = coinex_request.request(
#         "GET",
#         "{url}{request_path}".format(url=coinex_request.url, request_path=request_path),
#         params=params,
#         )
#     return response.json()["data"]["depth"]

def count_active_positions(all_trades, symbol):
    return (sum(1 for v in all_trades.values() if (v.get("position_status_active", False) and v.get('symbol', False) == symbol))), (sum(1 for v in all_trades.values() if v.get("position_status_active", False) ))

def open_position(
    account,
    coinex_request,
    bot,
    side: str,
    symbol: str,
    tp: float,
    sl: float,
    last_candle_close_price: float,
    base_price_mode: str = "tick_price",
    all_trades: dict = {},
    max_open_positions: int = 20,
    max_open_positions_all: int = 80,
    look_ahead_minutes: int = 10,
) -> dict:
    """
    Open a position using up to 1/n of available balance, with max n simultaneous positions.
    tp, sl: in percentage (e.g., 2 means 2%)
    base_price_mode: "tick_price" or "candle_close"
    """


    # 1. Check number of active positions
    current_active_positions, current_active_positions_all = count_active_positions(all_trades, symbol)
    if current_active_positions >= max_open_positions :
        bot.send_message(f"Max number of {max_open_positions} active positions in {symbol} reached for {account}.")
        return all_trades
    elif current_active_positions_all >= max_open_positions_all:
        bot.send_message(f"Max number of {max_open_positions_all} active positions reached for {account}.")
        return all_trades


    # remaining_slots = (max_open_positions - current_active_positions)

    # max_open_positions_all = max_open_positions * n_models
    remaining_slots = (max_open_positions_all - current_active_positions_all)

    # 2. Get USDT balance
    balance = get_spot_balance(coinex_request)
    usdt_balance = next((item for item in balance if item['ccy'] == 'USDT'), None)
    if not usdt_balance:
        bot.send_message("No USDT balance available.")
        return all_trades

    available_usdt = float(usdt_balance['available'])
    per_position_usdt = available_usdt / remaining_slots
    # per_position_usdt *= 1/n_models
    if current_active_positions == max_open_positions_all - 1:
        per_position_usdt *= 0.8
    if per_position_usdt < 1:
        bot.send_message("Available USDT per trade is below 1. Trade not opened.")
        return all_trades

    # 3. Calculate SL and TP prices
    if base_price_mode == "tick_price":
        entry_price = float(get_symbol_price(coinex_request, symbol))
    elif base_price_mode == "candle_close":
        entry_price = last_candle_close_price
    else:
        raise ValueError("Invalid base_price_mode")

    if side == "buy":
        tp_price = round(entry_price * (1 + tp / 100), 10)
        sl_price = round(entry_price * (1 - sl / 100), 10)
    elif side == "sell":
        tp_price = round(entry_price * (1 - tp / 100), 10)
        sl_price = round(entry_price * (1 + sl / 100), 10)
    else:
        raise ValueError("Invalid side")


    # 4. Calculate amount to trade
    amount_to_trade = round(per_position_usdt / entry_price, 10)


    # 5. Send market order
    if base_price_mode == "tick_price":
        order_result = send_order(
            coinex_request,
            order_order="open_pos",
            symbol=symbol,
            amount=amount_to_trade,
            side=side,
            type="market",
            market_type="SPOT"
        )

    elif base_price_mode == "candle_close":
        order_result = send_order(
            coinex_request,
            order_order="open_pos",
            symbol=symbol,
            amount=amount_to_trade,
            side=side,
            type="limit",
            price= entry_price,
            market_type="SPOT"
        )

    else:
        raise ValueError("Invalid base_price_mode")


    time.sleep(5)
    message = ""
    if not order_result.get("data"):
        bot.send_message(f"!!! Order failed to open for {account}.")
        return all_trades
    else:
        message += "--> position opened. 💲💲💲💲💲"
        order_id = int(order_result["data"]["order_id"])
        history_orders = get_history_orders(coinex_request, symbol)["data"]
        entry_order = next((o for o in history_orders if int(o["order_id"]) == order_id), None)
        filled_amount = float(entry_order["filled_amount"])
        flled_value = float(entry_order["filled_value"])
        if flled_value < 1:
            cancel_order(coinex_request, symbol, order_id, market_type="SPOT")
            bot.send_message(f"⚠️ Entry order was not filled.\ncancel order")
            return all_trades
        else:
            exit_side = "sell" if side == "buy" else "buy"
            tp_order = send_stop_order(coinex_request, symbol, filled_amount, "tp", tp_price)
            if not tp_order.get("data"):
                bot.send_message("!!!DANGER TP_Order failed to set.")
            else:
                message += "\nTP_Order successfully set."
            sl_order = send_stop_order(coinex_request, symbol, filled_amount, "sl", sl_price)
            if not tp_order.get("data"):
                bot.send_message("!!!DANGER SL_Order failed to set.")
            else:
                message += "\nSL_Order successfully set."

        bot.send_message(message)


    tp_order_id = int(tp_order["data"]["stop_id"])
    sl_order_id = int(sl_order["data"]["stop_id"])

    now = get_coinex_time_now()

    # 6. Register trade in all_trades
    all_trades[order_id] = {
        "side": side,
        "symbol": symbol,
        "entry_price": entry_price,
        "amount": amount_to_trade,
        "tp_price": tp_price,
        "sl_price": sl_price,
        "order_result": order_result,
        "open_position_time": now,
        "force_close_position_time": now + timedelta(minutes=look_ahead_minutes) - timedelta(seconds=now.second + 1),
        "position_status_active": True,
        "tp_order_id": tp_order_id,
        "sl_order_id": sl_order_id,
    }

    bot.send_message(f"--> Position opened for {account}.\nSymbol: {symbol},\nOrder ID: {order_id},\nSide: {side},\navg_price: {all_trades[order_id]['order_result']['data']['last_fill_price']},\nAmount: {amount_to_trade},\nTP: {tp_price},\nSL: {sl_price}\ntime to close: {(all_trades[order_id]['force_close_position_time']+timedelta(minutes=210)).strftime('%Y-%m-%d %H:%M:%S')}")

    return all_trades

def check_active_positions_for_time_force_close(account, coinex_request, all_trades, symbol, bot):
    now = get_coinex_time_now()
    # order_id = get_open_positions_order_id(coinex_request, symbol)
    # print(f"order_id: {order_id}")
    active_trades = []
    expired_trades = []
    n = 0
    for order_id, trade in all_trades.items():
        n += 1
        if n>400:
            break
        if trade["position_status_active"]:
            active_trades.append(int(order_id))
    for id in active_trades:
        force_close_position_time = all_trades[id]["force_close_position_time"]
        if (now >= force_close_position_time) and (all_trades[id]['symbol'] == symbol):
            expired_trades.append(id)
            bot.send_message(f"order {id}\nis expired for {account}")
            # now1 = now.strftime("%Y-%m-%d %H:%M:%S")
            # force_close_position_time1 = force_close_position_time.strftime("%Y-%m-%d %H:%M:%S")
            # bot.send_message(f"time to close order:{id} for {account} | Now time: {now1} , force_sell_time: {force_close_position_time1}")
            # order_result = close_positions_by_id(coinex_request, symbol, id, all_trades)

            # close_price = order_result["data"]["last_fill_price"]
            # bot.send_message(f"close order:{id}\nwith {close_price}$ price\nfor {account}")
            # tp_id = int(all_trades[id]["tp_order_id"])
            # sl_id = int(all_trades[id]["sl_order_id"])
            # cancel_stop_order(coinex_request, symbol, tp_id)
            # cancel_stop_order(coinex_request, symbol, sl_id)
            # all_trades[id]["position_status_active"] = False
            # all_trades[id]["position_close_time"] = get_coinex_time_now()

    return all_trades, expired_trades

def close_or_extend_expired_trades(account, coinex_request, signal_df, expired_trades, all_trades, bot):
    expired_symbols_id = {}
    expired_symbols = []
    for id in expired_trades:
        expired_symbols_id[id] = {}
        expired_symbols_id[id] = {'symbol': all_trades[id]['symbol']}
        expired_symbols.append(all_trades[id]['symbol'])
    expired_symbols = set(expired_symbols)

    for expired_symbol in expired_symbols:
        if expired_symbol in list(signal_df['strategy_target_symbol']):
            look_ahead_minutes = signal_df.loc[signal_df['strategy_target_symbol'] == expired_symbol, 'strategy_look_ahead'].values[0]
            signal_df = signal_df[signal_df['strategy_target_symbol'] != expired_symbol]
            for id in expired_trades:
                if all_trades[id]['symbol'] == expired_symbol:
                    now = get_coinex_time_now()
                    all_trades[id]['force_close_position_time'] = now + timedelta(minutes=look_ahead_minutes) - timedelta(seconds=now.second + 1)
                    bot.send_message(f"order:{id}\nis extended till {all_trades[id]['force_close_position_time']}\nfor {account}")
        else:
            for id in expired_trades:
                if all_trades[id]['symbol'] == expired_symbol:
                    now = get_coinex_time_now()
                    force_close_position_time = all_trades[id]["force_close_position_time"]
                    now1 = now.strftime("%Y-%m-%d %H:%M:%S")
                    force_close_position_time1 = force_close_position_time.strftime("%Y-%m-%d %H:%M:%S")
                    bot.send_message(f"time to close order:{id} for {account} | Now time: {now1} , force_sell_time: {force_close_position_time1}")
                    order_result = close_positions_by_id(coinex_request, expired_symbol, id, all_trades)
                    # print(f'order result:\n{order_result}')
                    close_price = order_result["data"]["last_fill_price"]
                    bot.send_message(f"close order:{id}\nwith {close_price}$ price\nfor {account}")
                    tp_id = int(all_trades[id]["tp_order_id"])
                    sl_id = int(all_trades[id]["sl_order_id"])
                    cancel_stop_order(coinex_request, expired_symbol, tp_id)
                    cancel_stop_order(coinex_request, expired_symbol, sl_id)
                    all_trades[id]["position_status_active"] = False
                    all_trades[id]["position_close_time"] = get_coinex_time_now()
    return all_trades, signal_df
    

def check_tp_sl_positions(coinex_request, all_trades, symbol):

    # check for open tp orders
    # tp_list = get_open_orders(coinex_request, symbol, "sell")["data"]
    # order_ids = [int(order["order_id"]) for order in tp_list]

    # check for open sl&tp orders
    id_list = get_open_stop_orders(coinex_request, symbol, "sell")["data"]
    stop_ids = [int(order["stop_id"]) for order in id_list]

    # try a loop for close open orders
    n = 0
    for trade in all_trades.values():
        n += 1
        if n>400:
            break
        if (trade['tp_order_id'] in stop_ids) and (trade['sl_order_id'] in stop_ids) :
            continue
        else:
            if trade['symbol'] == symbol:
                # print(trade['amount'])
                trade['position_status_active'] = False

                if (trade['tp_order_id'] in stop_ids) :
                    cancel_stop_order(coinex_request, symbol, int(trade['tp_order_id']))
                else:
                    cancel_stop_order(coinex_request, symbol, int(trade['sl_order_id']))

    return all_trades

def cancel_all_orders(coinex_request, symbol):
    limit_orders = get_open_orders(coinex_request, symbol)["data"]
    limit_ids = [int(order["order_id"]) for order in limit_orders]
    stop_orders = get_open_stop_orders(coinex_request, symbol)["data"]
    stop_ids = [int(order["stop_id"]) for order in stop_orders]
    for limit_id in limit_ids:
        cancel_order(coinex_request, symbol, limit_id)
    for stop_id in stop_ids:
        cancel_stop_order(coinex_request, symbol, stop_id)

def close_and_exit(coinex_request, symbol, all_trades=None):
    cancel_all_orders(coinex_request, symbol)
    balance = get_spot_balance(coinex_request)
    ccy = symbol[:-4]
    for i in range(len(balance)):
        if balance[i]["ccy"] == ccy:
            amount = balance[i]["available"]
            send_order(coinex_request, "close_pos", symbol, amount, "sell", "market")
    if all_trades:
        for id in all_trades.keys():
            if all_trades[id]["position_status_active"] == True:
                print(id)
                all_trades[id]["position_status_active"] = False
                print(all_trades[id]["position_status_active"])
    return all_trades


def stop_open_position():
    trade_permission = {'P': 0}
    current_path = os.getcwd()
    with open(f'{current_path}/data/trade_permission.pkl', 'wb') as f:
        pickle.dump(trade_permission, f)

def start_open_position():
    trade_permission = {'P': 1}
    current_path = os.getcwd()
    with open(f'{current_path}/data/trade_permission.pkl', 'wb') as f:
        pickle.dump(trade_permission, f)

