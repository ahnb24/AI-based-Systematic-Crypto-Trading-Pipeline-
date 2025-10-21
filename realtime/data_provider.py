import json
import sys
from dotenv import load_dotenv
import os
load_dotenv()

# access_id = os.getenv('COINEX_ACCES_ID')
# secret_key = os.getenv('COINEX_SECRET_KEY')


from trade_execution.coinex_trade_execution_functions import (
    CoinexRequestsClient,
    get_spot_balance,
    get_open_stop_orders,
    get_open_orders,
    get_symbol_price,
    get_value,
)




def update_data(access_id, secret_key, file_path):
    coinex_request = CoinexRequestsClient(access_id, secret_key)
    symbols = ["AAVEUSDT",]

    balance = get_spot_balance(coinex_request)
    balance_dict = {i["ccy"]: {"available": i["available"], "frozen": i["frozen"]} for i in balance}

    balance_text = ""
    for k, v in balance_dict.items():
        balance_text += f"({k}:\navailable: {v['available']},\nfrozen: {v['frozen']})\n\n"

    open_orders = {sym: get_open_orders(coinex_request, sym)["data"] for sym in symbols}
    open_stop_orders = {sym: get_open_stop_orders(coinex_request, sym)["data"] for sym in symbols}

    value = sum(get_value(coinex_request, sym)[0] - get_value(coinex_request, sym)[1] for sym in symbols)
    value += get_value(coinex_request, symbols[0])[1]  # last USDT value

    symbol_price = {sym: get_symbol_price(coinex_request, sym) for sym in symbols}

    data = {
        "balance": f"balance:\n{balance_text}",
        "open_orders": f"open_orders: {open_orders}",
        "open_stop_orders": f"open_stop_orders: {open_stop_orders}",
        "asset_value": f"asset_value: {value}",
        "symbol_price": f"symbol_price:\n{symbol_price}",
    }

    with open(file_path, "w") as f:
        json.dump(data, f)
