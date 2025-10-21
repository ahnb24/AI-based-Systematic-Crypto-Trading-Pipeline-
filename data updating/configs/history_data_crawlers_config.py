from datetime import datetime
from pathlib import Path
import os

start_date_str = "2023/01/01"
# stop_date_str = "2024/07/01"
start_date = datetime.strptime(start_date_str, "%Y/%m/%d")
# stop_date = datetime.strptime(stop_date_str, "%Y/%m/%d")
stop_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)



metatrader_number_of_days = 340
root_path = str(os.path.dirname(os.path.abspath(__file__))).replace("configs", "")
data_folder = f"{root_path}/data/klines_data/"
Path(data_folder).mkdir(parents=True, exist_ok=True)

symbols_dict = {

    "BTCUSDT" :{
        "decimal_divide": 1e5,
        "pip_size": 1,
        "binance_id": "BTCUSDT",
    },
    'ETHUSDT' :{
        "decimal_divide": 1e5,
        "pip_size": 1,
        "binance_id": "ETHUSDT",
    },
    'XRPUSDT':{
        "decimal_divide": 1e5,
        "pip_size": 1,
        "binance_id": "XRPUSDT",
    },
    'BNBUSDT':{
        "decimal_divide": 1e5,
        "pip_size": 1,
        "binance_id": "BNBUSDT",
    },
    'SOLUSDT':{
        "decimal_divide": 1e5,
        "pip_size": 1,
        "binance_id": "SOLUSDT",
    },
    'DOGEUSDT':{
        "decimal_divide": 1e5,
        "pip_size": 1,
        "binance_id": "DOGEUSDT",
    }, 
    'ADAUSDT':{
        "decimal_divide": 1e5,
        "pip_size": 1,
        "binance_id": "ADAUSDT",
    },
    'TRXUSDT':{
        "decimal_divide": 1e5,
        "pip_size": 1,
        "binance_id": "TRXUSDT",
    },
    'LTCUSDT':{
        "decimal_divide": 1e5,
        "pip_size": 1,
        "binance_id": "LTCUSDT",
    },
    'BCHUSDT':{
        "decimal_divide": 1e5,
        "pip_size": 1,
        "binance_id": "BCHUSDT",
    },
    'AAVEUSDT':{
        "decimal_divide": 1e5,
        "pip_size": 1,
        "binance_id": "AAVEUSDT",
    },
    'FETUSDT':{
        "decimal_divide": 1e5,
        "pip_size": 1,
        "binance_id": "FETUSDT",
    },
    'FILUSDT':{
        "decimal_divide": 1e5,
        "pip_size": 1,
        "binance_id": "FILUSDT",
    },
    'HBARUSDT':{
        "decimal_divide": 1e5,
        "pip_size": 1,
        "binance_id": "HBARUSDT",
    },
    'LINKUSDT':{
        "decimal_divide": 1e5,
        "pip_size": 1,
        "binance_id": "LINKUSDT",
    },
    'NEARUSDT':{
        "decimal_divide": 1e5,
        "pip_size": 1,
        "binance_id": "NEARUSDT",
    },
    'UNIUSDT':{
        "decimal_divide": 1e5,
        "pip_size": 1,
        "binance_id": "UNIUSDT",
    },
    }
