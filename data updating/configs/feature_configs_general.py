
symbols = ['FETUSDT', 'BTCUSDT']


general_config = {


  'base_candle_timeframe': [15,30,60,120,180,240,480,860,1440],

  

  'fe_ATR': {'timeframe': [15, 60, 240, 1440],
  'window_size': [7, 21, 52],
  'base_columns': ['HIGH', 'CLOSE', 'LOW']},


  'fe_RSTD': {'timeframe': [15, 60, 240, 1440],
  'window_size': [7, 21, 52],
  'base_columns': ['CLOSE']},


  'fe_WIN': {'timeframe': [5],
  'window_size': [5,],
  'base_columns': ['CLOSE']},


  'fe_cndl': [5, 15, 60, 240, 1440],


  'fe_EMA': {'timeframe': [15, 60,],
  'window_size': [7, 21, 52],
  'base_columns': ['CLOSE']},

  'fe_SMA': {'base_columns': ['CLOSE'],
  'timeframe': [240, 1440],
  'window_size': [7, 21, 52]},

  'fe_RSI': {'timeframe': [15, 60, 240, 1440],
  'window_size': [7, 21, 52],
  'base_columns': ['CLOSE']},


  'fe_cndl_shift': {'columns': ['OPEN', 'HIGH', 'LOW', 'CLOSE'],
  'shift_configs': [
    {'timeframe': 5, 'shift_sizes': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
    {'timeframe': 15, 'shift_sizes': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
    {'timeframe': 30, 'shift_sizes': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
    {'timeframe': 60, 'shift_sizes': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
    {'timeframe': 240, 'shift_sizes': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
    {'timeframe': 1440, 'shift_sizes': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}]},


  'fe_ratio': {'ATR': {'timeframe': [15, 60, 240, 1440],
    'window_size': [(7, 21), (7, 52),(21, 52),]},

  'EMA': {'timeframe': [15, 60,], 
    'window_size': [(7, 21), (7, 52),(21, 52), 
    ]},

  'RSI': {'timeframe': [15, 60, 240, 1440],
    'window_size': [(7, 21), (7, 52),(21, 52),]},

  'RSTD': {'timeframe': [15, 60, 240, 1440],
    'window_size': [(7, 21), (7, 52),(21, 52),]},

  'SMA': {'timeframe': [240, 1440], 
    'window_size': [(7, 21), (7, 52),(21, 52),]}
          },

}

def generate_general_config(symbols=symbols,general_config=general_config):
  config_dict = {}
  for sym in symbols:
    config_dict[sym] = general_config
  return config_dict