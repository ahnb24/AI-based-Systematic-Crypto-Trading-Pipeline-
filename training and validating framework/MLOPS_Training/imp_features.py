import pandas as pd
import os
import pickle as pkl

def imp_features_filter(exp_obj, man_params, top_percent=30, model_type='XGB'):
    dataset_path = os.path.dirname(os.getcwd()) + "/Framework data"
    importance_df = exp_obj.feature_importance
    top_n = int(top_percent*importance_df.shape[0]/100)
    mean_importance = importance_df['mean_importance']
    sorted_idx = mean_importance.argsort()[::-1]
    top_features = importance_df.iloc[sorted_idx[:top_n]]

    symbol = man_params[model_type]['target_symbol']
    trade_mode = man_params[model_type]['trade_mode']
    look_ahead = man_params[model_type]['trg_look_ahead']
    take_profit = man_params[model_type]['trg_take_profit']
    stop_loss = man_params[model_type]['trg_stop_loss']
    # Prepare data for scatter plot
    the_name = f"df_feature_imp_{trade_mode}_{symbol}_M{look_ahead}_TP{take_profit}_SL{stop_loss}.pkl"
    imp_df = top_features[['feature_name', 'mean_importance']]
    imp_df.to_pickle(f'{dataset_path}/{the_name}')
    raw_dataset = pd.read_parquet(f'{dataset_path}/dataset.parquet')
    print(list(imp_df['feature_name']))
    dataset = raw_dataset[['_time'] + list(imp_df['feature_name']) + ['symbol']]
    dataset.to_parquet(f'{dataset_path}/dataset.parquet')
    raw_dataset.to_parquet(f'{dataset_path}/raw_dataset.parquet')
