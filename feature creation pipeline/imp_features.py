import pandas as pd
import os
import pickle as pkl
from configs.feature_configs_general import generate_general_config
config_general = generate_general_config()
keys = list(config_general.keys())

def imp_features_filter(step, top_percent=10, model_type='XGB'):
    
    dataset_path = f'{os.getcwd()}/data/dataset/{keys[0]}/'
    with open(f'{dataset_path}/exp_obj_and_man_params.pkl', "rb") as f:
        exp_obj, man_params = pkl.load(f)
    importance_df = exp_obj.feature_importance
    if step == 2:
        with open(f'{dataset_path}/dropped_features.pkl', "rb") as f:
            drop_list = pkl.load(f)
        importance_df = importance_df[~importance_df['feature_name'].isin(drop_list)]
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
    if step == 1:
        the_name = f"df_feature_imp_{trade_mode}_{symbol}_M{look_ahead}_TP{take_profit}_SL{stop_loss}_step1.parquet"
        imp_df = top_features[['feature_name', 'mean_importance']]
        imp_df.to_parquet(f'{dataset_path}/{the_name}')
    if step == 2 :
        the_name = f"df_feature_imp_{trade_mode}_{symbol}_M{look_ahead}_TP{take_profit}_SL{stop_loss}_step2.parquet"
        imp_df = top_features[['feature_name', 'mean_importance']]
        imp_df.to_parquet(f'{dataset_path}/{the_name}')
        mid_dataset = pd.read_parquet(f'{dataset_path}/mid_dataset.parquet')
        # print(list(imp_df['feature_name']))
        dataset = mid_dataset[['_time'] + list(imp_df['feature_name']) + ['symbol']]
        dataset.to_parquet(f'{dataset_path}/dataset.parquet')
        # mid_dataset.to_parquet(f'{dataset_path}/raw_dataset.parquet')
