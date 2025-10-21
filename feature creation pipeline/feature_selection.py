import pandas as pd
import os
from configs.feature_configs_general import generate_general_config
config_general = generate_general_config()
keys = list(config_general.keys())
import pickle as pkl

def feature_selection(cor_tresh=0.85, model_type='XGB') :
    dataset_path = f'{os.getcwd()}/data/dataset/{keys[0]}/'
    with open(f'{dataset_path}/exp_obj_and_man_params.pkl', "rb") as f:
        exp_obj, man_params = pkl.load(f)
    df_cor = pd.read_parquet(f'{os.getcwd()}/data/dataset/{keys[0]}/features_correlation.parquet')
    df_cor = df_cor[abs(df_cor['correlation']) > cor_tresh]
    symbol = man_params[model_type]['target_symbol']
    trade_mode = man_params[model_type]['trade_mode']
    look_ahead = man_params[model_type]['trg_look_ahead']
    take_profit = man_params[model_type]['trg_take_profit']
    stop_loss = man_params[model_type]['trg_stop_loss']
    the_name = f"df_feature_imp_{trade_mode}_{symbol}_M{look_ahead}_TP{take_profit}_SL{stop_loss}_step1.parquet"
    df_imp = pd.read_parquet(f'{os.getcwd()}/data/dataset/{keys[0]}/{the_name}')
    merged = df_cor.merge(df_imp, left_on="feature1", right_on="feature_name", how="left").rename(columns={"mean_importance": "importance1"})
    merged = merged.merge(df_imp, left_on="feature2", right_on="feature_name", how="left").rename(columns={"mean_importance": "importance2"})

    # Fill NaN importance with 0 (assuming missing values mean no importance recorded)
    merged[["importance1", "importance2"]] = merged[["importance1", "importance2"]].fillna(0)

    # Keep the feature with higher importance
    merged["dropped_feature"] = merged.apply(lambda row: row["feature1"] if row["importance1"] <= row["importance2"] else row["feature2"], axis=1)

    # Select relevant columns
    drop_list = list(merged["dropped_feature"])
    with open(f"{os.getcwd()}/data/dataset/{keys[0]}/dropped_features.pkl", "wb") as f:
            pkl.dump(drop_list, f)

    # print(drop_list)
    initial_dataset = pd.read_parquet(f'{os.getcwd()}/data/dataset/{keys[0]}/initial_dataset.parquet')
    mid_dataset = initial_dataset.drop(columns=drop_list)
    mid_dataset.to_parquet(f'{os.getcwd()}/data/dataset/{keys[0]}/mid_dataset.parquet')

