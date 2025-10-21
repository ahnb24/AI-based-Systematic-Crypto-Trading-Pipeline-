from features_corr_shap import cal_shap
import pandas as pd
import os

from configs.feature_configs_general import generate_general_config
config_general = generate_general_config()
keys = list(config_general.keys())
dataset_path = f'{os.getcwd()}/data/dataset/{keys[0]}/'

def rank_and_filter() :
    in_dataset = pd.read_parquet(f'{dataset_path}/initial_dataset.parquet')
    in_len = in_dataset.shape[1]
    sel_feat1 = pd.read_parquet(f'{dataset_path}/selected_features1.parquet')
    shap_df = cal_shap(step=2)
    time = pd.DataFrame(sel_feat1['_time'])
    symbol = pd.DataFrame(sel_feat1['symbol'])
    shap_df = shap_df[shap_df['feature'].isin(sel_feat1.columns)]

    shap_df=shap_df.reset_index(drop=True)
    top_n = int(0.25*in_len)
    shap_df = shap_df[:top_n]
    sel_feat1 = sel_feat1[list(shap_df['feature'])]
    sel_feat1 = pd.concat([time, sel_feat1], axis=1)
    sel_feat1 = pd.concat([sel_feat1, symbol], axis=1)
    sel_feat1.to_parquet(f'{dataset_path}/dataset.parquet')

