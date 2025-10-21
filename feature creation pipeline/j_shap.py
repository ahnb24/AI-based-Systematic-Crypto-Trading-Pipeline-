import xgboost as xgb
import shap
import os
import pandas as pd
import numpy as np

from configs.feature_configs_general import generate_general_config
config_general = generate_general_config()
keys = list(config_general.keys())
dataset_path = f'{os.getcwd()}/data/dataset/{keys[0]}/'


def load_all_xgb_models(folder_path, extensions=(".json", ".bin")):
    models = []
    for filename in os.listdir(folder_path):
        if filename.endswith(extensions):
            full_path = os.path.join(folder_path, filename)
            model = xgb.XGBClassifier()
            model.load_model(full_path)
            models.append((filename, model))
    return models

# Example usage

folder = f"{os.getcwd()}/model/{keys[0]}/step1"
os.makedirs(folder, exist_ok=True)
models = load_all_xgb_models(folder)

def cal_shap():
    n = 0
    shap_importance_dict = {}
    for name, model in models:
        print(n)
        explainer = shap.TreeExplainer(model)
        X_train = pd.read_parquet(f'{folder}/X_train_fold_{n}.parquet')
        shap_values = explainer.shap_values(X_train)
        shap_importance = np.abs(shap_values).mean(axis=0)
        shap_importance_dict[f'fold_{n}'] = {
        feature: importance for feature, importance in zip(X_train.columns, shap_importance)}
        n += 1
    shap_importance_df = pd.DataFrame(shap_importance_dict)
    shap_importance_df["mean_importance"] = shap_importance_df.mean(axis=1)
    shap_importance_df = shap_importance_df.sort_values(by='mean_importance', ascending=False)
    shap_importance_df = shap_importance_df.reset_index().rename(columns={'index': 'feature'})
    return shap_importance_df
# df=cal_shap()

def just_shap(n) :
    in_data = pd.read_parquet(f'{dataset_path}/initial_dataset.parquet')
    shap_df = cal_shap()
    time = pd.DataFrame(in_data['_time'])
    symbol = pd.DataFrame(in_data['symbol'])
    # shap_df = shap_df[shap_df['feature'].isin(in_data.columns)]

    shap_df=shap_df.reset_index(drop=True)
    top_n = int(n*0.01*shap_df.shape[0])
    shap_df = shap_df[:top_n]
    in_data = in_data[list(shap_df['feature'])]
    in_data = pd.concat([time, in_data], axis=1)
    in_data = pd.concat([in_data, symbol], axis=1)
    in_data.to_parquet(f'{dataset_path}/secondary_dataset.parquet')