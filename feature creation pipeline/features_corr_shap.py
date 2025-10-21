from scipy.stats import pearsonr
import pandas as pd
import numpy as np
import pickle as pkl
import os
import xgboost as xgb
import shap
from configs.feature_configs_general import generate_general_config
from joblib import Parallel, delayed
from itertools import combinations


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


def cal_shap(step=1):
    folder = f"{os.getcwd()}/model/{keys[0]}/step{step}"
    models = load_all_xgb_models(folder)
    shap_importance_dict = {}

    for n, (name, model) in enumerate(models):
        X_train = pd.read_parquet(f'{folder}/X_train_fold_{n}.parquet')

        # Use "approximate" method for performance
        explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        shap_values = explainer.shap_values(X_train)

        shap_importance = np.abs(shap_values).mean(axis=0)
        shap_importance_dict[f'fold_{n}'] = dict(zip(X_train.columns, shap_importance))

    shap_importance_df = pd.DataFrame(shap_importance_dict)
    shap_importance_df["mean_importance"] = shap_importance_df.mean(axis=1)
    shap_importance_df = shap_importance_df.reset_index().rename(columns={'index': 'feature'})
    shap_importance_df.sort_values(by="mean_importance", ascending=False, inplace=True)

    return shap_importance_df


def calculate_correlation(data1, data2):
    return pearsonr(data1, data2)[0]


def corr_shap(step):
    if step == 1:
        df = pd.read_parquet(f'{dataset_path}/secondary_dataset.parquet')
    else:
        df = pd.read_parquet(f'{dataset_path}/dataset.parquet')

    df_features = df.select_dtypes(include=[np.number])
    shap_df = cal_shap()
    shap_map = dict(zip(shap_df["feature"], shap_df["mean_importance"]))

    feature_list = list(df_features.columns)
    del_set = set()

    def process_pair(i, j):
        if i in del_set or j in del_set:
            return None
        corr_val = calculate_correlation(df_features[i], df_features[j])
        if abs(corr_val) > 0.85:
            if shap_map.get(i, 0) > shap_map.get(j, 0):
                return j
            else:
                return i
        return None

    pairs = list(combinations(feature_list, 2))
    to_delete = Parallel(n_jobs=-1)(delayed(process_pair)(i, j) for i, j in pairs)
    
    for col in to_delete:
        if col is not None:
            del_set.add(col)

    df = df.drop(columns=list(del_set))

    if step == 1:
        df.to_parquet(f"{dataset_path}/selected_features1.parquet")
    else:
        print(df)




# from scipy.stats import pearsonr
# import pandas as pd
# import numpy as np
# import pickle as pkl
# import os
# import xgboost as xgb
# import shap
# from configs.feature_configs_general import generate_general_config
# from joblib import Parallel, delayed
# from itertools import combinations


# config_general = generate_general_config()
# keys = list(config_general.keys())
# dataset_path = f'{os.getcwd()}/data/dataset/{keys[0]}/'


# def load_all_xgb_models(folder_path, extensions=(".json", ".bin")):
#     models = []
#     for filename in os.listdir(folder_path):
#         if filename.endswith(extensions):
#             full_path = os.path.join(folder_path, filename)
#             model = xgb.XGBClassifier()
#             model.load_model(full_path)
#             models.append((filename, model))
#     return models


# def cal_shap(step=1):
#     folder = f"{os.getcwd()}/model/{keys[0]}/step{step}"
#     models = load_all_xgb_models(folder)
#     shap_agg = {}

#     for n, (name, model) in enumerate(models):
#         X_train = pd.read_parquet(f'{folder}/X_train_fold_{n}.parquet').astype(np.float32)
#         explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
#         shap_values = explainer.shap_values(X_train)
        
#         # Free up memory
#         X_train = None

#         shap_mean = np.mean(np.abs(shap_values), axis=0)
#         shap_agg[f'fold_{n}'] = shap_mean
#         del shap_values

#     # Get feature names from last model input
#     X_sample = pd.read_parquet(f'{folder}/X_train_fold_0.parquet')
#     features = X_sample.columns.tolist()
#     X_sample = None

#     shap_df = pd.DataFrame(shap_agg, index=features)
#     shap_df["mean_importance"] = shap_df.mean(axis=1)
#     shap_df = shap_df.reset_index().rename(columns={'index': 'feature'})
#     shap_df.sort_values(by="mean_importance", ascending=False, inplace=True)

#     return shap_df


# def calculate_correlation(col1, col2):
#     return pearsonr(col1, col2)[0]


# def corr_shap(step):
#     if step == 1:
#         df = pd.read_parquet(f'{dataset_path}/secondary_dataset.parquet')
#     else:
#         df = pd.read_parquet(f'{dataset_path}/dataset.parquet')

#     df = df.select_dtypes(include=[np.number]).astype(np.float32)
#     shap_df = cal_shap(step)
#     shap_map = dict(zip(shap_df["feature"], shap_df["mean_importance"]))

#     features = df.columns.tolist()
#     del_set = set()

#     def process_pair(i, j):
#         if i in del_set or j in del_set:
#             return None
#         corr_val = calculate_correlation(df[i], df[j])
#         if abs(corr_val) > 0.85:
#             if shap_map.get(i, 0) > shap_map.get(j, 0):
#                 return j
#             else:
#                 return i
#         return None

#     pairs = combinations(features, 2)
#     to_delete = Parallel(n_jobs=-1, prefer="threads")(delayed(process_pair)(i, j) for i, j in pairs)

#     for col in to_delete:
#         if col is not None:
#             del_set.add(col)

#     df.drop(columns=list(del_set), inplace=True)

#     if step == 1:
#         df.to_parquet(f"{dataset_path}/selected_features1.parquet")
#     else:
#         print(df)
