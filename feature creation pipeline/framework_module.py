def run_all_framework(step):
    model_type = 'XGB' #? RF , XGB, 'LGBM'
    MANUAL_EXP = True
    import sys
    import warnings
    import os
    import pickle as pkl

    # Get the parent directory
    # pipeline_folder = os.path.dirname(os.path.dirname(os.getcwd()))

    # Add folder_a to sys.path
    # sys.path.append(os.path.join(pipeline_folder, "configs"))

    # Import the module
    from configs.feature_configs_general import generate_general_config
    config_general = generate_general_config()
    keys = list(config_general.keys())

    warnings.simplefilter(action='ignore', category=FutureWarning)

    from framework.MLOPS_Training.main_func import main
    from framework.MLOPS_Training.ETL import read_data_manual, ETL
    from framework.MLOPS_Training.utils.wandb_utils import fetch_artifacts, read_tracker_objects, download_wandb_artifact
    man_params = {'RF':None, 'XGB':None}
    man_params['XGB'] = {
        'model_name' : 'XGB',
        
        'target_symbol' : keys[0],
        'trade_mode': 'long' ,   #"long" , "short"
        'trg_look_ahead' : 1082,
        'trg_take_profit' : 5.324171640486003,
        'trg_stop_loss' : 8.311164521067003,

        'strg_look_ahead' : 1082,
        'strg_take_profit' : 5.324171640486003,
        'strg_stop_loss' : 8.311164521067003,

        'n_rand_features': None,
        'save_model_mode' : None, # None, 'sample_train_size', 'last_train_size', 'all_data',
        'n_splits' : 25,
        'max_train_size' : 51840,
        'test_size' : 7200,
        'train_test_gap': 00*288,

        "parameters": {
            "colsample_bylevel": 0.6341511291537605,
            "colsample_bynode": 0.672825148950015,
            "colsample_bytree": 0.7060142986367952,
            'device' : 'cuda',#  None,'cuda'
            'predictor': 'gpu_predictor',
            'eval_metric':'aucpr',
            "gamma": 7, #? float
            "learning_rate": 0.4784490065192545, 
            "max_bin": 512,            # New: Helps 'hist' method with numerical stability.
            "max_delta_step": 2, #?float
            "max_depth": 4,
            "max_leaves": 20,
            "min_child_weight": 8,
            "n_estimators": 295,
            # "n_jobs" : -1,
            "objective": "binary:logitraw",
            'random_state': 42,
            "reg_alpha": 14,
            "reg_lambda": 3,
            "scale_pos_weight" : 3.555452464954229,
            "subsample": 0.4567183777491009,
            'tree_method':'hist',#  None,hist

            # "early_stopping_rounds" :{"distribution": "int_uniform", "min": 5, "max": 150},
            # "early_stopping_rounds" :50,
            
            },

    }
    if step == 1:
        dataset_path = f'{os.getcwd()}/data/dataset/{keys[0]}/initial_dataset.parquet'
    if step == 2:
        dataset_path = f'{os.getcwd()}/data/dataset/{keys[0]}/selected_features1.parquet'
    if step == 3:
        dataset_path = f'{os.getcwd()}/data/dataset/{keys[0]}/dataset.parquet'

    if MANUAL_EXP:
        exp_obj, exp_metadata, artifact_name = main(
            manual = True,
            man_params = man_params[model_type],
            dataset_path = dataset_path,
            C5M_data_path = f'{os.getcwd()}/data/stage_one_data',
            step = step
        )
        # if step == 1:
        #     with open(f"{os.getcwd()}/data/dataset/{keys[0]}/exp_obj_and_man_params.pkl", "wb") as f:
        #         pkl.dump((exp_obj, man_params), f)
