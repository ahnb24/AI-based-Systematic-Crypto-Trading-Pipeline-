def run_all_pipeline():
    from configs.feature_configs_general import generate_general_config
    config_general = generate_general_config()

    from stage_one_data.history_data_stage_one_func import history_data_stage_one
    history_data_stage_one(config_general)
    
    from realtime_candle.realtime_candle_func import historiy_realtime_candle
    historiy_realtime_candle(config_general)

    from create_features.indicator_func import history_indicator_calculator
    history_indicator_calculator(config_general)

    from create_features.realtime_shift_func import history_cndl_shift
    history_cndl_shift(config_general)

    from create_features.create_basic_features_func import history_basic_features, history_fe_market_close, history_fe_time
    history_basic_features(config_general)
    history_fe_market_close(config_general)
    history_fe_time(config_general)

    from create_features.window_agg_features_func import history_fe_WIN_features
    history_fe_WIN_features(config_general)



