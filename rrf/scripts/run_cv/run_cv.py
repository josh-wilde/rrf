import sys
import yaml

from rrf.cv import collect_nested_cv_data, run_nested_cv


def main():
    # Collect the configuration from the command line
    config_fpath = sys.argv[1]
    load_models = sys.argv[2] == 'True'
    save_best_refit_inner_model = sys.argv[3] == 'True'
    save_cv_results = sys.argv[4] == 'True'
    if len(sys.argv) == 6:
        split_id = int(sys.argv[5])
    else:
        split_id = None
    save_data_only = 'data_save' in config_fpath

    # Print the config path for the cluster logs
    print(f"config path: {config_fpath}")
    print(f"outer split id: {split_id}")

    # Parse the config
    with open(config_fpath) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    # Change the randomized split id to the input split id
    if split_id is not None:
        assert (
            config.get('model_selection_n_splits') is None,
            'This is designed for a randomized CV config, not k-fold outer split'
        )
        cv_setup = config['nested_cv_setup']
        cv_setup['randomized_outer_start_split'] = split_id
        cv_setup['randomized_outer_end_split'] = split_id + 1
        # Change the file save location for the model and the results
        cv_setup['models_save_dirpath'] = cv_setup['models_save_dirpath'] + f'/split_{split_id}'
        cv_setup['results_save_dirpath'] = cv_setup['results_save_dirpath'] + f'/split_{split_id}'

    # Get labels (tensor), feature_data (dict of tensors), and n_obs (int)
    labels, feature_data, n_obs = collect_nested_cv_data(config)

    if config.get('classifier'):
        if config['classifier']['name'] == 'SupClassifier':
            labels = (labels > 0).long()

    # Run nested CV
    (outer_results_df,
     inner_best_index_series,
     inner_results_df,
     best_model_history_df) = run_nested_cv(labels,
                                            feature_data,
                                            n_obs,
                                            config,
                                            load_models=load_models,
                                            save_best_refit_inner_model=save_best_refit_inner_model,
                                            save_cv_results=save_cv_results,
                                            save_data_only=save_data_only)


if __name__ == '__main__':
    main()
