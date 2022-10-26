from typing import Dict, Any
import os


def validate_nested_cv_config(
    config: Dict[str, Any],
    load_models: bool,
    save_models: bool,
    save_results: bool,
    save_data_only: bool
):

    # Assert that the nested_cv_setup parameters exist
    assert 'nested_cv_setup' in config, 'CV setup parameters missing from config'

    # If you want to save or load models or save results, make sure there is a file path and the path exists
    if save_models or load_models:
        models_dirpath = config['nested_cv_setup'].get('models_save_dirpath')
        assert models_dirpath, 'Cannot save or load models, models_save_dirpath does not exist.'
        os.makedirs(models_dirpath, exist_ok=True)
    if save_results or load_models:
        results_dirpath = config['nested_cv_setup'].get('results_save_dirpath')
        assert results_dirpath, 'Cannot save results, results_save_dirpath does not exist.'
        os.makedirs(results_dirpath, exist_ok=True)
    if save_data_only:
        data_dirpath = config['nested_cv_setup'].get('cv_data_save_dirpath')
        assert data_dirpath, 'Cannot save data, cv_data_save_dirpath does not exist.'
        os.makedirs(data_dirpath, exist_ok=True)

    if config['nested_cv_setup'].get('hp_selection_n_splits') is None:
        assert config.get('classifier') is None or config['classifier'].get('search_space') is None, 'Error: search space cannot exist if there is no HP selection.'

    if (
            config['nested_cv_setup'].get('randomized_outer_start_split') is not None or
            config['nested_cv_setup'].get('randomized_outer_end_split') is not None
    ):
        assert config['nested_cv_setup'].get('model_selection_n_splits') is None, "Error: model_selection_n_splits should not exist if we are doing randomized splits"
        assert config['nested_cv_setup'].get('test_frac') is not None, 'Error: need test_frac in nested_cv_setup to do randomized splits'
        assert config['nested_cv_setup'].get('randomized_outer_end_split') is not None, f"Error: need randomized_outer_end_split if specifying randomized_outer_start_split."
        assert config['nested_cv_setup'].get('randomized_outer_start_split') is not None, f"Error: need randomized_outer_start_split if specifying randomized_outer_end_split."
        assert config['nested_cv_setup']['randomized_outer_end_split'] - config['nested_cv_setup']['randomized_outer_start_split'] > 0, "Error: randomized end split must be greater than randomized start split"

    if config['nested_cv_setup'].get('model_selection_n_splits') is not None:
        assert config['nested_cv_setup'].get('randomized_outer_start_split') is None, "Error: randomized_outer_start_split should not exist if we are doing k-fold splits"
        assert config['nested_cv_setup'].get('randomized_outer_end_split') is None, "Error: randomized_outer_end_split should not exist if we are doing k-fold splits"
