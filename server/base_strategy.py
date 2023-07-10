import flwr as fl

from model.utils import load_model
from server.utils import evaluate, fit_config, weighted_average
from utils import get_parameters


def base_strategy(config):
    # Create an instance of the model and get the parameters
    params = get_parameters(load_model(config))
    # Create FedAvg strategy
    strategy = fl.server.strategy.FedAdagrad(  # FedAdagrad  , FedAvg
        fraction_fit=config.fraction_fit,
        fraction_evaluate=config.fraction_evaluate,
        min_fit_clients=config.min_fit_clients,
        min_evaluate_clients=config.min_evaluate_clients,
        min_available_clients=config.num_clients,
        evaluate_metrics_aggregation_fn=weighted_average,  # <-- pass the metric aggregation function
        initial_parameters=fl.common.ndarrays_to_parameters(params),
        evaluate_fn=evaluate,  # Pass the evaluation function
        on_fit_config_fn=fit_config,
    )
    return strategy
