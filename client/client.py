import flwr as fl
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from model.utils import load_model, test, train
from utils import Config, get_parameters, set_parameters


class FlowerClient(fl.client.Client):
    def __init__(self, cid, net, trainloader, valloader):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        print(f"[Client {self.cid}] get_parameters")

        # Get parameters as a list of NumPy ndarray's
        ndarrays: List[np.ndarray] = get_parameters(self.net)

        # Serialize ndarray's into a Parameters object
        parameters = ndarrays_to_parameters(ndarrays)

        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return GetParametersRes(
            status=status,
            parameters=parameters,
        )

    def fit(self, ins: FitIns) -> FitRes:
        print(f"[Client {self.cid}] fit, config: {ins.config}")
        config = ins.config
        # Deserialize parameters to NumPy ndarray's
        parameters_original = ins.parameters
        ndarrays_original = parameters_to_ndarrays(parameters_original)

        # Update local model, train, get updated parameters
        set_parameters(self.net, ndarrays_original)
        train(
            self.net,
            self.trainloader,
            device=config["device"],
            epochs=config["local_epochs"],
        )

        # Serialize ndarray's into a Parameters object
        parameters_updated = ndarrays_to_parameters(ndarrays_updated)

        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return FitRes(
            status=status,
            parameters=parameters_updated,
            num_examples=len(self.trainloader),
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        print(f"[Client {self.cid}] evaluate, config: {ins.config}")

        # Deserialize parameters to NumPy ndarray's
        parameters_original = ins.parameters
        ndarrays_original = parameters_to_ndarrays(parameters_original)

        set_parameters(self.net, ndarrays_original)
        loss, accuracy = test(self.net, self.valloader)
        # return float(loss), len(self.valloader), {"accuracy": float(accuracy)}

        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return EvaluateRes(
            status=status,
            loss=float(loss),
            num_examples=len(self.valloader),
            metrics={"accuracy": float(accuracy)},
        )


# class FlowerClient(fl.client.NumPyClient):
#     def __init__(self, cid, net, trainloader, valloader):
#         self.cid = cid
#         self.net = net
#         self.trainloader = trainloader
#         self.valloader = valloader

#     def get_parameters(self, config):
#         print(f"[Client {self.cid}] get_parameters")
#         return get_parameters(self.net)

#     def fit(self, parameters, config):
#         # Read values from config
#         server_round = config["server_round"]
#         local_epochs = config["local_epochs"]

#         # Use values provided by the config
#         print(f"[Client {self.cid}, round {server_round}] fit, config: {config}")
#         set_parameters(self.net, parameters)
#         train(self.net, self.trainloader, device=config["device"], epochs=local_epochs)
#         return get_parameters(self.net), len(self.trainloader), {}

#     def evaluate(self, parameters, config):
#         server_round = config["server_round"]
#         set_parameters(self.net, parameters)
#         loss, accuracy = test(self.net, self.valloader)
#         print(
#             f"[Client {self.cid}] evaluate,server_round : {server_round}, config: {config} , accuracy : {accuracy}"
#         )
#         return float(loss), len(self.valloader), {"accuracy": float(accuracy)}
