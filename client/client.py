import flwr as fl
import numpy as np
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    GetParametersIns,
    GetParametersRes,
    Status,
)

from model.utils import load_model, test, train
from utils import (
    Config,
    FitRes,
    get_grad,
    get_parameters,
    ndarrays_to_sparse_parameters,
    set_parameters,
    sparse_parameters_to_ndarrays,
)


class FlowerClient(fl.client.Client):
    def __init__(self, cid, net, trainloader, valloader):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        print(f"[Client {self.cid}] get_parameters")

        # Get parameters as a list of NumPy ndarray's
        ndarrays: List[np.ndarray] = get_grad(self.net)

        # Serialize ndarray's into a Parameters object
        parameters = ndarrays_to_sparse_parameters(ndarrays)

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
        ndarrays_original = sparse_parameters_to_ndarrays(
            parameters_original, config["structure"], 1
        )
        # Update local model, train, get updated parameters
        set_parameters(self.net, ndarrays_original)
        train(
            self.net,
            self.trainloader,
            device=config["device"],
            epochs=config["local_epochs"],
        )
        ndarrays_updated = get_parameters(self.net)
        params = []

        for i in range(len(ndarrays_updated)):
            arr = ndarrays_updated[i] - ndarrays_original[i]
            params.append(arr)

        # Serialize ndarray's into a Parameters object
        parameters_updated, random_state = ndarrays_to_sparse_parameters(
            params, config["structure"], config["sparse_dense"]
        )

        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return FitRes(
            status=status,
            parameters=parameters_updated,
            num_examples=len(self.trainloader),
            random_state=random_state,
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        print(f"[Client {self.cid}] evaluate, config: {ins.config}")

        # Deserialize parameters to NumPy ndarray's
        parameters_original = ins.parameters
        ndarrays_original = sparse_parameters_to_ndarrays(parameters_original)

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
