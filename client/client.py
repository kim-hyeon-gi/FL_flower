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

from communication.base_utils import (
    ndarrays_to_sparse_parameters,
    sparse_parameters_to_ndarrays,
)
from communication.low_rank_utils import low_rank_ndarrays_to_sparse_parameters
from communication.prob_quantization_utils import (
    prob_quantization_ndarrays_to_sparse_parameters,
)
from communication.sparse_utils import (
    efficient_communication_ndarrays_to_sparse_parameters,
)
from model.utils import load_model, low_rank_train, test, train
from utils import Config, FitRes, get_grad, get_parameters, set_parameters


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
        params = []
        random_state = []
        V_value = []
        parameters_updated = []
        # Deserialize parameters to NumPy ndarray's
        parameters_original = ins.parameters
        ndarrays_original = sparse_parameters_to_ndarrays(
            parameters_original, config["structure"]
        )
        # Update local model, train, get updated parameters
        set_parameters(self.net, ndarrays_original)
        # if config["low_rank_p_value"] == []:
        train(
            self.net,
            self.trainloader,
            device=config["device"],
            epochs=config["local_epochs"],
            model_name=config["model_name"],
            lr=config["client_lr"],
        )
        ndarrays_updated = get_parameters(self.net)
        for i in range(len(ndarrays_updated)):
            arr = ndarrays_updated[i] - ndarrays_original[i]
            params.append(arr)
        # Serialize ndarray's into a Parameters object
        parameters_updated = ndarrays_to_sparse_parameters(params)
        # else:
        #     V_value, random_state = low_rank_train(
        #         self.net,
        #         self.trainloader,
        #         device=config["device"],
        #         epochs=config["local_epochs"],
        #         model_name=config["model_name"],
        #         lr=config["client_lr"],
        #         structure=config["structure"],
        #         shape_2d=config["shape_2d"],
        #         p_value=config["low_rank_p_value"],
        #     )

        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return FitRes(
            status=status,
            parameters=parameters_updated,
            num_examples=len(self.trainloader),
            random_state=random_state,
            V_value=V_value,
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        print(f"[Client {self.cid}] evaluate, config: {ins.config}")

        # Deserialize parameters to NumPy ndarray's
        parameters_original = ins.parameters
        ndarrays_original = sparse_parameters_to_ndarrays(
            parameters_original, self.config["structure"]
        )

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
