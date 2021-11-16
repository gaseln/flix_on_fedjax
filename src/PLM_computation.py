from typing import Callable

from fedjax.core import dataclasses
from fedjax.core import federated_data
from fedjax.core import tree_util
from fedjax.core.typing import BatchExample
from fedjax.core.typing import Params
from fedjax.core.typing import PRNGKey

import fedjax
import itertools
import PLM

Grads = Params


@dataclasses.dataclass
class PLMComputationHParams:
    num_epochs: int
    lr: float
    batch_size: int


@dataclasses.dataclass
class ComputationParams:
    init_params: Params
    num_clients: int
    num_clients_per_round: int


def plm_computation_with_statistics(
    train_fd: federated_data.FederatedData,
    validation_fd: federated_data.FederatedData,
    grad_fn: Callable[[Params, BatchExample, PRNGKey], Grads],
    grad_fn_eval: Callable[[Params, BatchExample], Grads],
    plm_comp_hparams: PLMComputationHParams,
    comp_params: ComputationParams
) -> Params:
    if comp_params.num_clients % comp_params.num_clients_per_round != 0:
        raise ValueError('num_clients_per_round must divide num_clients')

    print('PLM computation: num_epochs = {}, lr = {}, b_size = {}'.format(
        plm_comp_hparams.num_epochs,
        plm_comp_hparams.lr,
        plm_comp_hparams.batch_size))
    client_sampler = fedjax.client_samplers.UniformShuffledClientSampler(
        shuffled_clients_iter=itertools.islice(
            train_fd.clients(), comp_params.num_clients),
        num_clients=comp_params.num_clients_per_round)
    client_optimizer = fedjax.optimizers.sgd(learning_rate=plm_comp_hparams.lr)
    client_batch_hparams = fedjax.ShuffleRepeatBatchHParams(
        batch_size=plm_comp_hparams.batch_size,
        num_epochs=plm_comp_hparams.num_epochs,
        drop_remainder=False)
    algorithm = PLM.PLM(grad_fn, client_optimizer, client_batch_hparams)
    server_state = algorithm.init(comp_params.init_params)
    total_rounds = comp_params.num_clients // comp_params.num_clients_per_round
    grad_norms = []
    for round_num in range(total_rounds):
        print('Round {} / {}'.format(round_num + 1, total_rounds), end='\r')
        clients = client_sampler.sample()
        server_state, _ = algorithm.apply(server_state, clients)
        if grad_fn_eval is not None:
            grad_norms += [
                tree_util.tree_l2_norm(grad_fn_eval(
                    server_state.PLM[cid], cds.all_examples()))
                for cid, cds, _ in clients]
    return server_state.PLM, grad_norms


def plm_computation(
    train_fd: federated_data.FederatedData,
    grad_fn: Callable[[Params, BatchExample, PRNGKey], Grads],
    plm_comp_hparams: PLMComputationHParams,
    comp_params: ComputationParams
) -> Params:
    PLM_dict, _ = plm_computation_with_statistics(
        train_fd, None, grad_fn, None, plm_comp_hparams, comp_params)
    return PLM_dict