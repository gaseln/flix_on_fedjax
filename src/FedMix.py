from typing import Any, Callable, Mapping, Sequence, Tuple

from fedjax.core import client_datasets
from fedjax.core import dataclasses
from fedjax.core import federated_algorithm
from fedjax.core import federated_data
from fedjax.core import for_each_client
from fedjax.core import optimizers
from fedjax.core import tree_util
from fedjax.core.typing import BatchExample
from fedjax.core.typing import Params
from fedjax.core.typing import PRNGKey

import jax

Grads = Params


def convex_combination(
    x_global: Params,
    x_local: Params,
    alpha: float
) -> Params:
    """Computes alpha * x_global + (1 - alpha) * x_local for PyTrees."""
    return tree_util.tree_add(tree_util.tree_weight(x_global, alpha),
                              tree_util.tree_weight(x_local, 1 - alpha))


def create_train_for_each_client(grad_fn):
    """Builds client_init, client_step, client_final for for_each_client."""
    def client_init(server_params, client_input):
        client_alpha = client_input['alpha']
        client_plm = client_input['plm']
        client_rng = client_input['rng']
        client_step_state = {
            'params': server_params,
            'rng': client_rng,
            'grad': None,
            'plm': client_plm,
            'alpha': client_alpha,
        }
        return client_step_state

    def client_step(client_step_state, batch):
        rng, use_rng = jax.random.split(client_step_state['rng'])
        point = convex_combination(client_step_state['params'],
                                   client_step_state['plm'],
                                   client_step_state['alpha'])
        grad = grad_fn(point, batch, use_rng)
        grad = tree_util.tree_weight(grad, client_step_state['alpha'])
        next_client_step_state = {
            'params': client_step_state['alpha'],
            'rng': rng,
            'grad': grad,
            'plm': client_step_state['plm'],
            'alpha': client_step_state['alpha'],
        }
        return next_client_step_state

    def client_final(server_params, client_step_state):
        del server_params
        return client_step_state['grad']
    return for_each_client.for_each_client(
        client_init, client_step, client_final)


@dataclasses.dataclass
class ServerState:
    """State of server passed between rounds.
    Attributes:
        params: A pytree representing the server model parameters.
        opt_state: A pytree representing the server optimizer state.
      """
    params: Params
    opt_state: optimizers.OptState


def fedmix(
    grad_fn: Callable[[Params, BatchExample, PRNGKey], Grads],
    server_optimizer: optimizers.Optimizer,
    client_batch_hparams: client_datasets.ShuffleRepeatBatchHParams,
    plms: dict,
    alphas: dict
) -> federated_algorithm.FederatedAlgorithm:
    """Builds fedmix.
    Args:
        grad_fn: A function from (params, batch_example, rng) to gradients.
        This can be created with :func:`fedjax.core.model.model_grad`.
        client_optimizer: Optimizer for local client training.
        server_optimizer: Optimizer for server update.
        client_batch_hparams: Hyperparameters for batching client dataset.
        plms: A dictionary of PyTrees with pure local models for each client.
        alphas: A dictionary (client_dataset.client_id -> int) with mapping of
        clients to personalization parameter 'alpha'
      Returns:
        FederatedAlgorithm
      """
    train_for_each_client = create_train_for_each_client(grad_fn)

    def init(params: Params) -> ServerState:
        opt_state = server_optimizer.init(params)
        return ServerState(params, opt_state)

    def apply(
        server_state: ServerState,
        clients: Sequence[Tuple[federated_data.ClientId,
                                client_datasets.ClientDataset, PRNGKey]]
    ) -> Tuple[ServerState, Mapping[federated_data.ClientId, Any]]:
        batch_clients = [
            (cid, cds.shuffle_repeat_batch(client_batch_hparams),
             {'alpha': alphas[cid], 'plm': plms[cid], 'rng': crng})
            for cid, cds, crng in clients
        ]
        num_clients = len(clients)
        client_diagnostics = {}
        full_grad = tree_util.tree_zeros_like(server_state.params)
        for client_id, grad in train_for_each_client(
                server_state.params, batch_clients):
            full_grad = tree_util.tree_add(full_grad, grad)
            client_diagnostics[client_id] = {
                'delta_l2_norm': tree_util.tree_l2_norm(grad)
            }
        full_grad = tree_util.tree_inverse_weight(full_grad, num_clients)
        server_state = server_update(server_state, full_grad)
        return server_state, client_diagnostics

    def server_update(server_state, grad) -> ServerState:
        opt_state, params = server_optimizer.apply(grad,
                                                   server_state.opt_state,
                                                   server_state.params)
        return ServerState(params, opt_state)

    return federated_algorithm.FederatedAlgorithm(init, apply)