from fedjax.core import federated_data
import jax.numpy as jnp


def grid_search(
    train_fd: federated_data.FederatedData,
    validation_fd: federated_data.FederatedData,
    num_clients: int,
    server_lrs: jnp.array,
    client_lrs: jnp.array,
    fedmix_batch_sizes: list,
    plm_batch_sizes: list,
    num_epochs_plm: int,
    max_plm_rounds: int,
    max_fedmix_rounds: int,
) -> jnp.array:
    # client_ids = train_fd.client_ids()[:num_clients]
    # for b_id, batch_size in enumerate(plm_batch_sizes):
    #     print('Batch size = {}'.format(batch_size))
    #     for lr_id, lr in enumerate(client_lrs):
    #         print('Learning rate = {}'.format(lr))
    return NotImplementedError