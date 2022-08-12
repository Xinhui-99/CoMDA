from itertools import permutations, combinations
import torch


def create_domain_weight(source_domain_num):
    global_federated_matrix = [1 / (source_domain_num + 1)] * (source_domain_num + 1)
    return global_federated_matrix


def decentralized_training_strategy(communication_rounds, epoch_samples, batch_size, total_epochs):
    """
    Split one epoch into r rounds and perform model aggregation
    :param communication_rounds: the communication rounds in training process
    :param epoch_samples: the samples for each epoch
    :param batch_size: the batch_size for each epoch
    :param total_epochs: the total epochs for training
    :return: batch_per_epoch, total_epochs with communication rounds r
    """
    if communication_rounds >= 1:
        epoch_samples = round(epoch_samples / communication_rounds)
        total_epochs = round(total_epochs * communication_rounds)
        batch_per_epoch = round(epoch_samples / batch_size)
    elif communication_rounds in [0.1,0.2,0.25, 0.5]:
        total_epochs = round(total_epochs * communication_rounds)
        batch_per_epoch = round(epoch_samples / batch_size)
    else:
        raise NotImplementedError(
            "The communication round {} illegal, should be 0.2 or 0.5".format(communication_rounds))
    return batch_per_epoch, total_epochs



