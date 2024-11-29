import torch


def get_optimizer(config, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.TRAINING.LR)

    return optimizer
