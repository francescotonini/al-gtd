from .GTN import GTN


def get_model(config, device):
    model = GTN()
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total params: {total_params}")
    print(f"Total trainable params: {total_trainable_params}")

    return model
