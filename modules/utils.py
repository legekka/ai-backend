import torch

def load_checkpoint(model, path, device='cpu'):
    checkpoint_dict = torch.load(path)
    model.load_state_dict(checkpoint_dict['model'])
    model.to(device)
    return model
