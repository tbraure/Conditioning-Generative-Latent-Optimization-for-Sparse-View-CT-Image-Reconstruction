import torch
import logging
from time import time

def timer(f):
    def wrapper(*args, **kwargs):
        start  = time()
        output = f(*args, **kwargs)
        end    = time()
        logging.info(f"{f.__name__}: {end-start:2f}s")
        return output

    return wrapper

def export(x, device, grad=True):
    if type(x) != torch.Tensor: x = torch.Tensor(x)
    return x.to(torch.float).to(device).requires_grad_(grad)

