import os
import random

import numpy as np
import torch


def setup_runtime(seed=0, cuda_dev_id=[0]):
    """Initialize CUDA, CuDNN and the random seeds."""
    # Setup CUDA
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if len(cuda_dev_id) == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_dev_id[0])
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_dev_id[0])
        for i in cuda_dev_id[1:]:
            os.environ["CUDA_VISIBLE_DEVICES"] += "," + str(i)

    # global cuda_dev_id
    _cuda_device_id = cuda_dev_id
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False 
    # Fix random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
