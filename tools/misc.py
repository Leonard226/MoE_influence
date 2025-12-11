## NOTE: The following code was not used in the experiments in the paper but is provided for reference.

import torch
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

def layer_print(model):
    """ Print layer info of the assigned model. """
    for k, v in model.state_dict().items():
        print(k, v.shape)
    print(model.config)