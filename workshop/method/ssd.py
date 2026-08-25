import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, Dataset
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import time
import copy
import os
import pdb
import math
import shutil
from tqdm import tqdm
import seaborn as sns
import scipy.stats as stats
from typing import Dict, List
import copy
import torch.nn.functional as F
import itertools
from collections import OrderedDict

from .utils import keys, eval_opt, plot_unlearn_remain_acc_figure, evaluate_model_on_all_loaders
from utils import *
from trainer import *
import log_utils
import tqdm

class ParameterPerturber:
    def __init__(self, model, opt, device, parameters=None):    
        self.model = model
        self.opt = opt
        self.device = device
        self.alpha = None
        self.xmin = None

        print(parameters)
        self.lower_bound = parameters["lower_bound"]
        self.exponent = parameters["exponent"]
        self.magnitude_diff = parameters["magnitude_diff"]
        self.min_layer = parameters["min_layer"]
        self.max_layer = parameters["max_layer"]    
        self.forget_threshold = parameters["forget_threshold"]
        self.dampening_constant = parameters["dampening_constant"]
        self.selective_weighting = parameters["selective_weighting"]
    
    def get_layer_num(self, layer_name: str) -> int:
        layer_id = layer_name.split('.')[1]
        if layer_id.isdigit():
            return int(layer_id)
        else:
            raise ValueError("Invalid layer name")
    
    def zero_like_params_dict(self, model:torch.nn)->Dict[str, torch.Tensor]:
        return dict(
            [
                (k, torch.zeros_like(p, device=self.device)) for k, p in model.named_parameters()
            ]
        )
    
    def full_like_params_dict(
        self, model:torch.nn, fill_value, as_tensor:bool = False
    )->Dict[str, torch.Tensor]:

        def full_like_tensor(fillval, shape:list)->list:
            if len(shape) > 1:
                fillval = full_like_tensor(fillval, shape[1:])
            tmp = [fillval for _ in range(shape[0])]
            return tmp
    
        dictionary = {}

        for n, p in model.named_parameters():
            _p = (
                torch.tensor(full_like_tensor(fill_value, p.shape), device=self.device)
                if as_tensor
                else full_like_tensor(fill_value, p.shape)
            )
            dictionary[n] = _p

        return dictionary
    
    def calc_importance(self, dataloader: DataLoader)->Dict[str, torch.Tensor]:
        criterion = nn.CrossEntropyLoss()
        importance_dict = self.zero_like_params_dict(self.model)

        for batch in dataloader:
            x, y = batch
            x, y = x.to(self.device), y.to(self.device)
            self.opt.zero_grad()
            out = self.model(x)
            loss = criterion(out, y)
            loss.backward()

            for (k1, p), (k2, imp) in zip(
                self.model.named_parameters(), importance_dict.items()
            ):
                if p.grad is not None:
                    imp.data += p.grad.data.clone().pow(2)
            
        for _, imp in importance_dict.items():
            imp.data /= float(len(dataloader))
        return importance_dict
    
    def modify_weight(
        self,
        original_importance: List[Dict[str, torch.Tensor]],
        forget_importance: List[Dict[str, torch.Tensor]],
    ) -> None:
        """
        Perturb weights based on the SSD equations given in the paper
        Parameters:
        original_importance (List[Dict[str, torch.Tensor]]): list of importances for original dataset
        forget_importance (List[Dict[str, torch.Tensor]]): list of importances for forget sample
        threshold (float): value to multiply original imp by to determine memorization.

        Returns:
        None

        """

       
        total_selected = 0
        total_params = 0

        with torch.no_grad():
            for (n, p), (oimp_n, oimp), (fimp_n, fimp) in zip(
                self.model.named_parameters(),
                original_importance.items(),
                forget_importance.items(),
            ):
                # Synapse Selection with parameter alpha
                oimp_norm = oimp.mul(self.selective_weighting)
                locations = torch.where(fimp > oimp_norm)

                selected = locations[0].numel()
                total = p.numel()

                total_selected += selected
                total_params += total

                if selected == 0:
                    print(f"[SSD] {n}: selected=0/{total}")
                    continue

                print(f"[SSD] {n}: selected={selected}/{total} ({selected / total:.6%})")

                # Synapse Dampening with parameter lambda
                weight = ((oimp.mul(self.dampening_constant)).div(fimp)).pow(
                    self.exponent
                )
                update = weight[locations]

                # Bound by 1 to prevent parameter values to increase.
                min_locs = torch.where(update > self.lower_bound)
                update[min_locs] = self.lower_bound

                p[locations] = p[locations].mul(update)

        print(
            f"[SSD TOTAL] selected={total_selected}/{total_params} "
            f"({total_selected / total_params:.6%})"
        )

def ssd(
    model,
    full_train_dl=None,
    retain_train_dl=None,
    forget_train_dl=None,
    retain_valid_dl=None,
    forget_valid_dl=None,
    dampening_constant=1,
    selection_weighting=10,
    loader_dict=None,
    device="cuda",
    logger=None,
    console_handler=None,
    eval_opt=eval_opt,
    
):
    if logger is not None:
        logger.info("unlearn_epoch : ssd, unlearn_rate : ssd")
        logger.info(f"eval option : {eval_opt}")
        

    parameters = {
        "lower_bound": 1.0,
        "exponent": 1.0,
        "magnitude_diff": None,
        "min_layer": -1,
        "max_layer": -1,
        "forget_threshold": 1.0,
        "dampening_constant": dampening_constant,
        "selective_weighting": selection_weighting,
    }
    logger.info(f"parameters : {parameters}")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    pdr = ParameterPerturber(model, optimizer, device, parameters)

    model.eval()
    sample_importance = pdr.calc_importance(forget_train_dl)
    original_importance = pdr.calc_importance(full_train_dl)
    pdr.modify_weight(original_importance, sample_importance)

    if loader_dict is not None:
        cur_accs_dict = evaluate_model_on_all_loaders(model, loader_dict, eval_opt, logger)

    if logger is not None and console_handler is not None:
        log_utils.enable_console_logging(logger, console_handler, True)

    return model