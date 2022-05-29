import os
import json
import logging
from shutil import copyfile
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import integrate
# from torch.utils.tensorboard import SummaryWriter # TODO : Tensorboard visualization

# User modules
import models.model as module_model
import models.loss as module_loss
import data_loaders.data_loaders as module_data_loader


def get_train_config(args):
    """
    Initialize training config, resume or new experiment.
    """
    if args.resume is not None:
        # Resume experiment
        resume_path = Path(args.resume)
        #####################################################
        #####################################################
        ####           TODO : Resume experiment          ####
        #####################################################
        #####################################################
        pass
    elif args.config is not None:
        # New experiment
        cfg_path = Path(args.config)
        config = read_json(cfg_path)

        # Create experiment instance
        config['save_dir'] = Path(config['save_dir']) / config['name'] / config['model']['type'] / 'train'
        ensure_dir(config['save_dir'])
        
        # Prepare devices
        config['device'], config['device_ids'] = prepare_device(config)

        # Load model
        config['model'] = _init_obj(config['model']['type'], module_model, **config['model']['args'])
        if config['n_gpu'] > 1:
            config['model'] = torch.nn.DataParallel(config['model'])
        config['model'] = config['model'].to(config['device'])
        config['trainable_params'] = filter(lambda p: p.requires_grad, config['model'].parameters())
        config['model_save_dir'] = config['save_dir'] / 'save_instances'
        ensure_dir(config['model_save_dir'])

        # Load dataloader
        config['data_loader'] = _init_obj(config['data_loader']['type'], module_data_loader, **config['data_loader']['args'])
        config['valid_loader'] = _init_obj(config['valid_loader']['type'], module_data_loader, **config['valid_loader']['args'])


        # Load metrics
        config['loss'] = _init_callable(config['loss']['type'], module_loss)

        # Load optimizer
        config['optimizer'] = _init_obj(config['optimizer']['type'], torch.optim, config['trainable_params'], **config['optimizer']['args'])
    
        # Load learning rate scheduler
        config['lr_scheduler'] = _init_obj(config['lr_scheduler']['type'], torch.optim.lr_scheduler, config['optimizer'], **config['lr_scheduler']['args'])

        

        config['config_file_path'] = config['save_dir'] / args.config.split('/')[-1]
        copyfile(args.config, config['config_file_path']) # Save config instance
    else:
        # No arguments
        msg_no_cfg = "Configuration file path is required. Add '-c config.json'"
        raise ValueError(msg_no_cfg)

    return config


def get_logger(config):
    # Logger instance
    logger = logging.getLogger(config["name"])
    logger.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")    

    # Console stream
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File stream
    file_handler = logging.FileHandler(filename=config["save_dir"] / "experiment.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def read_json(path):
    with path.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def ensure_dir(path):
    if not path.is_dir():
        path.mkdir(parents=True, exist_ok=False)


def prepare_device(config):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if config['n_gpu'] > 0 and n_gpu == 0:
        print(
            "Warning: There's no GPU available on this machine,"
            "training will be performed on CPU."
        )
        config['n_gpu'] = 0
    if config['n_gpu'] > n_gpu:
        print(
            f"Warning: The number of GPU's configured to use is {config['n_gpu']}, but only {n_gpu} are "
            "available on this machine."
        )
        config['n_gpu'] = n_gpu
    device = torch.device("cuda:0" if config['n_gpu'] > 0 else "cpu")
    list_ids = list(range(config['n_gpu']))
    return device, list_ids


def _init_obj(module_name, module, *args, **kwargs):
    return getattr(module, module_name)(*args, **kwargs)


def _init_metrics(cfg_sub_list, module_metric):
    return [getattr(module_metric, met) for met in cfg_sub_list]

def _init_callable(cfg_sub_str, module):
    return getattr(module, cfg_sub_str)


def roc_curve(df_metrics, save_dir):
    if os.path.isdir(save_dir) is None:
        raise ValueError("Invalid input(save_dir)")

    # roc curve
    sorted_index = np.argsort(df_metrics["false_positive_rate"])
    fpr_list_sorted = np.array(df_metrics["false_positive_rate"])[sorted_index]
    tpr_list_sorted = np.array(df_metrics["recall"])[sorted_index]
    roc_auc = integrate.trapz(y=tpr_list_sorted, x=fpr_list_sorted)
    plt.figure()
    lw = 2

    plt.plot(
        df_metrics["false_positive_rate"],
        df_metrics["recall"],
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.3f)" % roc_auc,
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic")
    plt.legend(loc="lower right")
    plt.savefig(save_dir, dpi=300)