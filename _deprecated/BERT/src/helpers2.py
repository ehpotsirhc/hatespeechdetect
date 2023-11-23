#!/usr/bin/env python3

# BERT Classifier Helper Functions
# Christophe Leung
# October 30, 2023

# The seed_everything() function has been adapted from the 
# assignment-provided code from CS678.
# Tested on Python 3.10.7; should be compatible between v3.6 v3.11

# -----------------------------------------------------------------------------
from pathlib import Path
import numpy as np, torch, random, json
from config import Constants

# -----------------------------------------------------------------------------
# Helper Functions
class Helpers2:
    @staticmethod
    def json_load(fpath: str):
        with open(Path(fpath), 'r') as f:
            return json.load(f)

    @staticmethod
    def separator(n=80, **kwargs):
        msg = kwargs['msg'] if 'msg' in kwargs else None
        print('%s\n%s' % ('-'*n, msg)) if msg else print('-'*n)

    @staticmethod
    def id2label(id: int):
        # return list(Constants.LABEL2ID.items())[int(id)-1]
        return (Constants.LABELS[id], id)

    @staticmethod
    def seed_everything(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    @staticmethod
    def gpu_init(**kwargs):
        showmsg = kwargs['showmsg'] if 'showmsg' in kwargs else False
        torch.cuda.empty_cache()
        assert torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name()
        n_gpu = torch.cuda.device_count()
        gpu_device = torch.device('cuda')
        print(f'Found device: {gpu_name}, n_gpu: {n_gpu}\n') if showmsg is True else None
        return (gpu_device, gpu_name, n_gpu)

