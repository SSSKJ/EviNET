import torch

def gamma(x: torch.Tensor):

    return torch.lgamma(x).exp()

import os
import logging
from logging import handlers


def get_logger(LOG_ROOT, log_filename, level=logging.INFO, when='D', back_count=0):

    logger = logging.getLogger(log_filename)
    logger.setLevel(level)
    log_path = os.path.join(LOG_ROOT, "logs")
    os.makedirs(log_path, exist_ok=True)
    log_file_path = os.path.join(log_path, log_filename)
    formatter = logging.Formatter('%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(level)
    fh = handlers.TimedRotatingFileHandler(
        filename=log_file_path,
        when=when,
        backupCount=back_count,
        encoding='utf-8')
    fh.setLevel(level)
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

from torch.utils.tensorboard import SummaryWriter

class tb_handler:

    def __init__(self, path: str, name: str, model) -> None:

        if path[-1] != '/':
            path += '/'

        if not os.path.exists(path):
            os.mkdir(path)
        
        self.writer = SummaryWriter(f'{path}{name}')
        self.model = model

    def add_image(self, image, name):

        self.writer.add_image(name, image)
        self.writer.flush()

    def add_graph(self, input_data):
        self.writer.add_graph(self.model, input_data)
        self.writer.flush()

    def add_scalar(self, data, step, tag):
        self.writer.add_scalar(tag = tag, scalar_value = data, global_step = step)
        self.writer.flush()

    def show_params(self, global_step, mode = ['weight', 'grad']):

        for name, param in self.model.named_parameters():
            # print(f'{name}: {param.grad}')
            if param.grad is None:
                continue
            if 'weight' in mode:
                self.add_histogram(param, global_step, name + '_weight')
            if 'grad' in mode:
                self.add_histogram(param.grad, global_step, name + '_grad')
        self.writer.flush()


    def add_histogram(self, weight, global_step, name):
        self.writer.add_histogram(tag = name, values = weight, global_step = global_step)
        self.writer.flush()