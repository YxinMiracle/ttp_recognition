import logging
import os
import pathlib
from datetime import datetime
from typing import Optional, Mapping, List
from collections import deque
import numpy as np
from scipy.sparse import csr_matrix

import torch
from torch_geometric.data import DataLoader
from tqdm import tqdm
import torch.nn as nn


from model.optimizers.optimizers import DenseSparseAdam
from model.trainer.evaluation import get_p_5, get_n_5

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseTrainer:
    def __init__(self, params, model, gradient_clip_value=10.0):
        self.root_dir = str(pathlib.Path(__file__).resolve().parent.parent.parent)
        self.params = params
        self._init_log_file()  # 初始化日志信息
        self.model = model
        self.optimizer = None
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.model_path = self._init_model_path()
        self.state = {}
        self.gradient_clip_value, self.gradient_norm_queue = gradient_clip_value, deque([np.inf], maxlen=5)


    def _init_model_path(self):
        saved_model_path = self.root_dir + os.path.sep + self.params.save_directory_name
        if not os.path.exists(saved_model_path):
            os.makedirs(saved_model_path)
        current_datetime = datetime.now()
        output_path_name = current_datetime.strftime('%Y-%m-%d-%H-%M-%S')  # type: str
        return saved_model_path + os.path.sep + output_path_name + ".pt"

    def get_optimizer(self, **kwargs):
        self.optimizer = DenseSparseAdam(self.model.parameters(), **kwargs)

    def _init_log_file(self):
        log_save_path = self.root_dir + os.path.sep + self.params.log_directory_name
        if not os.path.exists(log_save_path):
            os.makedirs(log_save_path)
        current_datetime = datetime.now()
        output_path_name = current_datetime.strftime('%Y-%m-%d-%H-%M-%S')  # type: str
        logger.addHandler(
            logging.FileHandler(log_save_path + os.path.sep + output_path_name + ".log", 'w'))
        logger.info(self.params)

    def _can_save_model(self, epoch: int, average_f1: float, best_result: float) -> bool:
        """ 看看当前情况下是否可以保存模型，如果可以的话就返回True """
        if epoch == 0 or average_f1 > best_result:
            torch.save(self.model.state_dict(), self.model_path)
            logger.info("Best result on dev saved!!!")
            return True
        return False

    def swa_init(self):
        if 'swa' not in self.state:
            logger.info('SWA Initializing')
            swa_state = self.state['swa'] = {'models_num': 1}
            for n, p in self.model.named_parameters():
                swa_state[n] = p.data.cpu().detach()

    def swa_step(self):
        if 'swa' in self.state:
            swa_state = self.state['swa']
            swa_state['models_num'] += 1
            beta = 1.0 / swa_state['models_num']
            with torch.no_grad():
                for n, p in self.model.named_parameters():
                    swa_state[n].mul_(1.0 - beta).add_(beta, p.data.cpu())

    def swap_swa_params(self):
        if 'swa' in self.state:
            swa_state = self.state['swa']
            for n, p in self.model.named_parameters():
                p.data, swa_state[n] = swa_state[n].cuda(), p.data.cpu()

    def disable_swa(self):
        if 'swa' in self.state:
            del self.state['swa']

    def clip_gradient(self):
        if self.gradient_clip_value is not None:
            max_norm = max(self.gradient_norm_queue)
            total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm * self.gradient_clip_value)
            self.gradient_norm_queue.append(min(total_norm, max_norm * 2.0, 1.0))
            if total_norm > max_norm * self.gradient_clip_value:
                logger.warn(F'Clipping gradients with total norm {round(total_norm, 5)} '
                            F'and max norm {round(max_norm, 5)}')

    def save_model(self, last_epoch):
        if not last_epoch: return
        for trial in range(5):
            try:
                torch.save(self.model.module.state_dict(), self.model_path)
                break
            except:
                print('saving failed')

    def train_step(self, train_x: torch.Tensor, train_y: torch.Tensor):
        self.optimizer.zero_grad()
        self.model.train()
        scores = self.model(train_x)
        loss = self.loss_fn(scores, train_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.params.clip)
        self.optimizer.step(closure=None)
        return loss.item()


    def train_model(self, train_loader: DataLoader, valid_loader: DataLoader, test_loader: DataLoader,
                    opt_params: Optional[Mapping] = None, verbose=True, early=100,  k=5):
        self.get_optimizer(**({} if opt_params is None else opt_params))
        global_step, best_n5, e = 0, 0.0, 0
        for epoch_idx in range(self.params.epoch):
            pbar = tqdm(enumerate(train_loader), total=len(train_loader))
            loss_list = []
            for i ,(train_x, train_y) in pbar:
                loss = self.train_step(train_x, train_y.cuda())
                loss_list.append(loss)
                pbar.set_description("(Epoch {}) LOSS:{:.4f}".format(epoch_idx, np.mean(loss_list)))

            labels = []
            valid_loss = 0.0
            self.model.eval()
            with torch.no_grad():
                for (valid_x, valid_y) in valid_loader:
                    logits = self.model(valid_x)
                    valid_loss += self.loss_fn(logits, valid_y.cuda()).item()
                    scores, tmp = torch.topk(logits, k)
                    labels.append(tmp.cpu())
            valid_loss /= len(valid_loader)
            # labels = np.concatenate(labels)
            # targets = csr_matrix(valid_loader.dataset.data_y)

            log_msg = '%d train loss: %.7f valid loss: %.7f ' % \
                      (epoch_idx, np.mean(loss_list), valid_loss)
            logger.info(log_msg)

