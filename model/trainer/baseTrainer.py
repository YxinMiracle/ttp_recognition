import json
import logging
import os
import pathlib
from datetime import datetime
from typing import Optional, Mapping, List
from collections import deque
import numpy as np
from sklearn.metrics import coverage_error, label_ranking_average_precision_score, label_ranking_loss
from sklearn.metrics import hamming_loss, precision_score, recall_score, f1_score, fbeta_score
from sklearn.metrics import accuracy_score
import torch
from torch_geometric.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
from pprint import pprint, pformat

from model.optimizers.optimizers import DenseSparseAdam
from process_model.enum.EvaluateTypeEnum import EvaluateTypeEnum
from process_model.enum.TrainTypeEnum import TrainTypeEnum

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseTrainer:
    def __init__(self, params, model, gradient_clip_value=10.0):
        self.root_dir = str(pathlib.Path(__file__).resolve().parent.parent.parent)
        self.params = params
        self.model = model
        self._init_log_file()  # 初始化日志信息
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.model_path = self._init_model_path()
        self.state = {}
        self.gradient_clip_value, self.gradient_norm_queue = gradient_clip_value, deque([np.inf], maxlen=5)
        self.optimizer = DenseSparseAdam(model.parameters(), lr=self.params.lr, weight_decay=params.weight_decay)

    def _init_model_path(self):
        saved_model_path = self.root_dir + os.path.sep + self.params.save_directory_name + os.path.sep + self.model.model_type
        if not os.path.exists(saved_model_path):
            os.makedirs(saved_model_path)
        current_datetime = datetime.now()
        output_path_name = current_datetime.strftime('%Y-%m-%d-%H-%M-%S')  # type: str
        return saved_model_path + os.path.sep + output_path_name + ".pt"

    def _init_log_file(self):
        log_save_path = self.root_dir + os.path.sep + self.params.log_directory_name + os.path.sep + self.model.model_type
        if not os.path.exists(log_save_path):
            os.makedirs(log_save_path)
        current_datetime = datetime.now()
        output_path_name = current_datetime.strftime('%Y-%m-%d-%H-%M-%S')  # type: str
        logger.addHandler(
            logging.FileHandler(log_save_path + os.path.sep + output_path_name + ".log", 'w'))
        logger.info(self.params)

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

    def _can_save_model(self, epoch: int, f1: float, best_result: float) -> bool:
        """ 看看当前情况下是否可以保存模型，如果可以的话就返回True """
        if epoch == 0 or f1 > best_result:
            torch.save(self.model.state_dict(), self.model_path)
            logger.info("Best result on dev saved!!!")
            return True
        return False

    def train_step(self, train_x: torch.Tensor, train_y: torch.Tensor):
        self.optimizer.zero_grad()
        self.model.train()
        scores = self.model(train_x)
        loss = self.loss_fn(scores, train_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.params.clip)
        self.optimizer.step(closure=None)
        return loss.item()

    def evaluate_classification_metrics(self, evaluate_res_list, evaluate_label_list):
        evaluate_res_list_bi = (evaluate_res_list >= 0.5).astype(int)
        results = {}
        results[EvaluateTypeEnum.Coverage_Error.get_enum_name()] = coverage_error(evaluate_label_list, evaluate_res_list)
        results[EvaluateTypeEnum.LRAP.get_enum_name()] = label_ranking_average_precision_score(evaluate_label_list, evaluate_res_list)
        results[EvaluateTypeEnum.Label_Ranking_Loss.get_enum_name()] = label_ranking_loss(evaluate_label_list, evaluate_res_list)
        results[EvaluateTypeEnum.Hamming_Loss.get_enum_name()] = hamming_loss(evaluate_label_list, evaluate_res_list_bi)
        results[EvaluateTypeEnum.Precision_Score_Samples.get_enum_name()] = precision_score(evaluate_label_list, evaluate_res_list_bi, average='samples', zero_division=0)
        results[EvaluateTypeEnum.Precision_Score_Macro.get_enum_name()] = precision_score(evaluate_label_list, evaluate_res_list_bi, average='macro', zero_division=0)
        results[EvaluateTypeEnum.Precision_Score_Micro.get_enum_name()] = precision_score(evaluate_label_list, evaluate_res_list_bi, average='micro', zero_division=0)
        results[EvaluateTypeEnum.Recall_Score_Samples.get_enum_name()] = recall_score(evaluate_label_list, evaluate_res_list_bi, average='samples', zero_division=0)
        results[EvaluateTypeEnum.Recall_Score_Macro.get_enum_name()] = recall_score(evaluate_label_list, evaluate_res_list_bi, average='macro', zero_division=0)
        results[EvaluateTypeEnum.Recall_Score_Micro.get_enum_name()] = recall_score(evaluate_label_list, evaluate_res_list_bi, average='micro', zero_division=0)
        results[EvaluateTypeEnum.F1_Score_Samples.get_enum_name()] = f1_score(evaluate_label_list, evaluate_res_list_bi, average='samples', zero_division=0)
        results[EvaluateTypeEnum.F1_Score_Macro.get_enum_name()] = f1_score(evaluate_label_list, evaluate_res_list_bi, average='macro', zero_division=0)
        results[EvaluateTypeEnum.F1_Score_Micro.get_enum_name()] = f1_score(evaluate_label_list, evaluate_res_list_bi, average='micro', zero_division=0)
        results[EvaluateTypeEnum.F05_Score_Samples.get_enum_name()] = fbeta_score(evaluate_label_list, evaluate_res_list_bi, beta=0.5, average='samples', zero_division=0)
        results[EvaluateTypeEnum.F05_Score_Macro.get_enum_name()] = fbeta_score(evaluate_label_list, evaluate_res_list_bi, beta=0.5, average='macro', zero_division=0)
        results[EvaluateTypeEnum.F05_Score_Micro.get_enum_name()] = fbeta_score(evaluate_label_list, evaluate_res_list_bi, beta=0.5, average='micro', zero_division=0)
        results[EvaluateTypeEnum.Accuracy_Score.get_enum_name()] = accuracy_score(evaluate_label_list, evaluate_res_list_bi)
        return results

    def evaluate(self, test_batch: DataLoader, test_or_dev: str):
        evaluate_res_list, evaluate_label_list, evaluate_loss_list = [], [], []
        self.model.eval()
        with torch.no_grad():
            for (valid_x, valid_y) in test_batch:
                model_output = self.model(valid_x)
                evaluate_loss_list.append(self.loss_fn(model_output, valid_y.cuda()).item())
                evaluate_label_list.extend(valid_y.data.cpu().numpy())
                evaluate_res_list.extend(model_output.data.cpu().numpy())
            evaluate_res_list = np.array(evaluate_res_list)
            evaluate_label_list = np.array(evaluate_label_list)
            results = self.evaluate_classification_metrics(evaluate_res_list, evaluate_label_list)
            logger.info("------ {} Results ------".format(test_or_dev))
            logger.info("loss : {:.4f}".format(np.mean(evaluate_loss_list)))
            logger.info(pformat(results))
        return results

    def train_model(self, train_loader: DataLoader, valid_loader: DataLoader, test_loader: DataLoader,
                    verbose=True, early=8, k=5):
        best_f05, e = 0, 0
        for epoch_idx in range(self.params.epoch):
            pbar = tqdm(enumerate(train_loader), total=len(train_loader))
            loss_list = []
            for i, (train_x, train_y) in pbar:
                loss = self.train_step(train_x, train_y.cuda())
                loss_list.append(loss)
                pbar.set_description("(Epoch {}) LOSS:{:.4f}".format(epoch_idx, np.mean(loss_list)))
            logger.info("------ Training Set Results ------")
            logger.info("Finish training epoch %d. loss: %.4f" % (epoch_idx, np.mean(loss_list)))
            logger.info("------ Start Evaluate ------")
            evaluate_result = self.evaluate(valid_loader, TrainTypeEnum.VAL_TYPE.get_enum_name())
            f05_score = evaluate_result[EvaluateTypeEnum.F05_Score_Samples.get_enum_name()]
            if self._can_save_model(epoch_idx, f05_score, best_f05):
                best_f05, e = f05_score, 0
            else:
                e += 1
                if early is not None and e > early:
                    logger.info("------ Finish training ------")
                    logger.info("------ Start Test ------")
                    self.evaluate(test_loader, TrainTypeEnum.TEST_TYPE.get_enum_name())
                    return
        logger.info("------ Finish training ------")
        logger.info("------ Start Test ------")
        self.evaluate(test_loader, TrainTypeEnum.TEST_TYPE.get_enum_name())
