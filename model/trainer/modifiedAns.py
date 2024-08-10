import os.path
import pathlib
import torch

from model.MutiLabelModel import MutiLabelModel
from process_model.enum.ModelTypeEnum import ModelTypeEnum
from ttp_config.config import TACTICS_TECHNIQUES_RELATIONSHIP_DF, TACTIC, TECHNIQUE

from sklearn.metrics import coverage_error, label_ranking_average_precision_score, label_ranking_loss
from sklearn.metrics import hamming_loss, precision_score, recall_score, f1_score, fbeta_score
from sklearn.metrics import accuracy_score
import numpy as np


class ModifyTeTaAns:
    def __init__(self, params):
        self.params = params
        self.root_dir = str(pathlib.Path(__file__).resolve().parent.parent.parent)
        self.technique_model_path = self.root_dir + os.path.sep + self.params.save_directory_name + os.path.sep + ModelTypeEnum.TECHNIQUE.get_enum_name() + os.path.sep + '2024-08-10-13-40-43.pt'
        self.tactic_model_path = self.root_dir + os.path.sep + self.params.save_directory_name + os.path.sep + ModelTypeEnum.TACTIC.get_enum_name() + os.path.sep + '2024-08-10-19-26-08.pt'
        self.technique_model = self._init_technique_model()
        self.tactic_model = self._init_tactic_model()

    def _init_technique_model(self):
        model = MutiLabelModel(self.params, self.params.technique_n_class, ModelTypeEnum.TECHNIQUE.get_enum_name())
        model.load_state_dict(torch.load(self.technique_model_path))
        model.cuda()
        return model

    def _init_tactic_model(self):
        model = MutiLabelModel(self.params, self.params.tactic_n_class, ModelTypeEnum.TACTIC.get_enum_name())
        model.load_state_dict(torch.load(self.tactic_model_path))
        model.cuda()
        return model

    def evaluate_technique_data(self, test_batch):
        evaluate_res_list, evaluate_label_list = [], []
        self.technique_model.eval()
        with torch.no_grad():
            for (valid_x, valid_y) in test_batch:
                model_output = self.technique_model(valid_x)
                evaluate_label_list.extend(valid_y.data.cpu().numpy())
                evaluate_res_list.extend(model_output.data.cpu().numpy())
            evaluate_res_list = np.array(evaluate_res_list)
            evaluate_label_list = np.array(evaluate_label_list)
        return evaluate_res_list, evaluate_label_list

    def evaluate_tactic_data(self, test_batch):
        evaluate_res_list, evaluate_label_list = [], []
        self.tactic_model.eval()
        with torch.no_grad():
            for (valid_x, valid_y) in test_batch:
                model_output = self.tactic_model(valid_x)
                evaluate_label_list.extend(valid_y.data.cpu().numpy())
                evaluate_res_list.extend(model_output.data.cpu().numpy())
            evaluate_res_list = np.array(evaluate_res_list)
            evaluate_label_list = np.array(evaluate_label_list)
        return evaluate_res_list, evaluate_label_list

    def do_modify(self, ta_model_outputs, ta_true, te_model_outputs, te_true):

        # Post-processing
        print('Post-processing-----------------------------------------------------')
        ta_correct_true = {}
        ta_correct_false = {}
        sub_correct_true = {}
        sub_correct_false = {}
        highrate_correct_true = {}
        highrate_correct_false = {}
        all_true_mod = 0
        all_false_mod = 0
        all_true = 0
        all_false = 0
        true_origin = 0
        false_origin = 0

        te_modified = []
        modified_ind = set()
        for ind in range(len(ta_model_outputs)):
            te_tmp = {}
            te_mask = {}
            te_pred = {}
            te_real = {}
            ta_real = {}
            ta_pred = {}

            ta_threshold = 0.01
            te_threshold = 0.25

            for i, v in enumerate(ta_model_outputs[ind]):  # ta_model_outputs[ind] 代表的是一句话中的数据， 其中的i表示的就是 第几个战术 v表示的是这个句子有多少概率
                ta_pred[TACTIC[i]] = v  # 获取这个句子中的这个战术对应的分数是什么
                for te in TACTICS_TECHNIQUES_RELATIONSHIP_DF[TACTIC[i]]:  # 然后去循环这个战术中的所有技术
                    try:  # 如果说这个战术预测出来的概率大于了阈值，
                        te_mask[te] |= int(v > ta_threshold)  # 无论原来的值是什么，结果都会是一
                    except KeyError:
                        te_mask[te] = int(v > ta_threshold)
            for te, v in list(zip(TECHNIQUE, te_model_outputs[ind])):
                te_pred[te] = v
            for te, v in list(zip(TECHNIQUE, te_true[ind])):
                te_real[te] = int(v)
            for ta, v in list(zip(TACTIC, ta_true[ind])):
                ta_real[ta] = int(v)

            tp = fp = tn = fn = 0
            tp_ = fp_ = tn_ = fn_ = 0
            for te in TECHNIQUE:
                # if te was set to True by sub-tech then pass
                try:
                    if te_tmp[te]:
                        continue
                except KeyError:
                    # Te prediction>0.95 then discard correction
                    if te_pred[te] > 0.95:  # 如果说战术预测的绝对高
                        te_mask[te] = 1  # 那么不管之前他所对应的战术识别出来的概率有没有超过这个阈值，都认为这个这个技术所对应的战术是正确的
                        if te_real[te]:  # 那么如果说是这个技术是真的正确的，也就是技术模型预测的没有错
                            try:
                                highrate_correct_true[te] += 1  # 在整个测试集中的这个技术预测正确的次数+1
                            except KeyError:
                                highrate_correct_true[te] = 1
                            # print(f'{ind}: {te}, te_real={te_real[te]} te_pred={te_pred[te]} high pred rate -> True')
                        else:
                            try:
                                highrate_correct_false[te] += 1  # 在整个测试集中，这个技术预测错误的次数+1
                            except KeyError:
                                highrate_correct_false[te] = 1
                            # print(f'{ind}: {te}, te_real={te_real[te]} te_pred={te_pred[te]} high pred rate -> False')
                    # te_tep 表示这个技术，在这个句子中，经过最后的矫正，他最终是识别正确还是错误
                    te_tmp[te] = int(te_pred[te] >= te_threshold) & te_mask[te]  # te mask 表示的是这个技术所对应的战术有没有识别出来，就是有没有大于某个阈值

                # Te set to True if te_pred>=threshold
                if te_pred[te] >= te_threshold:  # 如果说这个技术对应的预测值大于了他对应的阈值
                    if te_real[te]:  # 并且这个句子中真的包含了这个技术
                        tp += 1  # real=1 and pred=1 => TP
                        if te_mask[te]:
                            tp_ += 1  # real=1 and pred=1->1 => TP_

                            # Set parent technique to True if sub-tech is TP_
                            if len(te.split('.')) > 1:
                                te_parent = te.split('.')[0]
                                try:
                                    if te_tmp[te_parent]:
                                        pass
                                    else:
                                        raise KeyError
                                except KeyError:
                                    te_tmp[te_parent] = 1

                                    if te_real[te_parent]:
                                        if te_pred[te_parent] < te_threshold:
                                            try:
                                                sub_correct_true[te_parent] += 1
                                            except KeyError:
                                                sub_correct_true[te_parent] = 1
                                            status = 'True'
                                        else:
                                            status = 'Useless'
                                    else:
                                        try:
                                            sub_correct_false[te_parent] += 1
                                        except KeyError:
                                            sub_correct_false[te_parent] = 1
                                        status = 'False'
                                    if status != 'Useless':
                                        print(
                                            f'{ind}: {te_parent}: {te}, te_parent_real={te_real[te_parent]} te_parent_pred={te_pred[te_parent]}, te_real={te_real[te]} te_pred={te_pred[te]} -> {status}')

                        else:
                            fn_ += 1  # real=1 and pred=1->0 => FN_

                            for ta in TACTICS_TECHNIQUES_RELATIONSHIP_DF:
                                if te in TACTICS_TECHNIQUES_RELATIONSHIP_DF[ta].unique():
                                    print(
                                        f'{ind}: {ta}: {te}, ta_real={ta_real[ta]} ta_pred={ta_pred[ta]}, te_real={te_real[te]} te_mask={te_mask[te]} te_pred={te_pred[te]} -> false')
                                    break
                            try:
                                ta_correct_false[te] += 1
                            except KeyError:
                                ta_correct_false[te] = 1
                    else:
                        fp += 1  # real=0 and pred=1 => FP
                        if te_mask[te]:
                            fp_ += 1  # real=0 and pred=1->1 => FP_
                        else:
                            tn_ += 1  # real=0 and pred=1->0 => TN_

                            for ta in TACTICS_TECHNIQUES_RELATIONSHIP_DF:
                                if te in TACTICS_TECHNIQUES_RELATIONSHIP_DF[ta].unique():
                                    print(
                                        f'{ind}: {ta}: {te}, ta_real={ta_real[ta]} ta_pred={ta_pred[ta]}, te_real={te_real[te]} te_mask={te_mask[te]} te_pred={te_pred[te]} -> true')
                                    break
                            try:
                                ta_correct_true[te] += 1
                            except KeyError:
                                ta_correct_true[te] = 1

                # Te set to False if te_pred<threshold
                else:
                    if te_real[te]:
                        fn += 1  # real=1 and pred=0 => FN
                        fn_ += 1  # real=1 and pred=0->0 => FN_
                    else:
                        tn += 1  # real=0 and pred=0 => TN
                        tn_ += 1  # real=0 and pred=0->0 =>TN_

            true_mod = 0
            false_mod = 0
            true = 0
            false = 0
            true_ori = 0
            false_ori = 0
            for te in TECHNIQUE:
                if te_real[te]:
                    if te_pred[te] >= 0.5:
                        true_ori += 1
                        if te_tmp[te]:
                            true += 1
                        else:
                            false += 1
                            false_mod += 1
                    else:
                        false_ori += 1
                        if te_tmp[te]:
                            true += 1
                            true_mod += 1
                        else:
                            false += 1
                # else:
                #     if te_pred[te]>=0.5:
                #         false_ori += 1
                #         if te_tmp[te]:
                #             false += 1
                #         else:
                #             true += 1
                #             true_mod += 1
                #     else:
                #         true_ori += 1
                #         if te_tmp[te]:
                #             false += 1
                #             false_mod += 1
                #         else:
                #             true += 1

            all_true_mod += true_mod
            all_false_mod += false_mod
            all_true += true
            all_false += false
            true_origin += true_ori
            false_origin += false_ori
            # print(f'{ind}: true modified {true}, false modified {false}')

            te_tmp = [te_tmp[te] for te in TECHNIQUE]
            te_modified.append(te_tmp)
            # print([[tp, fn],     #        [fp, tn]], '\n',
            #      [[tp_, fn_],     #       [fp_, tn_]])

        te_model_outputs_bi = (te_model_outputs > 0.5).astype(np.int_)
        print('Hamming loss: %f -> %f' % (hamming_loss(te_true, te_model_outputs_bi), hamming_loss(te_true, te_modified)))
        print('Precision score(samples): %f -> %f' % (
            precision_score(te_true, te_model_outputs_bi, average='samples', zero_division=0),
            precision_score(te_true, te_modified, average='samples', zero_division=0)))
        print('Precision score(macro): %f -> %f' % (
            precision_score(te_true, te_model_outputs_bi, average='macro', zero_division=0),
            precision_score(te_true, te_modified, average='macro', zero_division=0)))
        print('Precision score(micro): %f -> %f' % (
            precision_score(te_true, te_model_outputs_bi, average='micro', zero_division=0),
            precision_score(te_true, te_modified, average='micro', zero_division=0)))

        print('Recall score(samples): %f -> %f' % (
            recall_score(te_true, te_model_outputs_bi, average='samples', zero_division=0),
            recall_score(te_true, te_modified, average='samples', zero_division=0)))
        print('Recall score(macro): %f -> %f' % (recall_score(te_true, te_model_outputs_bi, average='macro', zero_division=0),
                                                 recall_score(te_true, te_modified, average='macro', zero_division=0)))
        print('Recall score(micro): %f -> %f' % (recall_score(te_true, te_model_outputs_bi, average='micro', zero_division=0),
                                                 recall_score(te_true, te_modified, average='micro', zero_division=0)))

        print('F1 score(samples): %f -> %f' % (f1_score(te_true, te_model_outputs_bi, average='samples', zero_division=0),
                                               f1_score(te_true, te_modified, average='samples', zero_division=0)))
        print('F1 score(macro): %f -> %f' % (f1_score(te_true, te_model_outputs_bi, average='macro', zero_division=0),
                                             f1_score(te_true, te_modified, average='macro', zero_division=0)))
        print('F1 score(micro): %f -> %f' % (f1_score(te_true, te_model_outputs_bi, average='micro', zero_division=0),
                                             f1_score(te_true, te_modified, average='micro', zero_division=0)))

        print('F0.5 score(samples): %f -> %f' % (
            fbeta_score(te_true, te_model_outputs_bi, beta=0.5, average='samples', zero_division=0),
            fbeta_score(te_true, te_modified, beta=0.5, average='samples', zero_division=0)))
        print('F0.5 score(macro): %f -> %f' % (
            fbeta_score(te_true, te_model_outputs_bi, beta=0.5, average='macro', zero_division=0),
            fbeta_score(te_true, te_modified, beta=0.5, average='macro', zero_division=0)))
        print('F0.5 score(micro): %f -> %f' % (
            fbeta_score(te_true, te_model_outputs_bi, beta=0.5, average='micro', zero_division=0),
            fbeta_score(te_true, te_modified, beta=0.5, average='micro', zero_division=0)))

        print('Accuracy score: %f -> %f' % (accuracy_score(te_true, te_model_outputs_bi), accuracy_score(te_true, te_modified)))
