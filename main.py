import os
import pandas as pd

from model.MutiLabelModel import MutiLabelModel
from model.dataloader.dataloader import get_technique_train_dataloader, get_tactic_train_dataloader
from model.trainer.baseTrainer import BaseTrainer
from model.trainer.modifiedAns import ModifyTeTaAns
from model_cofig.config import get_params
from process_model.enum.ModelTypeEnum import ModelTypeEnum
from ttp_config.config import TACTIC, TECHNIQUE
from utils.process_util import preprocess
import pickle


def get_source_data():
    df_tram = pd.read_csv('./data/tram_with_all_labels.csv', encoding='utf-8')
    df_attack = pd.read_csv('./data/attack_with_all_labels.csv', encoding='utf-8')
    pickle.dump(df_tram, open('./data/tram_with_all_labels.pt', "wb"))
    pickle.dump(df_attack, open('./data/attack_with_all_labels.pt', "wb"))
    df_tram['tactic_label'] = df_tram.apply(lambda x: list(x[TACTIC]), axis=1)
    df_tram['technique_label'] = df_tram.apply(lambda x: list(x[TECHNIQUE]), axis=1)
    df_attack['tactic_label'] = df_attack.apply(lambda x: list(x[TACTIC]), axis=1)
    df_attack['technique_label'] = df_attack.apply(lambda x: list(x[TECHNIQUE]), axis=1)
    df = pd.concat([df_tram, df_attack], ignore_index=True)
    df['text_clean'] = df['text'].map(lambda t: preprocess(t))  # 把数据的text字段进行了ioc保护
    pickle.dump(df_attack, open('./data/df.pt', "wb"))
    return df

def train_tactic_main():
    params = get_params()
    df = pickle.load(open("./data/df.pt", "rb"))
    train_loader, valid_loader, test_loader = get_tactic_train_dataloader(df)
    model = MutiLabelModel(params, params.tactic_n_class, ModelTypeEnum.TACTIC.get_enum_name())
    model.cuda()
    trainer = BaseTrainer(params, model)
    trainer.train_model(train_loader, valid_loader, test_loader)

def train_technique_main():
    params = get_params()
    df = pickle.load(open("./data/df.pt", "rb"))
    train_loader, valid_loader, test_loader = get_technique_train_dataloader(df)
    model = MutiLabelModel(params, params.technique_n_class, ModelTypeEnum.TECHNIQUE.get_enum_name())
    model.cuda()
    trainer = BaseTrainer(params, model)
    trainer.train_model(train_loader, valid_loader, test_loader)

def do_modify():
    params = get_params()
    df = pickle.load(open("./data/df.pt", "rb"))
    tactic_train_loader, tactic_valid_loader, tactic_test_loader = get_tactic_train_dataloader(df)
    technique_train_loader, technique_valid_loader, technique_test_loader = get_technique_train_dataloader(df)
    m_obj = ModifyTeTaAns(params)
    ta_model_outputs, ta_true = m_obj.evaluate_tactic_data(tactic_train_loader)
    te_model_outputs, te_true = m_obj.evaluate_technique_data(technique_train_loader)
    m_obj.do_modify(ta_model_outputs, ta_true, te_model_outputs, te_true)


if __name__ == '__main__':
    # train_technique_main()
    # train_tactic_main()
    do_modify()