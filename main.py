import os
import pandas as pd

from model.MutiLabelModel import MutiLabelModel
from model.dataloader.dataloader import get_train_dataloader
from model.trainer.baseTrainer import BaseTrainer
from model_cofig.config import get_params
from ttp_config.config import TACTIC, TECHNIQUE
from utils.process_util import preprocess
import pickle


def main():
    # df_tram = pd.read_csv('./data/tram_with_all_labels.csv', encoding='utf-8')
    # df_attack = pd.read_csv('./data/attack_with_all_labels.csv', encoding='utf-8')
    # pickle.dump(df_tram, open('./data/tram_with_all_labels.pt', "wb"))
    # pickle.dump(df_attack, open('./data/attack_with_all_labels.pt', "wb"))
    #
    # df_tram['tactic_label'] = df_tram.apply(lambda x: list(x[TACTIC]), axis=1)
    # df_tram['technique_label'] = df_tram.apply(lambda x: list(x[TECHNIQUE]), axis=1)
    # df_attack['tactic_label'] = df_attack.apply(lambda x: list(x[TACTIC]), axis=1)
    # df_attack['technique_label'] = df_attack.apply(lambda x: list(x[TECHNIQUE]), axis=1)
    #
    # df = pd.concat([df_tram, df_attack], ignore_index=True)
    #
    # df['text_clean'] = df['text'].map(lambda t: preprocess(t))  # 把数据的text字段进行了ioc保护
    params = get_params()
    df = pickle.load(open("./data/df.pt","rb"))
    train_loader, valid_loader, test_loader = get_train_dataloader(df)
    model = MutiLabelModel(params, 576)
    model.cuda()
    trainer = BaseTrainer(params, model)
    trainer.train_model(train_loader, valid_loader, test_loader)



if __name__ == '__main__':
    main()