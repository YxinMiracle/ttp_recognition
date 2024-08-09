from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import numpy as np


class MultiLabelDataset(Dataset):
    def __init__(self, data_x, data_y=None):
        self.data_x, self.data_y = data_x, data_y

    def __getitem__(self, item):
        data_x = self.data_x[item]
        if self.data_y is not None:
            data_y = np.array(self.data_y[item]).astype(np.float32)
            return data_x, data_y
        else:
            return data_x

    def __len__(self):
        return len(self.data_x)


def get_train_dataloader(data):
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    validation, test = train_test_split(test, test_size=0.5, random_state=42)

    df_te_train = data.iloc[train.index][['text_clean', 'technique_label']]  # 划分训练集数据
    df_te_val = data.iloc[validation.index][['text_clean', 'technique_label']]  # 划分测验证集数据
    df_te_test = data.iloc[test.index][['text_clean', 'technique_label']]  # 划分测试集数据
    df_te_train = df_te_train.rename(columns={'text_clean': 'text', 'technique_label': 'labels'})  # 改名字
    df_te_val = df_te_val.rename(columns={'text_clean': 'text', 'technique_label': 'labels'})  # 改名字
    df_te_test = df_te_test.rename(columns={'text_clean': 'text', 'technique_label': 'labels'})  # 改名字

    train_dataset = MultiLabelDataset(df_te_train['text'].tolist(), df_te_train['labels'].tolist())
    val_dataset = MultiLabelDataset(df_te_val['text'].tolist(), df_te_val['labels'].tolist() )
    test_dataset = MultiLabelDataset(df_te_test['text'].tolist(), df_te_test['labels'].tolist())
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True,
                              drop_last=True)
    valid_loader = DataLoader(val_dataset, batch_size=64, shuffle=True, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, pin_memory=True, drop_last=True)

    return train_loader, valid_loader, test_loader
