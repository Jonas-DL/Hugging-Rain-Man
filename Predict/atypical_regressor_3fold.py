import logging
import os
import random
from collections import Counter
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchmetrics.functional import mean_absolute_error, mean_squared_error
from torchmetrics import MeanSquaredError, R2Score,MeanAbsolutePercentageError
from tqdm import tqdm
import torch
import torch.nn as nn

from abnormal_regression.au_ab_feat_LSTM import create_my_logger

"""
使用AU序列，回归异常等级
"""

class TransformerModel(nn.Module):
    def __init__(self, input_size, num_heads, num_layers, num_classes, dim_feedforward=512):
        super(TransformerModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=num_heads, dim_feedforward=dim_feedforward,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(num_layers, x.size(0), hidden_size).to(device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, batch_first=True, bidirectional=False,drop_out=0.5):
        super(LSTMModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=batch_first, bidirectional=bidirectional)
        # self.fc = nn.Linear(hidden_size, num_classes) # 单向
        self.dropout = nn.Dropout(p=drop_out) # 防止过拟合
        self.fc = nn.Linear(lstm_output_size, num_classes)

    def forward(self, x):
        h_0 = torch.randn(self.num_layers * (1 + self.bidirectional), x.size(0), self.hidden_size).to(x.device)
        c_0 = torch.randn(self.num_layers * (1 + self.bidirectional), x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h_0, c_0))
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])  ## output_size,2
        return out

class RNNRegressor(nn.Module):
   def __init__(self, input_size, hidden_size, num_layers,output_size ,dropout_prob=0.5):
       super(RNNRegressor, self).__init__()
       self.hidden_size = hidden_size
       self.num_layers = num_layers
       self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
       self.dropout = nn.Dropout(dropout_prob)
       self.fc = nn.Linear(hidden_size, output_size)

   def forward(self, x):
       h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
       out, _ = self.rnn(x, h0)
       # 使用最后一个时间步的输出
       out = out[:, -1, :]
       out = self.dropout(out)
       out = self.fc(out)
       return out

def combine_csv(df,fold_subs):

    for cur_sub in fold_subs:
        if 'TD' in cur_sub:
            file_path = '../AU-ab-5-level-end/TD/' + cur_sub[:-2] + '.csv'
        else:
            file_path = '../AU-ab-5-level-end/ASD/' + cur_sub + '.csv'
        df_new = pd.read_csv(file_path)
        # 按帧升序排序
        df_new = df_new.sort_values(by='frame',ascending=True)
        df = pd.concat([df, df_new], ignore_index=True)
    return df


class FrameDataset(Dataset):
    def __init__(self, train=True, fold=1, time_step=10, convert1=False):
        self.data = []
        self.labels = []
        self.time_step = time_step
        self._train = train

        if self._train:
            df = pd.DataFrame()
            # fold1:1+2 tr,3ts; fold2:1+3 tr,2ts; fold3:2+3 tr,1ts;
            if fold == 1:
                df = combine_csv(df,fold_1)
                df = combine_csv(df, fold_2)
            elif fold == 2:
                df = combine_csv(df, fold_1)
                df = combine_csv(df, fold_3)
            elif fold == 3:
                df = combine_csv(df, fold_2)
                df = combine_csv(df, fold_3)
            else:
                raise ValueError("Invalid fold value. Fold must be between 1 and 3.")

        else:
            df = pd.DataFrame()
            # fold1:1+2 tr,3ts; fold2:1+3 tr,2ts; fold3:2+3 tr,1ts;
            if fold == 1:
                df = combine_csv(df, fold_3)
            elif fold == 2:
                df = combine_csv(df, fold_2)
            elif fold == 3:
                df = combine_csv(df, fold_1)
            else:
                raise ValueError("Invalid fold value. Fold must be between 1 and 3.")

        if 'AU0' in df.columns and 'AU21' in df.columns and 'AU31' in df.columns:
            if use_selected_au:
                print('Using selected 22 AU')
                # df = df.drop(columns=['AU0', 'AU21', 'AU31','ASD','Abnormal-j','Abnormal-jie','Abnormal-y','Abnormal-z','Abnormal-w','AU5','AU8','AU13','AU18','AD19','AU22','AU28','AD29','AD30','AD32','AD33','AD34','AD35','AD36','AD37','AU43'])  # 17个AU训练
                df = df.drop(columns=['AU0', 'AU21', 'AU31','ASD','Abnormal-j','Abnormal-jie','Abnormal-y','Abnormal-z','Abnormal-w','AU5','AU8','AU13','AU22','AD29','AD30','AD33','AD34','AD35','AD36','AD37'])  # 22个AU
            else:
                print('Using all AU')
                df = df.drop(
                    columns=['AU0', 'AU21', 'AU31', 'ASD', 'Abnormal-j', 'Abnormal-jie', 'Abnormal-y', 'Abnormal-z',
                             'Abnormal-w'])  # 全部AU/AD训练
        if convert1:
            columns_to_modify = df.columns[1:-1]  # 标签转为1
            # print(columns_to_modify)
            df[columns_to_modify] = df[columns_to_modify].map(lambda x: 1 if x > 1 else x)
        else:
            print('标签未转换为1')
        segments, segment_labels = load_and_process_csv(df)

        for segment, label_segment in zip(segments, segment_labels):
            if len(segment) >= time_step:  # 滑动窗口
                for cur_sub in range(len(segment) - time_step + 1):
                    self.data.append(segment[cur_sub:cur_sub + time_step])
                    # 需要判断这个片段里哪个标签最多，就采用哪个标签， 因为不同人标注的起止点可能不同导致平均分不一致；或采用整个片段评分的均值做标签
                    most_label = Counter(label_segment[cur_sub:cur_sub + time_step]).most_common(1)[0][0]
                    self.labels.append(most_label)
            else:
                if label_segment[0] > 2.0:
                    print(f'片段长度不满足窗口且等级>2，丢弃:{len(segment),label_segment}')

                # 制造重复数据
                # repeats = (time_step // len(segment)) + 1
                #
                # # 使用np.tile重复data,重复repeats次，列出不变
                # duplicated_data = np.tile(segment, (repeats,1))
                #
                # # duplicated_data1 = np.repeat(segment, repeats, axis=0)
                # duplicated_label = np.repeat(label_segment, repeats, axis=0)
                # # 扩充后继续滑动窗口
                # for cur_sub in range(len(duplicated_data) - time_step + 1):
                #     self.data.append(duplicated_data[cur_sub:cur_sub + time_step])
                #     0
                #     self.labels.append(duplicated_label[cur_sub + time_step - 1])
                # print('片段长度:', len(segment), '重复制造的数据shape:', duplicated_data.shape)
                # print('重复制造的标签shape:', duplicated_label.shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

def load_and_process_csv(df):
    """
    按照帧的连续片段进行划分数据和标签
    :param df: 已经丢弃了 ['AU0', 'AU21', 'AU31','ASD'])
    :return:
    """

    frames = df['frame'].values

    features = df.iloc[:, 1:-1].values # 到-1 不包含Abnormal列
    labels = df.iloc[:, -1].values + 1.0 # 最后一列标签Abnormal-avg 【0，1，2，3，4】+1
    labels = labels.astype(float)

    segments = []
    segment_labels = []

    start_idx = 0
    for i in range(1, len(frames)):
        if frames[i] != frames[i - 1] + 1:
            segments.append(features[start_idx:i])
            segment_labels.append(labels[start_idx:i])
            start_idx = i
    segments.append(features[start_idx:])  # 最后剩余的数据
    segment_labels.append(labels[start_idx:])

    return segments, segment_labels

def main(fold,model_name='GRU',convert1=False):

    tr_dataset = FrameDataset(train=True, fold=fold, time_step=time_step,convert1=convert1)
    tr_dataloader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
                                      num_workers=num_workers)
    ts_dataset = FrameDataset(train=False, fold=fold, time_step=time_step,convert1=convert1)
    ts_dataloader = DataLoader(ts_dataset, batch_size=batch_size, shuffle=False)

    if model_name == 'BiLSTM':
        model = LSTMModel(input_size, hidden_size, num_layers, num_classes, bidirectional=True).to(device)
    elif model_name == 'GRU':
        model = GRUModel(input_size, hidden_size, num_layers, num_classes).to(device)
    elif model_name == 'TR':
        # 使用示例
        model = TransformerModel(input_size, num_heads=3, num_layers=num_layers, num_classes=num_classes).to(device)
    elif model_name == 'LSTM':
        model = LSTMModel(input_size, hidden_size, num_layers, num_classes, bidirectional=False).to(device)
    elif model_name == 'RNN':
        model = RNNRegressor(input_size, hidden_size, num_layers, num_classes).to(device)
    else:
        raise ValueError('Invalid model name.')

    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=init_lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=sche_factor, patience=2)

    best_mae = np.inf
    best_mse = np.inf
    best_rmse =  np.inf
    best_mape = np.inf

    for epoch in range(num_epochs):
        model.train()
        # with tqdm(total=len(X_train), desc=f'Epoch {epoch + 1}/{num_epochs}', unit='sample') as pbar:
        for i, (inputs, labels) in enumerate(tqdm(tr_dataloader)):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            labels = labels.unsqueeze(dim=1)
            loss = criterion(outputs, labels.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # val
        model.eval()
        val_loss = 0.0
        total_mae = 0.0  # acc
        total_mse = 0.0
        total_rmse = 0.0
        total_mape = 0.0

        with torch.no_grad():
            for inputs, labels in ts_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                labels = labels.unsqueeze(dim=1)
                loss = criterion(outputs, labels.float())
                val_loss += loss.item()

                total_mae += mean_absolute_error(outputs, labels).item()
                total_mse += mean_squared_error(outputs, labels).item()
                total_rmse += rmse_metric(outputs, labels)
                total_mape  += mape_metric(outputs, labels)

        mae_avg = total_mae / len(ts_dataloader)
        mse_avg = total_mse / len(ts_dataloader)
        rmse_avg = total_rmse / len(ts_dataloader)
        mape_avg = total_mape / len(ts_dataloader)

        scheduler.step(val_loss / len(ts_dataloader))
        print(
            f'Fold [Epoch [{epoch + 1}/{num_epochs}], Loss: {val_loss / len(ts_dataloader):.2f}, MAE: {mae_avg:.4f}, MSE:{mse_avg:.4f}, RMSE:{rmse_avg:.4f}, MAPE:{mape_avg:.4f}, LR: {scheduler.get_last_lr()[0]:.2e}')

        if mse_avg < best_mse:
            best_mae = mae_avg
            best_mse = mse_avg
            best_rmse = rmse_avg
            best_mape = mape_avg

            checkpoint = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(checkpoint,
                       os.path.join(model_name+'_fold_' + str(fold) + '_feature_size_' + str(input_size)+ '_best_mse'+ '.pth'))  # 保存当前epoch模型

        # 记录当前epoch 各种数据,四舍五入
        logger.info(f'Fold [Epoch [{epoch + 1}/{num_epochs}], Loss: {val_loss / len(ts_dataloader):.4f}, Based on Best-MSE: {best_mse:.4f} ,Best-MAE: {best_mae:.4f}, Best-RMSE:{best_rmse}, Best_MAPE: {best_mape}, Cur-MAE: {mae_avg:.4f}, Cur-MSE:{mse_avg:.4f}, Cur-RMSE:{rmse_avg:.4f}, Cur-MAPE:{mape_avg}, LR: {scheduler.get_last_lr()[0]:.2e}')

    return best_mae,best_mse,best_rmse,best_mape



if __name__ == '__main__':
    """
    异常等级回归
    """
    model_name = 'GRU'

    logger = create_my_logger(model_name,exp_name='AU-AB-regressor-22-AU')
    fold_index = 3
    num_epochs = 15
    time_step = 15  # 宏表情0.5-4s
    batch_size = 64
    input_size = 22  # 特征维度删掉AU0，删掉21，31，就是33.
    hidden_size = 128
    num_layers = 2
    num_classes = 1
    init_lr = 1e-6
    num_workers = 8
    drop_out = 0.5
    sche_factor = 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_mcc_para = [0, 0, 0,0]
    all_cv_best_mcc = 0.0 # 所有交叉验证里最好的mcc

    convert1 = True
    use_selected_au = True # 17个或者22个

    rmse_metric = MeanSquaredError(squared=False).to(device)
    mape_metric = MeanAbsolutePercentageError().to(device)


    asd_fold_1 = ['S6', 'S45', 'S62', 'S39', 'S70', 'S65', 'S3', 'S57', 'S32', 'S27', 'S72', 'S31', 'S75', 'S4', 'S21',
                  'S22',
                  'S59', 'S7', 'S2', 'S74', 'S53', 'S43']
    asd_fold_2 = ['S47', 'S41', 'S8', 'S80', 'S51', 'S42', 'S36', 'S63', 'S54', 'S33', 'S10', 'S12', 'S16', 'S44',
                  'S38', 'S15',
                  'S81', 'S25', 'S14', 'S28', 'S68', 'S64']
    asd_fold_3 = ['S5', 'S29', 'S24', 'S58', 'S67', 'S49', 'S37', 'S35', 'S34', 'S71', 'S19', 'S52', 'S9', 'S23', 'S1',
                  'S40',
                  'S26', 'S13', 'S11', 'S46', 'S60', 'S61']
    asd_all = asd_fold_1 + asd_fold_2 + asd_fold_3

    td_all = ['S32TD', 'S40TD', 'S79TD', 'S73TD', 'S33TD', 'S74TD', 'S75TD', 'S28TD', 'S85TD', 'S27TD',
                 'S1TD','S2TD','S3TD','S4TD','S5TD','S6TD','S7TD','S8TD','S9TD','S10TD',
              'S11TD','S12TD','S13TD','S14TD','S15TD','S16TD','S17TD','S18TD','S19TD','S20TD',
              'S21TD', 'S22TD']

    random.shuffle(asd_all)
    random.shuffle(td_all)

    # 3折
    fold_1 = asd_all[:22] + td_all[:11]
    fold_2 = asd_all[22:44] + td_all[11:22]
    fold_3 = asd_all[44:] + td_all[22:]
    logger.info("fold_1:\t" + str(fold_1))
    logger.info("fold_2:\t" + str(fold_2))
    logger.info("fold_3:\t" + str(fold_3))

    mse_list = np.empty(3, dtype=float)
    mae_list = np.empty(3, dtype=float)
    rmse_list = np.empty(3, dtype=float)
    mape_list = np.empty(3, dtype=float)
    for fold_index in range(3):
        logger.info(
              f'Test fold:{fold_index+1}, num_epochs:{num_epochs},time_step:{time_step},batch_size:{batch_size},input_size:{input_size},hidden_size:{hidden_size},num_layers:{num_layers},num_classes:{num_classes},init_lr:{init_lr},num_workers:{num_workers},drop_out:{drop_out},Schedule_factor:{sche_factor}')
        best_mae,best_mse,best_rmse,best_mape= main(fold_index+1, model_name, convert1)

        logger.info('-----------------------------------')
        mse_list[fold_index] = best_mse
        mae_list[fold_index] = best_mae
        rmse_list[fold_index] = best_rmse
        mape_list[fold_index] = best_mape

    print(f'ALL Fold Mean: mse:{mse_list.mean() :.4f},mae:{mae_list.mean():.4f},rmse:{rmse_list.mean():.4f},mape:{mape_list.mean():.4f}')
    # print(mcc_list)
    logger.info('ALL Fold Mean:')
    logger.info(f'mse:{mse_list.mean() :.4f},mae:{mae_list.mean():.4f}, rmse:{rmse_list.mean():.4f},mape:{mape_list.mean():.4f}')
    logger.info('----------------------------------')