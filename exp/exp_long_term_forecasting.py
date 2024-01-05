from torch.utils.tensorboard import SummaryWriter
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from tqdm import tqdm

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag): # 加载数据
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        
        # 如果是继续训练，加载模型
        if self.args.back_training:
            print('loading model')
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location=torch.device(device)))
            print('加载模型结束')
            
        # 取得训练、验证、测试数据及数据加载器
        writer = SummaryWriter(log_dir=os.path.join('tensorboard_logs', setting))
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        # 取训练步数
        train_steps = len(train_loader)
        # 设置早停参数
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        # 选择优化器
        model_optim = self._select_optimizer()
        # 选择损失函数
        criterion = self._select_criterion()

        # 如果多GPU并行
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        
        # 训练次数
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train() # 将模型设置为训练模式.
            epoch_time = time.time()
            train_pbar = tqdm(train_loader, position=0, leave=True) # 载入进度条
            
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_pbar):
                iter_count += 1
                # 梯度归零
                model_optim.zero_grad()
                
                # 取训练数据
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder 并行计算
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    # 如果预测方式为MS，取最后1列否则取第1列
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    # 计算损失
                    loss = criterion(outputs, batch_y)
                    # 将损失放入train_loss列表中
                    train_loss.append(loss.item())

                # 记录训练过程
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    # 反向传播
                    loss.backward()
                    # 更新梯度
                    model_optim.step()
            # 在 tqdm 进度条上显示当前 epoch number和loss .
            train_pbar.set_description(f'Epoch [{epoch+1}/{self.args.train_epochs}]') # 设置进度条的前缀
            train_pbar.set_postfix({'loss' : loss.detach().item()}) # 设置进度条的后缀

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss) # 计算平均损失
            writer.add_scalar('Loss/train_epoch', train_loss, epoch) # 将训练损失写入tensorboard
            print("正在计算验证集Loss...")
            vali_loss = self.vali(vali_data, vali_loader, criterion) # 计算验证损失
            writer.add_scalar('Loss/val', vali_loss, epoch) # 将验证损失写入tensorboard
            print("正在计算测试集Loss...")
            test_loss = self.vali(test_data, test_loader, criterion) # 计算测试损失

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # 更新学习率
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        # 保存模型
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        writer.close()

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test') # 加载test数据
        if test:
            print('loading model')
            # 原始代码
            # self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

            # 修改后的代码
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location=torch.device(device)))
            print('加载模型结束')

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            data_loader_length = len(test_loader)
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                if i < data_loader_length - 1 and self.args.pred_once:
                    continue  # 跳过除了最后一批数据之外的所有批次
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                '''
                batch_x = seq_len 60(里面包含了label_len,lable_len是已知的数据,在batch_x后跟batch_y交叉重叠的部分)
                batch_y = label_len 20 (batch_x和batch_y交叉重叠的部分) + pred_len 20 (batch_y的后半部分)
                decoder input 将batch_y前半部分（lable_len）保留，后面的预测长度替换成零拼接进去，形状跟batch_y相同
                '''
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()  # 预测长度的零矩阵，pred_len 20 (batch_y的后半部分)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device) # 截取batch_y前半部分（lable_len 20），后面的预测长度替换成零拼接进去，形状跟batch_y 40相同
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        # batch_x = seq_len 60,dec_inp = batch_y = label_len 20（原数据） + pred_len 20(零矩阵)
                        # outputs 输出预测值= pred_len 20
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :] # 取outputs的后半部分（pred_len 20 预测值），相当于全取
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device) # 取batch_y的后半部分（pred_len 20 真实值）,形状跟outputs相同
                outputs = outputs.detach().cpu().numpy() # 将outputs转换为numpy格式
                batch_y = batch_y.detach().cpu().numpy() # 将batch_y转换为numpy格式
                if test_data.scale and self.args.inverse: # 如果数据进行了归一化且需要逆归一化
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)
        
                # 如果预测方式为MS，取最后1列否则取全部
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0: # 每20个batch画一次图
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds) # 将预测结果转换为numpy格式
        trues = np.array(trues) # 将真实结果转换为numpy格式
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1]) # 将预测结果转换为三维矩阵
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
