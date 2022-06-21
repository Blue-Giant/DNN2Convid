"""
@author: LXA
Benchmark Code of SIRD model
2022-06-18
"""
import os
import sys
import tensorflow as tf
import numpy as np
import time
import platform
import shutil
import DNN_base
import DNN_tools
import DNN_data
import plotData
import saveData


# 记录字典中的一些设置
def dictionary_out2file(R_dic, log_fileout):
    DNN_tools.log_string('Equation name for problem: %s\n' % (R_dic['eqs_name']), log_fileout)
    DNN_tools.log_string('Network model of dealing with SIR: %s\n' % str(R_dic['model2SIRD']), log_fileout)
    DNN_tools.log_string('Network model of dealing with parameters: %s\n' % str(R_dic['model2paras']), log_fileout)
    if str.upper(R_dic['model2SIRD']) == 'DNN_FOURIERBASE':
        DNN_tools.log_string('The input activate function for SIRD: %s\n' % '[sin;cos]', log_fileout)
    else:
        DNN_tools.log_string('The input activate function for SIRD: %s\n' % str(R_dic['actIn_Name2SIRD']), log_fileout)

    DNN_tools.log_string('The hidden-layer activate function for SIRD: %s\n' % str(R_dic['act_Name2SIRD']), log_fileout)

    if str.upper(R_dic['model2paras']) == 'DNN_FOURIERBASE':
        DNN_tools.log_string('The input activate function for parameter: %s\n' % '[sin;cos]', log_fileout)
    else:
        DNN_tools.log_string('The input activate function for parameter: %s\n' % str(R_dic['actIn_Name2paras']), log_fileout)

    DNN_tools.log_string('The hidden-layer activate function for parameter: %s\n' % str(R_dic['act_Name2paras']), log_fileout)

    DNN_tools.log_string('hidden layers for SIR: %s\n' % str(R_dic['hidden2SIRD']), log_fileout)
    DNN_tools.log_string('hidden layers for parameters: %s\n' % str(R_dic['hidden2para']), log_fileout)

    if str.upper(R_dic['model2SIRD']) != 'DNN':
        DNN_tools.log_string('The scale for frequency to SIR NN: %s\n' % str(R_dic['freq2SIRD']), log_fileout)
        DNN_tools.log_string('Repeat the high-frequency scale or not for SIR-NN: %s\n' % str(R_dic['if_repeat_High_freq2SIRD']), log_fileout)
    if str.upper(R_dic['model2paras']) != 'DNN':
        DNN_tools.log_string('The scale for frequency to SIR NN: %s\n' % str(R_dic['freq2paras']), log_fileout)
        DNN_tools.log_string('Repeat the high-frequency scale or not for para-NN: %s\n' % str(R_dic['if_repeat_High_freq2paras']), log_fileout)

    DNN_tools.log_string('Init learning rate: %s\n' % str(R_dic['learning_rate']), log_fileout)
    DNN_tools.log_string('Decay to learning rate: %s\n' % str(R_dic['lr_decay']), log_fileout)
    DNN_tools.log_string('The type for Loss function: %s\n' % str(R_dic['loss_function']), log_fileout)

    if (R_dic['optimizer_name']).title() == 'Adam':
        DNN_tools.log_string('optimizer:%s\n' % str(R_dic['optimizer_name']), log_fileout)
    else:
        DNN_tools.log_string('optimizer:%s  with momentum=%f\n' % (R_dic['optimizer_name'], R_dic['momentum']), log_fileout)

    if R_dic['activate_stop'] != 0:
        DNN_tools.log_string('activate the stop_step and given_step= %s\n' % str(R_dic['max_epoch']), log_fileout)
    else:
        DNN_tools.log_string('no activate the stop_step and given_step = default: %s\n' % str(R_dic['max_epoch']), log_fileout)

    DNN_tools.log_string(
        'Initial penalty for difference of predict and true: %s\n' % str(R_dic['init_penalty2predict_true']), log_fileout)

    DNN_tools.log_string('The model of regular weights and biases: %s\n' % str(R_dic['regular_weight_model']), log_fileout)

    DNN_tools.log_string('Regularization parameter for weights and biases: %s\n' % str(R_dic['regular_weight']), log_fileout)

    DNN_tools.log_string('Size 2 training set: %s\n' % str(R_dic['size2train']), log_fileout)

    DNN_tools.log_string('Batch-size 2 training: %s\n' % str(R_dic['batch_size2train']), log_fileout)

    DNN_tools.log_string('Batch-size 2 testing: %s\n' % str(R_dic['batch_size2test']), log_fileout)


def print_and_log2train(i_epoch, run_time, tmp_lr, temp_penalty_nt, penalty_wb2s, penalty_wb2i, penalty_wb2r,
                        penalty_wb2d, loss_s, loss_i, loss_r, loss_d, loss_n, log_out=None):
    print('train epoch: %d, time: %.3f' % (i_epoch, run_time))
    print('learning rate: %f' % tmp_lr)
    print('penalty for difference of predict and true : %f' % temp_penalty_nt)
    print('penalty weights and biases for S: %f' % penalty_wb2s)
    print('penalty weights and biases for I: %f' % penalty_wb2i)
    print('penalty weights and biases for R: %f' % penalty_wb2r)
    print('penalty weights and biases for D: %f' % penalty_wb2d)
    print('loss for S: %.16f' % loss_s)
    print('loss for I: %.16f' % loss_i)
    print('loss for R: %.16f' % loss_r)
    print('loss for D: %.16f' % loss_d)
    print('total loss: %.16f\n' % loss_n)

    DNN_tools.log_string('train epoch: %d,time: %.3f' % (i_epoch, run_time), log_out)
    DNN_tools.log_string('learning rate: %f' % tmp_lr, log_out)
    DNN_tools.log_string('penalty for difference of predict and true : %f' % temp_penalty_nt, log_out)
    DNN_tools.log_string('penalty weights and biases for S: %f' % penalty_wb2s, log_out)
    DNN_tools.log_string('penalty weights and biases for I: %f' % penalty_wb2i, log_out)
    DNN_tools.log_string('penalty weights and biases for R: %.10f' % penalty_wb2r, log_out)
    DNN_tools.log_string('penalty weights and biases for D: %.10f' % penalty_wb2d, log_out)
    DNN_tools.log_string('loss for S: %.16f' % loss_s, log_out)
    DNN_tools.log_string('loss for I: %.16f' % loss_i, log_out)
    DNN_tools.log_string('loss for R: %.16f' % loss_r, log_out)
    DNN_tools.log_string('loss for D: %.16f' % loss_d, log_out)
    DNN_tools.log_string('total loss: %.16f \n\n' % loss_n, log_out)


def solve_SIRD2COVID(R):
    log_out_path = R['FolderName']  # 将路径从字典 R 中提取出来
    if not os.path.exists(log_out_path):  # 判断路径是否已经存在
        os.mkdir(log_out_path)  # 无 log_out_path 路径，创建一个 log_out_path 路径
    log_fileout = open(os.path.join(log_out_path, 'log_train.txt'), 'w')  # 在这个路径下创建并打开一个可写的 log_train.txt文件
    dictionary_out2file(R, log_fileout)

    log2trianSolus = open(os.path.join(log_out_path, 'train_Solus.txt'), 'w')  # 在这个路径下创建并打开一个可写的 log_train.txt文件
    log2testSolus = open(os.path.join(log_out_path, 'test_Solus.txt'), 'w')  # 在这个路径下创建并打开一个可写的 log_train.txt文件
    log2testSolus2 = open(os.path.join(log_out_path, 'test_Solus_temp.txt'), 'w')  # 在这个路径下创建并打开一个可写的 log_train.txt文件

    log2testParas = open(os.path.join(log_out_path, 'test_Paras.txt'), 'w')  # 在这个路径下创建并打开一个可写的 log_train.txt文件

    trainSet_szie = R['size2train']                   # 训练集大小,给定一个数据集，拆分训练集和测试集时，需要多大规模的训练集
    batchSize_train = R['batch_size2train']           # 训练批量的大小,该值远小于训练集大小
    batchSize_test = R['batch_size2test']             # 测试批量的大小,该值小于等于测试集大小
    pt_penalty_init = R['init_penalty2predict_true']  # 预测值和真值得误差惩罚因子的初值,用于处理那些具有真实值得变量
    wb_penalty = R['regular_weight']                  # 神经网络参数的惩罚因子
    lr_decay = R['lr_decay']                          # 学习率额衰减
    init_lr = R['learning_rate']                      # 初始学习率

    act_func2SIRD = R['act_Name2SIRD']                 # S, I, R D 四个神经网络的隐藏层激活函数
    act_func2paras = R['act_Name2paras']              # 参数网络的隐藏层激活函数

    input_dim = R['input_dim']                        # 输入维度
    out_dim = R['output_dim']                         # 输出维度

    flag2S = 'WB2S'
    flag2I = 'WB2I'
    flag2R = 'WB2R'
    flag2D = 'WB2D'
    flag2beta = 'WB2beta'
    flag2gamma = 'WB2gamma'
    hidden_sird = R['hidden2SIRD']
    hidden_para = R['hidden2para']

    if str.upper(R['model2SIRD']) == 'DNN_FOURIERBASE':
        Weight2S, Bias2S = DNN_base.Xavier_init_NN_Fourier(input_dim, out_dim, hidden_sird, flag2S)
        Weight2I, Bias2I = DNN_base.Xavier_init_NN_Fourier(input_dim, out_dim, hidden_sird, flag2I)
        Weight2R, Bias2R = DNN_base.Xavier_init_NN_Fourier(input_dim, out_dim, hidden_sird, flag2R)
        Weight2D, Bias2D = DNN_base.Xavier_init_NN_Fourier(input_dim, out_dim, hidden_sird, flag2D)
    else:
        Weight2S, Bias2S = DNN_base.Xavier_init_NN(input_dim, out_dim, hidden_sird, flag2S)
        Weight2I, Bias2I = DNN_base.Xavier_init_NN(input_dim, out_dim, hidden_sird, flag2I)
        Weight2R, Bias2R = DNN_base.Xavier_init_NN(input_dim, out_dim, hidden_sird, flag2R)
        Weight2D, Bias2D = DNN_base.Xavier_init_NN(input_dim, out_dim, hidden_sird, flag2D)

    if str.upper(R['model2paras']) == 'DNN_FOURIERBASE':
        Weight2beta, Bias2beta = DNN_base.Xavier_init_NN_Fourier(input_dim, out_dim, hidden_para, flag2beta)
        Weight2gamma, Bias2gamma = DNN_base.Xavier_init_NN_Fourier(input_dim, out_dim, hidden_para, flag2gamma)
    else:
        Weight2beta, Bias2beta = DNN_base.Xavier_init_NN(input_dim, out_dim, hidden_para, flag2beta)
        Weight2gamma, Bias2gamma = DNN_base.Xavier_init_NN(input_dim, out_dim, hidden_para, flag2gamma)

    global_steps = tf.Variable(0, trainable=False)
    with tf.device('/gpu:%s' % (R['gpuNo'])):
        with tf.compat.v1.variable_scope('vscope', reuse=tf.compat.v1.AUTO_REUSE):
            T_it = tf.compat.v1.placeholder(tf.float32, name='T_it', shape=[None, out_dim])
            S_observe = tf.compat.v1.placeholder(tf.float32, name='S_observe', shape=[None, out_dim])
            I_observe = tf.compat.v1.placeholder(tf.float32, name='I_observe', shape=[None, out_dim])
            N_observe = tf.compat.v1.placeholder(tf.float32, name='N_observe', shape=[None, out_dim])
            predict_true_penalty = tf.compat.v1.placeholder_with_default(input=1e3, shape=[], name='pt_p')
            in_learning_rate = tf.compat.v1.placeholder_with_default(input=1e-5, shape=[], name='lr')
            # in_beta = tf.compat.v1.placeholder_with_default(input=1e-5, shape=[], name='beta')
            # in_gamma = tf.compat.v1.placeholder_with_default(input=1e-5, shape=[], name='gamma')

            freq2SIRD = R['freq2SIRD']
            if 'DNN' == str.upper(R['model2SIRD']):
                SNN_temp = DNN_base.DNN(T_it, Weight2S, Bias2S, hidden_sird, activateIn_name=R['actIn_Name2SIR'],
                                        activate_name=act_func2SIRD)
                INN_temp = DNN_base.DNN(T_it, Weight2I, Bias2I, hidden_sird, activateIn_name=R['actIn_Name2SIR'],
                                        activate_name=act_func2SIRD)
                RNN_temp = DNN_base.DNN(T_it, Weight2R, Bias2R, hidden_sird, activateIn_name=R['actIn_Name2SIR'],
                                        activate_name=act_func2SIRD)
                DNN_temp = DNN_base.DNN(T_it, Weight2D, Bias2D, hidden_sird, activateIn_name=R['actIn_Name2SIR'],
                                        activate_name=act_func2SIRD)
            elif 'DNN_SCALE' == str.upper(R['model2SIRD']):
                SNN_temp = DNN_base.DNN_scale(T_it, Weight2S, Bias2S, hidden_sird, freq2SIRD,
                                              activateIn_name=R['actIn_Name2SIR'], activate_name=act_func2SIRD)
                INN_temp = DNN_base.DNN_scale(T_it, Weight2I, Bias2I, hidden_sird, freq2SIRD,
                                              activateIn_name=R['actIn_Name2SIR'], activate_name=act_func2SIRD)
                RNN_temp = DNN_base.DNN_scale(T_it, Weight2R, Bias2R, hidden_sird, freq2SIRD,
                                              activateIn_name=R['actIn_Name2SIR'], activate_name=act_func2SIRD)
                DNN_temp = DNN_base.DNN_scale(T_it, Weight2D, Bias2D, hidden_sird, freq2SIRD,
                                              activateIn_name=R['actIn_Name2SIR'], activate_name=act_func2SIRD)
            elif str.upper(R['model2SIRD']) == 'DNN_FOURIERBASE':
                SNN_temp = DNN_base.DNN_FourierBase(T_it, Weight2S, Bias2S, hidden_sird, freq2SIRD,
                                                    activate_name=act_func2SIRD, sFourier=1.0)
                INN_temp = DNN_base.DNN_FourierBase(T_it, Weight2I, Bias2I, hidden_sird, freq2SIRD,
                                                    activate_name=act_func2SIRD, sFourier=1.0)
                RNN_temp = DNN_base.DNN_FourierBase(T_it, Weight2R, Bias2R, hidden_sird, freq2SIRD,
                                                    activate_name=act_func2SIRD, sFourier=1.0)
                DNN_temp = DNN_base.DNN_FourierBase(T_it, Weight2D, Bias2D, hidden_sird, freq2SIRD,
                                                    activate_name=act_func2SIRD, sFourier=1.0)

            freq2paras = R['freq2paras']
            if 'DNN' == str.upper(R['model2paras']):
                in_beta = DNN_base.DNN(T_it, Weight2beta, Bias2beta, hidden_para, activateIn_name=R['actIn_Name2paras'],
                                       activate_name=act_func2paras)
                in_gamma = DNN_base.DNN(T_it, Weight2gamma, Bias2gamma, hidden_para,
                                        activateIn_name=R['actIn_Name2paras'], activate_name=act_func2paras)
            elif 'DNN_SCALE' == str.upper(R['model2paras']):
                in_beta = DNN_base.DNN_scale(T_it, Weight2beta, Bias2beta, hidden_para, freq2paras,
                                             activateIn_name=R['actIn_Name2paras'], activate_name=act_func2paras)
                in_gamma = DNN_base.DNN_scale(T_it, Weight2gamma, Bias2gamma, hidden_para, freq2paras,
                                              activateIn_name=R['actIn_Name2paras'], activate_name=act_func2paras)
            elif str.upper(R['model2SIR']) == 'DNN_FOURIERBASE':
                in_beta = DNN_base.DNN_FourierBase(T_it, Weight2beta, Bias2beta, hidden_para, freq2paras,
                                                   activate_name=act_func2paras, sFourier=1.0)
                in_gamma = DNN_base.DNN_FourierBase(T_it, Weight2gamma, Bias2gamma, hidden_para, freq2paras,
                                                    activate_name=act_func2paras, sFourier=1.0)

            # Remark: beta, gamma,S_NN.I_NN,R_NN都应该是正的. beta.1--15之间，gamma在(0,1）使用归一化的话S_NN.I_NN,R_NN都在[0,1)范围内
            # 在归一化条件下: 如果总的“人口”和归一化"人口"的数值一致，这样的话，归一化后的数值会很小
            if (R['total_population'] == R['normalize_population']) and R['normalize_population'] != 1:
                beta = tf.square(in_beta)
                gamma = tf.nn.sigmoid(in_gamma)
                # S_NN = SNN_temp
                # I_NN = INN_temp
                # R_NN = RNN_temp
                # D_NN = DNN_temp

                # S_NN = tf.nn.relu(SNN_temp)
                # I_NN = tf.nn.relu(INN_temp)
                # R_NN = tf.nn.relu(RNN_temp)
                # D_NN = tf.nn.relu(DNN_temp)

                # S_NN = tf.abs(SNN_temp)
                # I_NN = tf.abs(INN_temp)
                # R_NN = tf.abs(RNN_temp)
                # D_NN = tf.abs(DNN_temp)

                # S_NN = DNN_base.gauss(SNN_temp)
                # I_NN = tf.square(INN_temp)
                # R_NN = tf.square(RNN_temp)
                # D_NN = tf.square(DNN_temp)

                # S_NN = DNN_base.gauss(SNN_temp)
                # I_NN = tf.square(INN_temp)
                # R_NN = tf.nn.sigmoid(RNN_temp)
                # D_NN = tf.nn.sigmoid(DNN_temp)

                # S_NN = DNN_base.gauss(SNN_temp)
                # I_NN = tf.nn.sigmoid(INN_temp)
                # R_NN = tf.square(RNN_temp)
                # D_NN = tf.square(DNN_temp)

                # S_NN = tf.sqrt(tf.square(SNN_temp))
                # I_NN = tf.sqrt(tf.square(INN_temp))
                # R_NN = tf.sqrt(tf.square(RNN_temp))
                # D_NN = tf.sqrt(tf.square(DNN_temp))

                S_NN = tf.nn.sigmoid(SNN_temp)
                I_NN = tf.nn.sigmoid(INN_temp)
                R_NN = tf.nn.sigmoid(RNN_temp)
                D_NN = tf.nn.sigmoid(DNN_temp)

                # S_NN = tf.tanh(SNN_temp)
                # I_NN = tf.tanh(INN_temp)
                # R_NN = tf.tanh(RNN_temp)
                # D_NN = tf.tanh(DNN_temp)
            else:
                beta = tf.square(in_beta)
                gamma = tf.nn.sigmoid(in_gamma)

                # S_NN = SNN_temp
                # I_NN = INN_temp
                # R_NN = RNN_temp
                # D_NN = DNN_temp

                # S_NN = tf.nn.relu(SNN_temp)
                # I_NN = tf.nn.relu(INN_temp)
                # R_NN = tf.nn.relu(RNN_temp)
                # D_NN = tf.nn.relu(DNN_temp)

                S_NN = tf.nn.sigmoid(SNN_temp)
                I_NN = tf.nn.sigmoid(INN_temp)
                R_NN = tf.nn.sigmoid(RNN_temp)
                D_NN = tf.nn.sigmoid(DNN_temp)

                # S_NN = tf.tanh(SNN_temp)
                # I_NN = tf.tanh(INN_temp)
                # R_NN = tf.tanh(RNN_temp)
                # D_NN = tf.tanh(DNN_temp)

            N_NN = S_NN + I_NN + R_NN + + D_NN

            dS_NN2t = tf.gradients(S_NN, T_it)[0]
            dI_NN2t = tf.gradients(I_NN, T_it)[0]
            dR_NN2t = tf.gradients(R_NN, T_it)[0]
            dD_NN2t = tf.gradients(D_NN, T_it)[0]
            dN_NN2t = tf.gradients(N_NN, T_it)[0]

            temp_snn2t = -beta*S_NN*I_NN
            temp_inn2t = beta*S_NN*I_NN - gamma * I_NN
            temp_rnn2t = gamma *I_NN
            temp_dnn2t = gamma * I_NN

            if str.lower(R['loss_function']) == 'l2_loss' and R['scale_up'] == 0:
                # LossS_Net_obs = tf.reduce_mean(tf.square(S_NN - S_observe))
                LossI_Net_obs = tf.reduce_mean(tf.square(I_NN - I_observe))
                # LossR_Net_obs = tf.reduce_mean(tf.square(R_NN - R_observe))
                # LossD_Net_obs = tf.reduce_mean(tf.square(D_NN - D_observe))
                LossN_Net_obs = tf.reduce_mean(tf.square(N_NN - N_observe))

                Loss2dS = tf.reduce_mean(tf.square(dS_NN2t - temp_snn2t))
                Loss2dI = tf.reduce_mean(tf.square(dI_NN2t - temp_inn2t))
                Loss2dR = tf.reduce_mean(tf.square(dR_NN2t - temp_rnn2t))
                Loss2dN = tf.reduce_mean(tf.square(dN_NN2t))
                Loss2dD = tf.reduce_mean(tf.square(dD_NN2t - temp_dnn2t))
            elif str.lower(R['loss_function']) == 'l2_loss' and R['scale_up'] == 1:
                scale_up = R['scale_factor']
                # LossS_Net_obs = tf.reduce_mean(tf.square(scale_up*S_NN - scale_up*S_observe))
                LossI_Net_obs = tf.reduce_mean(tf.square(scale_up*I_NN - scale_up*I_observe))
                # LossR_Net_obs = tf.reduce_mean(tf.square(scale_up*R_NN - scale_up*R_observe))
                # LossD_Net_obs = tf.reduce_mean(tf.square(scale_up*D_NN - scale_up*D_observe))
                LossN_Net_obs = tf.reduce_mean(tf.square(scale_up*N_NN - scale_up*N_observe))

                Loss2dS = tf.reduce_mean(tf.square(dS_NN2t - temp_snn2t))
                Loss2dI = tf.reduce_mean(tf.square(dI_NN2t - temp_inn2t))
                Loss2dR = tf.reduce_mean(tf.square(dR_NN2t - temp_rnn2t))
                Loss2dN = tf.reduce_mean(tf.square(dN_NN2t))
                Loss2dD = tf.reduce_mean(tf.square(dD_NN2t - temp_dnn2t))
            elif str.lower(R['loss_function']) == 'lncosh_loss' and R['scale_up'] == 0:
                # LossS_Net_obs = tf.reduce_mean(tf.ln(tf.cosh(S_NN - S_observe)))
                LossI_Net_obs = tf.reduce_mean(tf.log(tf.cosh(I_NN - I_observe)))
                # LossR_Net_obs = tf.reduce_mean(tf.log(tf.cosh(R_NN - R_observe)))
                # LossD_Net_obs = tf.reduce_mean(tf.log(tf.cosh(D_NN - D_observe)))
                LossN_Net_obs = tf.reduce_mean(tf.log(tf.cosh(N_NN - N_observe)))

                Loss2dS = tf.reduce_mean(tf.log(tf.cosh(dS_NN2t - temp_snn2t)))
                Loss2dI = tf.reduce_mean(tf.log(tf.cosh(dI_NN2t - temp_inn2t)))
                Loss2dR = tf.reduce_mean(tf.log(tf.cosh(dR_NN2t - temp_rnn2t)))
                Loss2dD = tf.reduce_mean(tf.log(tf.cosh(dD_NN2t - temp_dnn2t)))
                Loss2dN = tf.reduce_mean(tf.log(tf.cosh(dN_NN2t)))
            elif str.lower(R['loss_function']) == 'lncosh_loss' and R['scale_up'] == 1:
                scale_up = R['scale_factor']
                # LossS_Net_obs = tf.reduce_mean(tf.ln(tf.cosh(scale_up*S_NN - scale_up*S_observe)))
                LossI_Net_obs = tf.reduce_mean(tf.log(tf.cosh(scale_up*I_NN - scale_up*I_observe)))
                # LossR_Net_obs = tf.reduce_mean(tf.log(tf.cosh(scale_up*R_NN - scale_up*R_observe)))
                # LossD_Net_obs = tf.reduce_mean(tf.log(tf.cosh(scale_up*D_NN - scale_up*D_observe)))
                LossN_Net_obs = tf.reduce_mean(tf.log(tf.cosh(scale_up*N_NN - scale_up*N_observe)))

                Loss2dS = tf.reduce_mean(tf.log(tf.cosh(dS_NN2t - temp_snn2t)))
                Loss2dI = tf.reduce_mean(tf.log(tf.cosh(dI_NN2t - temp_inn2t)))
                Loss2dR = tf.reduce_mean(tf.log(tf.cosh(dR_NN2t - temp_rnn2t)))
                Loss2dD = tf.reduce_mean(tf.log(tf.cosh(dD_NN2t - temp_dnn2t)))
                Loss2dN = tf.reduce_mean(tf.log(tf.cosh(dN_NN2t)))

            if R['regular_weight_model'] == 'L1':
                regular_WB2S = DNN_base.regular_weights_biases_L1(Weight2S, Bias2S)
                regular_WB2I = DNN_base.regular_weights_biases_L1(Weight2I, Bias2I)
                regular_WB2R = DNN_base.regular_weights_biases_L1(Weight2R, Bias2R)
                regular_WB2D = DNN_base.regular_weights_biases_L1(Weight2D, Bias2D)
                regular_WB2Beta = DNN_base.regular_weights_biases_L1(Weight2beta, Bias2beta)
                regular_WB2Gamma = DNN_base.regular_weights_biases_L1(Weight2gamma, Bias2gamma)
            elif R['regular_weight_model'] == 'L2':
                regular_WB2S = DNN_base.regular_weights_biases_L2(Weight2S, Bias2S)
                regular_WB2I = DNN_base.regular_weights_biases_L2(Weight2I, Bias2I)
                regular_WB2R = DNN_base.regular_weights_biases_L2(Weight2R, Bias2R)
                regular_WB2D = DNN_base.regular_weights_biases_L2(Weight2D, Bias2D)
                regular_WB2Beta = DNN_base.regular_weights_biases_L2(Weight2beta, Bias2beta)
                regular_WB2Gamma = DNN_base.regular_weights_biases_L2(Weight2gamma, Bias2gamma)
            else:
                regular_WB2S = tf.constant(0.0)
                regular_WB2I = tf.constant(0.0)
                regular_WB2R = tf.constant(0.0)
                regular_WB2D = tf.constant(0.0)
                regular_WB2Beta = tf.constant(0.0)
                regular_WB2Gamma = tf.constant(0.0)

            PWB2S = wb_penalty*regular_WB2S
            PWB2I = wb_penalty*regular_WB2I
            PWB2R = wb_penalty*regular_WB2R
            PWB2D = wb_penalty * regular_WB2D
            PWB2Beta = wb_penalty * regular_WB2Beta
            PWB2Gamma = wb_penalty * regular_WB2Gamma

            Loss2S = Loss2dS + PWB2S
            Loss2I = predict_true_penalty * LossI_Net_obs + Loss2dI + PWB2I
            Loss2R = Loss2dR + PWB2R
            Loss2D = Loss2dD + PWB2D
            Loss2N = LossN_Net_obs + Loss2dN

            Loss = Loss2S + Loss2I + Loss2R + Loss2D + Loss2N + PWB2Beta + PWB2Gamma

            my_optimizer = tf.compat.v1.train.AdamOptimizer(in_learning_rate)
            if R['train_model'] == 'train_group':
                train_Loss2S = my_optimizer.minimize(Loss2S, global_step=global_steps)
                train_Loss2I = my_optimizer.minimize(Loss2I, global_step=global_steps)
                train_Loss2R = my_optimizer.minimize(Loss2R, global_step=global_steps)
                train_Loss2D = my_optimizer.minimize(Loss2D, global_step=global_steps)
                train_Loss2N = my_optimizer.minimize(Loss2N, global_step=global_steps)
                train_Losses = tf.group(train_Loss2S, train_Loss2I, train_Loss2R, train_Loss2D, train_Loss2N)
            elif R['train_model'] == 'train_union_loss':
                train_Losses = my_optimizer.minimize(Loss, global_step=global_steps)

    t0 = time.time()
    loss_s_all, loss_i_all, loss_r_all, loss_d_all, loss_n_all, loss_all = [], [], [], [], [], []
    test_epoch = []
    test_mse2I_all, test_rel2I_all = [], []

    # filename = 'data2csv/Wuhan.csv'
    # filename = 'data2csv/Italia_data.csv'
    filename = 'data2csv/Korea_data.csv'
    # filename = 'data2csv/minnesota.csv'
    date, data2I, data2S = DNN_data.load_2csvData_cal_S(datafile=filename, total_population=R['total_population'])

    assert (trainSet_szie + batchSize_test <= len(data2I))
    if R['normalize_population'] == 1:
        # 不归一化数据
        train_date, train_data2i, train_data2s, test_date, test_data2i, test_data2s = \
            DNN_data.split_3csvData2train_test(date, data2I, data2S, size2train=trainSet_szie, normalFactor=1.0)
        nbatch2train = np.ones(batchSize_train, dtype=np.float32) * float(R['total_population'])

    elif (R['total_population'] != R['normalize_population']) and R['normalize_population'] != 1:
        # 归一化数据，使用的归一化数值小于总“人口”
        train_date, train_data2i, train_data2s, test_date, test_data2i, test_data2s = \
            DNN_data.split_3csvData2train_test(date, data2I, data2S, size2train=trainSet_szie,
                                               normalFactor=R['normalize_population'])
        nbatch2train = np.ones(batchSize_train, dtype=np.float32) * (
                    float(R['total_population']) / float(R['normalize_population']))

    elif (R['total_population'] == R['normalize_population']) and R['normalize_population'] != 1:
        # 归一化数据，使用总“人口”归一化数据
        train_date, train_data2i, train_data2s, test_date, test_data2i, test_data2s = \
            DNN_data.split_3csvData2train_test(date, data2I, data2S, size2train=trainSet_szie,
                                               normalFactor=R['normalize_population'])
        nbatch2train = np.ones(batchSize_train, dtype=np.float32)

    # 对于时间数据来说，验证模型的合理性，要用连续的时间数据验证.
    test_t_bach = DNN_data.sample_testDays_serially(test_date, batchSize_test)

    # 由于将数据拆分为训练数据和测试数据时，进行了归一化处理，故这里不用归一化
    i_obs_test = DNN_data.sample_testData_serially(test_data2i, batchSize_test, normalFactor=1.0)
    s_obs_test = DNN_data.sample_testData_serially(test_data2s, batchSize_test, normalFactor=1.0)

    print('The test data about i:\n', str(np.transpose(i_obs_test)))
    print('\n')
    print('The test data about s:\n', str(np.transpose(s_obs_test)))
    print('\n')
    DNN_tools.log_string('The test data about i:\n%s\n' % str(np.transpose(i_obs_test)), log_fileout)
    DNN_tools.log_string('The test data about s:\n%s\n' % str(np.transpose(s_obs_test)), log_fileout)

    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True  # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True  # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        tmp_lr = init_lr
        for i_epoch in range(R['max_epoch'] + 1):
            t_batch, i_obs = \
                DNN_data.randSample_Normalize_existData(train_date, train_data2i, batchsize=batchSize_train,
                                                        normalFactor=1.0, sampling_opt=R['opt2sample'])
            n_obs = nbatch2train.reshape(batchSize_train, 1)
            tmp_lr = tmp_lr * (1 - lr_decay)
            if R['activate_stage_penalty'] == 1:
                if i_epoch < int(R['max_epoch'] / 10):
                    temp_penalty_pt = pt_penalty_init
                elif i_epoch < int(R['max_epoch'] / 5):
                    temp_penalty_pt = 10 * pt_penalty_init
                elif i_epoch < int(R['max_epoch'] / 4):
                    temp_penalty_pt = 50 * pt_penalty_init
                elif i_epoch < int(R['max_epoch'] / 2):
                    temp_penalty_pt = 100 * pt_penalty_init
                elif i_epoch < int(3 * R['max_epoch'] / 4):
                    temp_penalty_pt = 200 * pt_penalty_init
                else:
                    temp_penalty_pt = 500 * pt_penalty_init
            elif R['activate_stage_penalty'] == 2:
                if i_epoch < int(R['max_epoch'] / 3):
                    temp_penalty_pt = pt_penalty_init
                elif i_epoch < 2 * int(R['max_epoch'] / 3):
                    temp_penalty_pt = 10 * pt_penalty_init
                else:
                    temp_penalty_pt = 50 * pt_penalty_init
            else:
                temp_penalty_pt = pt_penalty_init

            _, loss_s, loss_i, loss_r, loss_d, loss_n, loss, pwb2s, pwb2i, pwb2r, pwb2d = sess.run(
                [train_Losses, Loss2S, Loss2I, Loss2R, Loss2D, Loss2N, Loss, PWB2S, PWB2I, PWB2R, PWB2D],
                feed_dict={T_it: t_batch, I_observe: i_obs, N_observe: n_obs, in_learning_rate: tmp_lr,
                           predict_true_penalty: temp_penalty_pt})

            loss_s_all.append(loss_s)
            loss_i_all.append(loss_i)
            loss_r_all.append(loss_r)
            loss_d_all.append(loss_d)
            loss_n_all.append(loss_n)
            loss_all.append(loss)

            if i_epoch % 1000 == 0:
                # 以下代码为输出训练过程中 S_NN, I_NN, R_NN, beta, gamma 的训练结果
                print_and_log2train(i_epoch, time.time() - t0, tmp_lr, temp_penalty_pt, pwb2s, pwb2i, pwb2r, loss_s,
                                    loss_i, loss_r, loss_d, loss_n, loss, log_out=log_fileout)

                s_nn2train, i_nn2train, r_nn2train, d_nn2train = sess.run(
                    [S_NN, I_NN, R_NN, D_NN], feed_dict={T_it: np.reshape(train_date, [-1, 1])})

                # 以下代码为输出训练过程中 S_NN, I_NN, R_NN, beta, gamma 的测试结果
                test_epoch.append(i_epoch / 1000)
                s_nn2test, i_nn2test, r_nn2test, d_nn2test, beta_test, gamma_test = sess.run(
                    [S_NN, I_NN, R_NN, D_NN, beta, gamma], feed_dict={T_it: test_t_bach})
                point_ERR2I = np.square(i_nn2test - i_obs_test)
                test_mse2I = np.mean(point_ERR2I)
                test_mse2I_all.append(test_mse2I)
                test_rel2I = test_mse2I / np.mean(np.square(i_obs_test))
                test_rel2I_all.append(test_rel2I)

                DNN_tools.print_and_log_test_one_epoch(test_mse2I, test_rel2I, log_out=log_fileout)
                DNN_tools.log_string('------------------The epoch----------------------: %s\n' % str(i_epoch),
                                     log2testSolus)
                DNN_tools.log_string('The test result for s:\n%s\n' % str(np.transpose(s_nn2test)), log2testSolus)
                DNN_tools.log_string('The test result for i:\n%s\n' % str(np.transpose(i_nn2test)), log2testSolus)
                DNN_tools.log_string('The test result for r:\n%s\n\n' % str(np.transpose(r_nn2test)), log2testSolus)
                DNN_tools.log_string('The test result for d:\n%s\n\n' % str(np.transpose(d_nn2test)), log2testSolus)

                # --------以下代码为输出训练过程中 S_NN_temp, I_NN_temp, R_NN_temp, in_beta, in_gamma 的测试结果-------------
                s_nn_temp2test, i_nn_temp2test, r_nn_temp2test, d_nn_temp2test, in_beta_test, in_gamma_test = sess.run(
                    [SNN_temp, INN_temp, RNN_temp, DNN_temp, in_beta, in_gamma],
                    feed_dict={T_it: test_t_bach})

                DNN_tools.log_string('------------------The epoch----------------------: %s\n' % str(i_epoch),
                                     log2testSolus2)
                DNN_tools.log_string('The test result for s_temp:\n%s\n' % str(np.transpose(s_nn_temp2test)),
                                     log2testSolus2)
                DNN_tools.log_string('The test result for i_temp:\n%s\n' % str(np.transpose(i_nn_temp2test)),
                                     log2testSolus2)
                DNN_tools.log_string('The test result for r_temp:\n%s\n\n' % str(np.transpose(r_nn_temp2test)),
                                     log2testSolus2)
                DNN_tools.log_string('The test result for d_temp:\n%s\n\n' % str(np.transpose(d_nn_temp2test)),
                                     log2testSolus2)

                DNN_tools.log_string('------------------The epoch----------------------: %s\n' % str(i_epoch),
                                     log2testParas)
                DNN_tools.log_string('The test result for in_beta:\n%s\n' % str(np.transpose(in_beta_test)),
                                     log2testParas)
                DNN_tools.log_string('The test result for in_gamma:\n%s\n' % str(np.transpose(in_gamma_test)),
                                     log2testParas)

        DNN_tools.log_string('The train result for S:\n%s\n' % str(np.transpose(s_nn2train)), log2trianSolus)
        DNN_tools.log_string('The train result for I:\n%s\n' % str(np.transpose(i_nn2train)), log2trianSolus)
        DNN_tools.log_string('The train result for R:\n%s\n\n' % str(np.transpose(r_nn2train)), log2trianSolus)

        saveData.true_value2convid(train_data2i, name2Array='itrue2train', outPath=R['FolderName'])
        saveData.save_Solu2mat_Covid(s_nn2train, name2solus='s2train', outPath=R['FolderName'])
        saveData.save_Solu2mat_Covid(i_nn2train, name2solus='i2train', outPath=R['FolderName'])
        saveData.save_Solu2mat_Covid(r_nn2train, name2solus='r2train', outPath=R['FolderName'])
        saveData.save_Solu2mat_Covid(d_nn2train, name2solus='d2train', outPath=R['FolderName'])

        saveData.save_SIR_trainLoss2mat_Covid(loss_s_all, loss_i_all, loss_r_all, loss_n_all, actName=act_func2SIRD,
                                              outPath=R['FolderName'])

        plotData.plotTrain_loss_1act_func(loss_s_all, lossType='loss2s', seedNo=R['seed'], outPath=R['FolderName'],
                                          yaxis_scale=True)
        plotData.plotTrain_loss_1act_func(loss_i_all, lossType='loss2i', seedNo=R['seed'], outPath=R['FolderName'],
                                          yaxis_scale=True)
        plotData.plotTrain_loss_1act_func(loss_r_all, lossType='loss2r', seedNo=R['seed'], outPath=R['FolderName'],
                                          yaxis_scale=True)
        plotData.plotTrain_loss_1act_func(loss_n_all, lossType='loss2n', seedNo=R['seed'], outPath=R['FolderName'],
                                          yaxis_scale=True)

        saveData.true_value2convid(i_obs_test, name2Array='i_true2test', outPath=R['FolderName'])
        saveData.save_testMSE_REL2mat(test_mse2I_all, test_rel2I_all, actName='Infected', outPath=R['FolderName'])
        plotData.plotTest_MSE_REL(test_mse2I_all, test_rel2I_all, test_epoch, actName='Infected', seedNo=R['seed'],
                                  outPath=R['FolderName'], yaxis_scale=True)
        saveData.save_SIR_testSolus2mat_Covid(s_nn2test, i_nn2test, r_nn2test, name2solus1='snn2test',
                                              name2solus2='inn2test', name2solus3='rnn2test', outPath=R['FolderName'])
        saveData.save_SIR_testParas2mat_Covid(beta_test, gamma_test, name2para1='beta2test', name2para2='gamma2test',
                                              outPath=R['FolderName'])

        plotData.plot_testSolu2convid(i_obs_test, name2solu='i_true', coord_points2test=test_t_bach,
                                      outPath=R['FolderName'])
        plotData.plot_testSolu2convid(s_nn2test, name2solu='s_test', coord_points2test=test_t_bach,
                                      outPath=R['FolderName'])
        plotData.plot_testSolu2convid(i_nn2test, name2solu='i_test', coord_points2test=test_t_bach,
                                      outPath=R['FolderName'])
        plotData.plot_testSolu2convid(r_nn2test, name2solu='r_test', coord_points2test=test_t_bach,
                                      outPath=R['FolderName'])

        plotData.plot_testSolus2convid(i_obs_test, i_nn2test, name2solu1='i_true', name2solu2='i_test',
                                       coord_points2test=test_t_bach, seedNo=R['seed'], outPath=R['FolderName'])

        plotData.plot_testSolu2convid(beta_test, name2solu='beta_test', coord_points2test=test_t_bach,
                                      outPath=R['FolderName'])
        plotData.plot_testSolu2convid(gamma_test, name2solu='gamma_test', coord_points2test=test_t_bach,
                                      outPath=R['FolderName'])


if __name__ == "__main__":
    R = {}
    R['gpuNo'] = 0  # 默认使用 GPU，这个标记就不要设为-1，设为0,1,2,3,4....n（n指GPU的数目，即电脑有多少块GPU）

    # 文件保存路径设置
    store_file = 'SIRD2covid'
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(BASE_DIR)
    OUT_DIR = os.path.join(BASE_DIR, store_file)
    if not os.path.exists(OUT_DIR):
        print('---------------------- OUT_DIR ---------------------:', OUT_DIR)
        os.mkdir(OUT_DIR)

    R['seed'] = np.random.randint(1e5)
    seed_str = str(R['seed'])  # int 型转为字符串型
    FolderName = os.path.join(OUT_DIR, seed_str)  # 路径连接
    R['FolderName'] = FolderName
    if not os.path.exists(FolderName):
        print('--------------------- FolderName -----------------:', FolderName)
        os.mkdir(FolderName)

    # ----------------------------------------  复制并保存当前文件 -----------------------------------------
    if platform.system() == 'Windows':
        tf.compat.v1.reset_default_graph()
        shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))
    else:
        shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))

    # if the value of step_stop_flag is not 0, it will activate stop condition of step to kill program
    step_stop_flag = input('please input an  integer number to activate step-stop----0:no---!0:yes--:')
    R['activate_stop'] = int(step_stop_flag)
    # if the value of step_stop_flag is not 0, it will activate stop condition of step to kill program
    R['max_epoch'] = 200000
    if 0 != R['activate_stop']:
        epoch_stop = input('please input a stop epoch:')
        R['max_epoch'] = int(epoch_stop)

    # ----------------------------------------- Convid 设置 ---------------------------------
    R['eqs_name'] = 'SIRD'
    R['input_dim'] = 1                       # 输入维数，即问题的维数(几元问题)
    R['output_dim'] = 1                      # 输出维数
    R['total_population'] = 9776000          # 总的“人口”数量

    R['normalize_population'] = 9776000      # 归一化时使用的“人口”数值
    # R['normalize_population'] = 100000
    # R['normalize_population'] = 1

    # ------------------------------------  神经网络的设置  ----------------------------------------
    R['size2train'] = 70                    # 训练集的大小
    R['batch_size2train'] = 20              # 训练数据的批大小
    R['batch_size2test'] = 10               # 训练数据的批大小
    # R['opt2sample'] = 'random_sample'     # 训练集的选取方式--随机采样
    # R['opt2sample'] = 'rand_sample_sort'    # 训练集的选取方式--随机采样后按时间排序
    R['opt2sample'] = 'windows_rand_sample'  # 训练集的选取方式--随机窗口采样(以随机点为基准，然后滑动窗口采样)

    R['init_penalty2predict_true'] = 50     # Regularization parameter for boundary conditions
    R['activate_stage_penalty'] = 1         # 是否开启阶段调整惩罚项，0 代表不调整，非 0 代表调整
    if R['activate_stage_penalty'] == 1 or R['activate_stage_penalty'] == 2:
        # R['init_penalty2predict_true'] = 1000
        # R['init_penalty2predict_true'] = 100
        # R['init_penalty2predict_true'] = 50
        # R['init_penalty2predict_true'] = 20
        R['init_penalty2predict_true'] = 1

    # R['regular_weight_model'] = 'L0'
    # R['regular_weight'] = 0.000             # Regularization parameter for weights

    # R['regular_weight_model'] = 'L1'
    R['regular_weight_model'] = 'L2'          # The model of regular weights and biases
    # R['regular_weight'] = 0.001             # Regularization parameter for weights
    # R['regular_weight'] = 0.0005            # Regularization parameter for weights
    # R['regular_weight'] = 0.0001            # Regularization parameter for weights
    R['regular_weight'] = 0.00005             # Regularization parameter for weights
    # R['regular_weight'] = 0.00001           # Regularization parameter for weights

    R['optimizer_name'] = 'Adam'              # 优化器
    R['loss_function'] = 'L2_loss'            # 损失函数的类型
    # R['loss_function'] = 'lncosh_loss'      # 损失函数的类型
    R['scale_up'] = 1                         # scale_up 用来控制湿粉扑对数值进行尺度提升，如1e-6量级提升到1e-2量级。不为 0 代表开启提升
    R['scale_factor'] = 100                   # scale_factor 用来对数值进行尺度提升，如1e-6量级提升到1e-2量级

    # R['train_model'] = 'train_group'        # 训练模式:各个不同的loss捆绑打包训练
    R['train_model'] = 'train_union_loss'     # 训练模式:各个不同的loss累加在一起，训练

    if 50000 < R['max_epoch']:
        R['learning_rate'] = 2e-3             # 学习率
        R['lr_decay'] = 1e-4                  # 学习率 decay
        # R['learning_rate'] = 2e-4           # 学习率
        # R['lr_decay'] = 5e-5                # 学习率 decay
    elif (20000 < R['max_epoch'] and 50000 >= R['max_epoch']):
        # R['learning_rate'] = 1e-3           # 学习率
        # R['lr_decay'] = 1e-4                # 学习率 decay
        # R['learning_rate'] = 2e-4           # 学习率
        # R['lr_decay'] = 1e-4                # 学习率 decay
        R['learning_rate'] = 1e-4             # 学习率
        R['lr_decay'] = 5e-5                  # 学习率 decay
    else:
        R['learning_rate'] = 5e-5             # 学习率
        R['lr_decay'] = 1e-5                  # 学习率 decay

    # SIRD和参数网络模型的选择
    # R['model2SIRD'] = 'DNN'
    # R['model2SIRD'] = 'DNN_scale'
    # R['model2SIRD'] = 'DNN_scaleOut'
    R['model2SIRD'] = 'DNN_FourierBase'

    # R['model2paras'] = 'DNN'
    # R['model2paras'] = 'DNN_scale'
    # R['model2paras'] = 'DNN_scaleOut'
    R['model2paras'] = 'DNN_FourierBase'

    # SIRD和参数网络模型的隐藏层单元数目
    if R['model2SIRD'] == 'DNN_FourierBase':
        R['hidden2SIRD'] = (35, 50, 30, 30, 20)  # 1*50+50*50+50*30+30*30+30*20+20*1 = 5570
    else:
        # R['hidden2SIRD'] = (10, 10, 8, 6, 6, 3)        # it is used to debug our work
        R['hidden2SIRD'] = (70, 50, 30, 30, 20)  # 1*50+50*50+50*30+30*30+30*20+20*1 = 5570
        # R['hidden2SIRD'] = (80, 80, 60, 40, 40, 20)    # 80+80*80+80*60+60*40+40*40+40*20+20*1 = 16100
        # R['hidden2SIRD'] = (100, 100, 80, 60, 60, 40)
        # R['hidden2SIRD'] = (200, 100, 100, 80, 50, 50)

    if R['model2paras'] == 'DNN_FourierBase':
        R['hidden2para'] = (35, 50, 30, 30, 20)  # 1*50+50*50+50*30+30*30+30*20+20*1 = 5570
    else:
        # R['hidden2para'] = (10, 10, 8, 6, 6, 3)       # it is used to debug our work
        R['hidden2para'] = (70, 50, 30, 30, 20)  # 1*50+50*50+50*30+30*30+30*20+20*1 = 5570
        # R['hidden2para'] = (80, 80, 60, 40, 40, 20)   # 80+80*80+80*60+60*40+40*40+40*20+20*1 = 16100
        # R['hidden2para'] = (100, 100, 80, 60, 60, 40)
        # R['hidden2para'] = (200, 100, 100, 80, 50, 50)

    # SIRD和参数网络模型的尺度因子
    if R['model2SIRD'] != 'DNN':
        R['freq2SIRD'] = np.concatenate(([1], np.arange(1, 20)), axis=0)
    if R['model2paras'] != 'DNN':
        R['freq2paras'] = np.concatenate(([1], np.arange(1, 20)), axis=0)

    # SIRD和参数网络模型为傅里叶网络和尺度网络时，重复高频因子或者低频因子
    if R['model2SIRD'] == 'DNN_FourierBase' or R['model2SIRD'] == 'DNN_scale':
        R['if_repeat_High_freq2SIRD'] = False
    if R['model2paras'] == 'DNN_FourierBase' or R['model2paras'] == 'DNN_scale':
        R['if_repeat_High_freq2paras'] = False

    # SIRD和参数网络模型的激活函数的选择
    # R['actIn_Name2SIRD'] = 'relu'
    # R['actIn_Name2SIRD'] = 'leaky_relu'
    # R['actIn_Name2SIRD'] = 'sigmod'
    R['actIn_Name2SIRD'] = 'tanh'
    # R['actIn_Name2SIRD'] = 'srelu'
    # R['actIn_Name2SIRD'] = 's2relu'
    # R['actIn_Name2SIRD'] = 'sin'
    # R['actIn_Name2SIRD'] = 'sinAddcos'
    # R['actIn_Name2SIRD'] = 'elu'
    # R['actIn_Name2SIRD'] = 'gelu'
    # R['actIn_Name2SIRD'] = 'mgelu'
    # R['actIn_Name2SIRD'] = 'linear'

    # R['act_Name2SIRD'] = 'relu'
    # R['act_Name2SIRD'] = 'leaky_relu'
    # R['act_Name2SIRD'] = 'sigmod'
    R['act_Name2SIRD'] = 'tanh'  # 这个激活函数比较s2ReLU合适
    # R['act_Name2SIRD'] = 'srelu'
    # R['act_Name2SIRD'] = 's2relu'
    # R['act_Name2SIRD'] = 'sin'
    # R['act_Name2SIRD'] = 'sinAddcos'
    # R['act_Name2SIRD'] = 'elu'
    # R['act_Name2SIRD'] = 'gelu'
    # R['act_Name2SIRD'] = 'mgelu'
    # R['act_Name2SIRD'] = 'linear'

    # R['actIn_Name2paras'] = 'relu'
    # R['actIn_Name2paras'] = 'leaky_relu'
    # R['actIn_Name2paras'] = 'sigmoid'
    R['actIn_Name2paras'] = 'tanh'
    # R['actIn_Name2paras'] = 'srelu'
    # R['actIn_Name2paras'] = 's2relu'
    # R['actIn_Name2paras'] = 'sin'
    # R['actIn_Name2paras'] = 'sinAddcod'
    # R['actIn_Name2paras'] = 'elu'
    # R['actIn_Name2paras'] = 'gelu'
    # R['actIn_Name2paras'] = 'mgelu'
    # R['actIn_Name2paras'] = 'linear'

    # R['act_Name2paras'] = 'relu'
    # R['act_Name2paras'] = 'leaky_relu'
    # R['act_Name2paras'] = 'sigmoid'
    R['act_Name2paras'] = 'tanh'  # 这个激活函数比较s2ReLU合适
    # R['act_Name2paras'] = 'srelu'
    # R['act_Name2paras'] = 's2relu'
    # R['act_Name2paras'] = 'sin'
    # R['act_Name2paras'] = 'sinAddcos'
    # R['act_Name2paras'] = 'elu'
    # R['act_Name2paras'] = 'gelu'
    # R['act_Name2paras'] = 'mgelu'
    # R['act_Name2paras'] = 'linear'

    solve_SIRD2COVID(R)

