"""
@author: xi'an Li
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


def act_gauss(input):
    # out = tf.exp(-0.25*tf.multiply(input, input))
    out = tf.exp(-0.5 * tf.multiply(input, input))
    return out


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

    DNN_tools.log_string('The training model for all networks: %s\n' % str(R_dic['train_model']), log_fileout)

    if (R_dic['optimizer_name']).title() == 'Adam':
        DNN_tools.log_string('optimizer:%s\n' % str(R_dic['optimizer_name']), log_fileout)
    else:
        DNN_tools.log_string('optimizer:%s  with momentum=%f\n' % (R_dic['optimizer_name'], R_dic['momentum']), log_fileout)

    DNN_tools.log_string('Init learning rate: %s\n' % str(R_dic['learning_rate']), log_fileout)
    DNN_tools.log_string('Decay to learning rate: %s\n' % str(R_dic['lr_decay']), log_fileout)
    DNN_tools.log_string('The type for Loss function: %s\n' % str(R_dic['loss_function']), log_fileout)

    if R_dic['activate_stop'] != 0:
        DNN_tools.log_string('activate the stop_step and given_step= %s\n' % str(R_dic['max_epoch']), log_fileout)
    else:
        DNN_tools.log_string('no activate the stop_step and given_step = default: %s\n' % str(R_dic['max_epoch']), log_fileout)

    DNN_tools.log_string(
        'Initial penalty for difference of predict and true: %s\n' % str(R_dic['init_penalty2predict_true']), log_fileout)

    if R_dic['activate_stage_penalty'] == 0:
        DNN_tools.log_string('Unchanging penalty for predict and true!!!\n', log_fileout)
    else:
        DNN_tools.log_string('Increasing penalty for predict and true!!!\n', log_fileout)

    DNN_tools.log_string('The model of regular weights and biases: %s\n' % str(R_dic['regular_weight_model']), log_fileout)

    DNN_tools.log_string('Regularizing scale of weights and biases for ODE: %s\n' % str(R_dic['regular_weight2ODE']), log_fileout)

    DNN_tools.log_string('Regularizing scale of weights and biases for Paras: %s\n' % str(R_dic['regular_weight2Paras']),
                         log_fileout)

    DNN_tools.log_string('Size 2 training set: %s\n' % str(R_dic['size2train']), log_fileout)

    DNN_tools.log_string('Batch-size 2 training: %s\n' % str(R_dic['batch_size2train']), log_fileout)

    DNN_tools.log_string('Batch-size 2 testing: %s\n' % str(R_dic['batch_size2test']), log_fileout)


def print_and_log2train(i_epoch, run_time, tmp_lr, temp_penalty_nt, penalty_wb2s, penalty_wb2i, penalty_wb2r,
                        penalty_wb2d, penalty_wb2beta, penalty_wb2gamma, penalty_wb2mu, loss_s, loss_i, loss_r, loss_d,
                        loss_n, loss2all, log_out=None):
    print('train epoch: %d, time: %.3f' % (i_epoch, run_time))
    print('learning rate: %f' % tmp_lr)
    print('penalty for difference of predict and true : %f' % temp_penalty_nt)
    print('penalty weights and biases for S: %.10f' % penalty_wb2s)
    print('penalty weights and biases for I: %.10f' % penalty_wb2i)
    print('penalty weights and biases for R: %.10f' % penalty_wb2r)
    print('penalty weights and biases for D: %.10f' % penalty_wb2d)
    print('penalty weights and biases for Beta: %.10f' % penalty_wb2beta)
    print('penalty weights and biases for Gamma: %.10f' % penalty_wb2gamma)
    print('penalty weights and biases for Mu: %.10f' % penalty_wb2mu)

    print('loss for S with penalty: %.16f' % loss_s)
    print('loss for I with penalty: %.16f' % loss_i)
    print('loss for R with penalty: %.16f' % loss_r)
    print('loss for D with penalty: %.16f' % loss_d)
    print('loss for N with penalty: %.16f' % loss_n)
    print('total loss with penalty: %.16f\n' % loss2all)

    DNN_tools.log_string('train epoch: %d,time: %.3f' % (i_epoch, run_time), log_out)
    DNN_tools.log_string('learning rate: %f' % tmp_lr, log_out)
    DNN_tools.log_string('penalty for difference of predict and true : %f' % temp_penalty_nt, log_out)
    DNN_tools.log_string('penalty weights and biases for S: %f' % penalty_wb2s, log_out)
    DNN_tools.log_string('penalty weights and biases for I: %f' % penalty_wb2i, log_out)
    DNN_tools.log_string('penalty weights and biases for R: %.10f' % penalty_wb2r, log_out)
    DNN_tools.log_string('penalty weights and biases for D: %.10f' % penalty_wb2d, log_out)
    DNN_tools.log_string('penalty weights and biases for Beta: %.10f' % penalty_wb2beta, log_out)
    DNN_tools.log_string('penalty weights and biases for Gamma: %.10f' % penalty_wb2gamma, log_out)
    DNN_tools.log_string('penalty weights and biases for Mu: %.10f' % penalty_wb2mu, log_out)

    DNN_tools.log_string('loss for S with penalty: %.16f' % loss_s, log_out)
    DNN_tools.log_string('loss for I with penalty: %.16f' % loss_i, log_out)
    DNN_tools.log_string('loss for R with penalty: %.16f' % loss_r, log_out)
    DNN_tools.log_string('loss for D with penalty: %.16f' % loss_d, log_out)
    DNN_tools.log_string('loss for N with penalty: %.16f' % loss_n, log_out)
    DNN_tools.log_string('total loss with penalty: %.16f \n\n' % loss2all, log_out)


def print_and_log_test_one_epoch(test_mes2S, test_rel2S, test_mes2I, test_rel2I, test_mes2R, test_rel2R, test_mes2D,
                                 test_rel2D, log_out=None):
    print('mean square error of predict and real for S in testing: %.10f' % test_mes2S)
    print('residual error of predict and real for S in testing: %.10f\n' % test_rel2S)

    print('mean square error of predict and real for I in testing: %.10f' % test_mes2I)
    print('residual error of predict and real for I in testing: %.10f\n' % test_rel2I)

    print('mean square error of predict and real for R in testing: %.10f' % test_mes2R)
    print('residual error of predict and real for R in testing: %.10f\n' % test_rel2R)

    print('mean square error of predict and real for D in testing: %.10f' % test_mes2D)
    print('residual error of predict and real for D in testing: %.10f\n' % test_rel2D)

    DNN_tools.log_string('mean square error of predict and real for S in testing: %.10f' % test_mes2S, log_out)
    DNN_tools.log_string('residual error of predict and real for S in testing: %.10f\n\n' % test_rel2S, log_out)

    DNN_tools.log_string('mean square error of predict and real for I in testing: %.10f' % test_mes2I, log_out)
    DNN_tools.log_string('residual error of predict and real for I in testing: %.10f\n\n' % test_rel2I, log_out)

    DNN_tools.log_string('mean square error of predict and real for R in testing: %.10f' % test_mes2R, log_out)
    DNN_tools.log_string('residual error of predict and real for R in testing: %.10f\n\n' % test_rel2R, log_out)

    DNN_tools.log_string('mean square error of predict and real for D in testing: %.10f' % test_mes2D, log_out)
    DNN_tools.log_string('residual error of predict and real for D in testing: %.10f\n\n' % test_rel2D, log_out)


def solve_SIRD2COVID(R):
    log_out_path = R['FolderName']        # 将路径从字典 R 中提取出来
    if not os.path.exists(log_out_path):  # 判断路径是否已经存在
        os.mkdir(log_out_path)            # 无 log_out_path 路径，创建一个 log_out_path 路径
    log_fileout = open(os.path.join(log_out_path, 'log_train.txt'), 'w')  # 在这个路径下创建并打开一个可写的 log_train.txt文件
    dictionary_out2file(R, log_fileout)

    trainSet_szie = R['size2train']                   # 训练集大小,给定一个数据集，拆分训练集和测试集时，需要多大规模的训练集
    batchSize_train = R['batch_size2train']           # 训练批量的大小,该值远小于训练集大小
    batchSize_test = R['batch_size2test']             # 测试批量的大小,该值小于等于测试集大小
    pt_penalty_init = R['init_penalty2predict_true']  # 预测值和真值得误差惩罚因子的初值,用于处理那些具有真实值得变量
    wb_penalty2ode = R['regular_weight2ODE']                  # 神经网络参数的惩罚因子
    wb_penalty2paras = R['regular_weight2Paras']  # 神经网络参数的惩罚因子
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
    flag2mu = 'WB2mu'
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
        Weight2mu, Bias2mu = DNN_base.Xavier_init_NN_Fourier(input_dim, out_dim, hidden_para, flag2mu)
    else:
        Weight2beta, Bias2beta = DNN_base.Xavier_init_NN(input_dim, out_dim, hidden_para, flag2beta)
        Weight2gamma, Bias2gamma = DNN_base.Xavier_init_NN(input_dim, out_dim, hidden_para, flag2gamma)
        Weight2mu, Bias2mu = DNN_base.Xavier_init_NN(input_dim, out_dim, hidden_para, flag2mu)

    global_steps = tf.Variable(0, trainable=False)
    with tf.device('/gpu:%s' % (R['gpuNo'])):
        with tf.compat.v1.variable_scope('vscope', reuse=tf.compat.v1.AUTO_REUSE):
            T_it = tf.compat.v1.placeholder(tf.float32, name='T_it', shape=[None, out_dim])
            S_observe = tf.compat.v1.placeholder(tf.float32, name='S_observe', shape=[None, out_dim])
            I_observe = tf.compat.v1.placeholder(tf.float32, name='I_observe', shape=[None, out_dim])
            R_observe = tf.compat.v1.placeholder(tf.float32, name='R_observe', shape=[None, out_dim])
            D_observe = tf.compat.v1.placeholder(tf.float32, name='D_observe', shape=[None, out_dim])
            N_observe = tf.compat.v1.placeholder(tf.float32, name='N_observe', shape=[None, out_dim])
            predict_true_penalty = tf.compat.v1.placeholder_with_default(input=1e3, shape=[], name='pt_p')
            in_learning_rate = tf.compat.v1.placeholder_with_default(input=1e-5, shape=[], name='lr')

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
                in_mu = DNN_base.DNN(T_it, Weight2mu, Bias2mu, hidden_para,
                                     activateIn_name=R['actIn_Name2paras'], activate_name=act_func2paras)
            elif 'DNN_SCALE' == str.upper(R['model2paras']):
                in_beta = DNN_base.DNN_scale(T_it, Weight2beta, Bias2beta, hidden_para, freq2paras,
                                             activateIn_name=R['actIn_Name2paras'], activate_name=act_func2paras)
                in_gamma = DNN_base.DNN_scale(T_it, Weight2gamma, Bias2gamma, hidden_para, freq2paras,
                                              activateIn_name=R['actIn_Name2paras'], activate_name=act_func2paras)
                in_mu = DNN_base.DNN_scale(T_it, Weight2mu, Bias2mu, hidden_para, freq2paras,
                                           activateIn_name=R['actIn_Name2paras'], activate_name=act_func2paras)
            elif str.upper(R['model2paras']) == 'DNN_FOURIERBASE':
                in_beta = DNN_base.DNN_FourierBase(T_it, Weight2beta, Bias2beta, hidden_para, freq2paras,
                                                   activate_name=act_func2paras, sFourier=1.0)
                in_gamma = DNN_base.DNN_FourierBase(T_it, Weight2gamma, Bias2gamma, hidden_para, freq2paras,
                                                    activate_name=act_func2paras, sFourier=1.0)
                in_mu = DNN_base.DNN_FourierBase(T_it, Weight2mu, Bias2mu, hidden_para, freq2paras,
                                                 activate_name=act_func2paras, sFourier=1.0)

            # Remark: beta, gamma,S_NN.I_NN,R_NN都应该是正的. beta.1--15之间，gamma在(0,1）使用归一化的话S_NN.I_NN,R_NN都在[0,1)范围内
            # 在归一化条件下: 如果总的“人口”和归一化"人口"的数值一致，这样的话，归一化后的数值会很小
            if (R['total_population'] == R['normalize_population']) and R['normalize_population'] != 1:
                # beta = tf.square(in_beta)
                # gamma = tf.nn.sigmoid(in_gamma)
                # mu = tf.nn.sigmoid(in_mu)

                # beta = tf.nn.relu(in_beta)
                # gamma = tf.nn.relu(in_gamma)
                # mu = tf.nn.relu(in_mu)

                # beta = tf.square(in_beta)
                # gamma = tf.square(in_gamma)
                # mu = 0.01*tf.square(in_mu)

                # beta = act_gauss(in_beta)
                # gamma = act_gauss(in_gamma)
                # mu = 0.01*act_gauss(in_mu)

                beta = tf.nn.sigmoid(in_beta)
                gamma = tf.nn.sigmoid(in_gamma)
                mu = 0.01*tf.nn.sigmoid(in_mu)

                S_NN = SNN_temp
                I_NN = 0.01*INN_temp
                R_NN = 0.1*RNN_temp
                D_NN = 0.005*DNN_temp

                # 使用ReLU激活函数，这种策略不收敛
                # S_NN = tf.nn.relu(SNN_temp)
                # I_NN = 0.01*tf.nn.relu(INN_temp)
                # R_NN = 0.1*tf.nn.relu(RNN_temp)
                # D_NN = 0.005*tf.nn.relu(DNN_temp)

                # S_NN = act_gauss(SNN_temp)
                # I_NN = 0.05 * act_gauss(INN_temp)
                # R_NN = 0.1 * act_gauss(RNN_temp)
                # D_NN = 0.005 * act_gauss(DNN_temp)

                # S_NN = tf.abs(SNN_temp)
                # I_NN = tf.abs(INN_temp)
                # R_NN = tf.abs(RNN_temp)
                # D_NN = tf.abs(DNN_temp)

                # S_NN = tf.square(SNN_temp)
                # I_NN = 0.01*tf.square(INN_temp)
                # R_NN = 0.1*tf.square(RNN_temp)
                # D_NN = 0.01*tf.square(DNN_temp)

                # S_NN = tf.nn.sigmoid(SNN_temp)
                # I_NN = tf.nn.sigmoid(INN_temp)
                # R_NN = tf.nn.sigmoid(RNN_temp)
                # D_NN = tf.nn.sigmoid(DNN_temp)
            else:
                # beta = tf.nn.relu(in_beta)
                # gamma = tf.nn.relu(in_gamma)
                # mu = tf.nn.relu(in_mu)

                # beta = tf.square(in_beta)
                # gamma = tf.square(in_gamma)
                # mu = 0.01*tf.square(in_mu)

                # beta = act_gauss(in_beta)
                # gamma = act_gauss(in_gamma)
                # mu = 0.01 * act_gauss(in_mu)

                beta = tf.nn.sigmoid(in_beta)
                gamma = tf.nn.sigmoid(in_gamma)
                mu = 0.01*tf.nn.sigmoid(in_mu)

                S_NN = SNN_temp
                I_NN = 0.01*INN_temp
                R_NN = 0.1*RNN_temp
                D_NN = 0.005*DNN_temp

                # 使用ReLU激活函数，这种策略不收敛
                # S_NN = tf.nn.relu(SNN_temp)
                # I_NN = 0.01*tf.nn.relu(INN_temp)
                # R_NN = 0.1*tf.nn.relu(RNN_temp)
                # D_NN = 0.01*tf.nn.relu(DNN_temp)

                # S_NN = act_gauss(SNN_temp)
                # I_NN = 0.05 * act_gauss(INN_temp)
                # R_NN = 0.1 * act_gauss(RNN_temp)
                # D_NN = 0.005 * act_gauss(DNN_temp)

                # S_NN = tf.abs(SNN_temp)
                # I_NN = tf.abs(INN_temp)
                # R_NN = tf.abs(RNN_temp)
                # D_NN = tf.abs(DNN_temp)

                # S_NN = tf.square(SNN_temp)
                # I_NN = 0.1*tf.square(INN_temp)
                # R_NN = 0.1*tf.square(RNN_temp)
                # D_NN = 0.01*tf.square(DNN_temp)

                # S_NN = tf.nn.sigmoid(SNN_temp)
                # I_NN = tf.nn.sigmoid(INN_temp)
                # R_NN = tf.nn.sigmoid(RNN_temp)
                # D_NN = tf.nn.sigmoid(DNN_temp)

            N_NN = S_NN + I_NN + R_NN + D_NN

            dS_NN2t = tf.gradients(S_NN, T_it)[0]
            dI_NN2t = tf.gradients(I_NN, T_it)[0]
            dR_NN2t = tf.gradients(R_NN, T_it)[0]
            dD_NN2t = tf.gradients(D_NN, T_it)[0]
            # dN_NN2t = tf.gradients(N_NN, T_it)[0]

            temp_snn2t = -beta * S_NN * I_NN / (S_NN + I_NN)
            temp_inn2t = beta * S_NN * I_NN - gamma * I_NN - mu * I_NN
            temp_rnn2t = gamma * I_NN
            temp_dnn2t = mu * I_NN

            if str.lower(R['loss_function']) == 'l2_loss' and R['scale_up'] == 0:
                LossS_Net_obs = tf.reduce_mean(tf.square(S_NN - S_observe))
                LossI_Net_obs = tf.reduce_mean(tf.square(I_NN - I_observe))
                LossR_Net_obs = tf.reduce_mean(tf.square(R_NN - R_observe))
                LossD_Net_obs = tf.reduce_mean(tf.square(D_NN - D_observe))
                LossN_Net_obs = tf.reduce_mean(tf.square(N_NN - N_observe))

                Loss2dS = tf.reduce_mean(tf.square(dS_NN2t - temp_snn2t))
                Loss2dI = tf.reduce_mean(tf.square(dI_NN2t - temp_inn2t))
                Loss2dR = tf.reduce_mean(tf.square(dR_NN2t - temp_rnn2t))
                # Loss2dN = tf.reduce_mean(tf.square(dN_NN2t))
                Loss2dD = tf.reduce_mean(tf.square(dD_NN2t - temp_dnn2t))
            elif str.lower(R['loss_function']) == 'l2_loss' and R['scale_up'] == 1:
                scale_up = R['scale_factor']
                LossS_Net_obs = tf.reduce_mean(tf.square(S_NN - S_observe))
                LossI_Net_obs = tf.reduce_mean(tf.square(scale_up*I_NN - scale_up*I_observe))
                LossR_Net_obs = tf.reduce_mean(tf.square(scale_up*R_NN - scale_up*R_observe))
                LossD_Net_obs = tf.reduce_mean(tf.square(scale_up*D_NN - scale_up*D_observe))
                LossN_Net_obs = tf.reduce_mean(tf.square(N_NN - N_observe))

                Loss2dS = tf.reduce_mean(tf.square(dS_NN2t - temp_snn2t))
                Loss2dI = tf.reduce_mean(tf.square(dI_NN2t - temp_inn2t))
                Loss2dR = tf.reduce_mean(tf.square(dR_NN2t - temp_rnn2t))
                # Loss2dN = tf.reduce_mean(tf.square(dN_NN2t))
                Loss2dD = tf.reduce_mean(tf.square(dD_NN2t - temp_dnn2t))
            elif str.lower(R['loss_function']) == 'lncosh_loss' and R['scale_up'] == 0:
                LossS_Net_obs = tf.reduce_mean(tf.log(tf.cosh(S_NN - S_observe)))
                LossI_Net_obs = tf.reduce_mean(tf.log(tf.cosh(I_NN - I_observe)))
                LossR_Net_obs = tf.reduce_mean(tf.log(tf.cosh(R_NN - R_observe)))
                LossD_Net_obs = tf.reduce_mean(tf.log(tf.cosh(D_NN - D_observe)))
                LossN_Net_obs = tf.reduce_mean(tf.log(tf.cosh(N_NN - N_observe)))

                Loss2dS = tf.reduce_mean(tf.log(tf.cosh(dS_NN2t - temp_snn2t)))
                Loss2dI = tf.reduce_mean(tf.log(tf.cosh(dI_NN2t - temp_inn2t)))
                Loss2dR = tf.reduce_mean(tf.log(tf.cosh(dR_NN2t - temp_rnn2t)))
                Loss2dD = tf.reduce_mean(tf.log(tf.cosh(dD_NN2t - temp_dnn2t)))
                # Loss2dN = tf.reduce_mean(tf.log(tf.cosh(dN_NN2t)))
            elif str.lower(R['loss_function']) == 'lncosh_loss' and R['scale_up'] == 1:
                scale_up = R['scale_factor']
                LossS_Net_obs = tf.reduce_mean(tf.log(tf.cosh(scale_up*S_NN - scale_up*S_observe)))
                LossI_Net_obs = tf.reduce_mean(tf.log(tf.cosh(scale_up*I_NN - scale_up*I_observe)))
                LossR_Net_obs = tf.reduce_mean(tf.log(tf.cosh(scale_up*R_NN - scale_up*R_observe)))
                LossD_Net_obs = tf.reduce_mean(tf.log(tf.cosh(scale_up*D_NN - scale_up*D_observe)))
                LossN_Net_obs = tf.reduce_mean(tf.log(tf.cosh(scale_up*N_NN - scale_up*N_observe)))

                Loss2dS = tf.reduce_mean(tf.log(tf.cosh(dS_NN2t - temp_snn2t)))
                Loss2dI = tf.reduce_mean(tf.log(tf.cosh(dI_NN2t - temp_inn2t)))
                Loss2dR = tf.reduce_mean(tf.log(tf.cosh(dR_NN2t - temp_rnn2t)))
                Loss2dD = tf.reduce_mean(tf.log(tf.cosh(dD_NN2t - temp_dnn2t)))
                # Loss2dN = tf.reduce_mean(tf.log(tf.cosh(dN_NN2t)))

            if R['regular_weight_model'] == 'L1':
                regular_WB2S = DNN_base.regular_weights_biases_L1(Weight2S, Bias2S)
                regular_WB2I = DNN_base.regular_weights_biases_L1(Weight2I, Bias2I)
                regular_WB2R = DNN_base.regular_weights_biases_L1(Weight2R, Bias2R)
                regular_WB2D = DNN_base.regular_weights_biases_L1(Weight2D, Bias2D)
                regular_WB2Beta = DNN_base.regular_weights_biases_L1(Weight2beta, Bias2beta)
                regular_WB2Gamma = DNN_base.regular_weights_biases_L1(Weight2gamma, Bias2gamma)
                regular_WB2Mu = DNN_base.regular_weights_biases_L1(Weight2mu, Bias2mu)
            elif R['regular_weight_model'] == 'L2':
                regular_WB2S = DNN_base.regular_weights_biases_L2(Weight2S, Bias2S)
                regular_WB2I = DNN_base.regular_weights_biases_L2(Weight2I, Bias2I)
                regular_WB2R = DNN_base.regular_weights_biases_L2(Weight2R, Bias2R)
                regular_WB2D = DNN_base.regular_weights_biases_L2(Weight2D, Bias2D)
                regular_WB2Beta = DNN_base.regular_weights_biases_L2(Weight2beta, Bias2beta)
                regular_WB2Gamma = DNN_base.regular_weights_biases_L2(Weight2gamma, Bias2gamma)
                regular_WB2Mu = DNN_base.regular_weights_biases_L2(Weight2mu, Bias2mu)
            else:
                regular_WB2S = tf.constant(0.0)
                regular_WB2I = tf.constant(0.0)
                regular_WB2R = tf.constant(0.0)
                regular_WB2D = tf.constant(0.0)
                regular_WB2Beta = tf.constant(0.0)
                regular_WB2Gamma = tf.constant(0.0)
                regular_WB2Mu = tf.constant(0.0)

            PWB2S = wb_penalty2ode*regular_WB2S
            PWB2I = wb_penalty2ode*regular_WB2I
            PWB2R = wb_penalty2ode*regular_WB2R
            PWB2D = wb_penalty2ode * regular_WB2D
            PWB2Beta = wb_penalty2paras * regular_WB2Beta
            PWB2Gamma = wb_penalty2paras * regular_WB2Gamma
            PWB2Mu = wb_penalty2paras * regular_WB2Mu

            Loss2S = predict_true_penalty * LossS_Net_obs + Loss2dS + PWB2S
            Loss2I = predict_true_penalty * LossI_Net_obs + Loss2dI + PWB2I
            Loss2R = predict_true_penalty * LossR_Net_obs + Loss2dR + PWB2R
            Loss2D = predict_true_penalty * LossD_Net_obs + Loss2dD + PWB2D
            Loss2N = predict_true_penalty * LossN_Net_obs

            Loss = Loss2S + Loss2I + Loss2R + Loss2D + Loss2N + PWB2Beta + PWB2Gamma + PWB2Mu

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
    test_mse2S_all, test_rel2S_all = [], []
    test_mse2I_all, test_rel2I_all = [], []

    test_mse2R_all, test_rel2R_all = [], []
    test_mse2D_all, test_rel2D_all = [], []

    # filename = 'data2csv/Wuhan.csv'
    # filename = 'data2csv/Italia_data.csv'
    # filename = 'data2csv/Korea_data.csv'
    # filename = 'data2csv/minnesota.csv'
    # filename = 'data2csv/minnesota2.csv'
    filename = 'data2csv/minnesota3.csv'
    date, data2S, data2I, data2R, data2D = DNN_data.load_4csvData_cal_S(
        datafile=filename, total_population=R['total_population'])

    assert (trainSet_szie + batchSize_test <= len(data2I))
    if R['normalize_population'] == 1:
        # 不归一化数据
        train_date, train_data2s, train_data2i, train_data2r, train_data2d, test_date, test_data2s, test_data2i, \
        test_data2r, test_data2d = DNN_data.split_5csvData2train_test(date, data2S, data2I, data2R, data2D,
                                                                      size2train=trainSet_szie, normalFactor=1.0)
        nbatch2train = np.ones(batchSize_train, dtype=np.float32) * float(R['total_population'])

    elif (R['total_population'] != R['normalize_population']) and R['normalize_population'] != 1:
        # 归一化数据，使用的归一化数值小于总“人口”
        train_date, train_data2s, train_data2i, train_data2r, train_data2d, test_date, test_data2s, test_data2i, \
        test_data2r, test_data2d = DNN_data.split_5csvData2train_test(date, data2S, data2I, data2R, data2D,
                                                                      size2train=trainSet_szie,
                                                                      normalFactor=R['normalize_population'])
        nbatch2train = np.ones(batchSize_train, dtype=np.float32) * (
                    float(R['total_population']) / float(R['normalize_population']))

    elif (R['total_population'] == R['normalize_population']) and R['normalize_population'] != 1:
        # 归一化数据，使用总“人口”归一化数据
        train_date, train_data2s, train_data2i, train_data2r, train_data2d, test_date, test_data2s, test_data2i, \
        test_data2r, test_data2d = DNN_data.split_5csvData2train_test(date, data2S, data2I, data2R, data2D,
                                                                      size2train=trainSet_szie,
                                                                      normalFactor=R['total_population'])
        nbatch2train = np.ones(batchSize_train, dtype=np.float32)

    # 对于时间数据来说，验证模型的合理性，要用连续的时间数据验证.
    test_t_bach = DNN_data.sample_testDays_serially(test_date, batchSize_test)

    # 由于将数据拆分为训练数据和测试数据时，进行了归一化处理，故这里不用归一化
    s_obs_test = DNN_data.sample_testData_serially(test_data2s, batchSize_test, normalFactor=1.0)
    i_obs_test = DNN_data.sample_testData_serially(test_data2i, batchSize_test, normalFactor=1.0)
    r_obs_test = DNN_data.sample_testData_serially(test_data2r, batchSize_test, normalFactor=1.0)
    d_obs_test = DNN_data.sample_testData_serially(test_data2d, batchSize_test, normalFactor=1.0)

    print('The test data about s:\n', str(np.transpose(s_obs_test)))
    print('\n')
    print('The test data about i:\n', str(np.transpose(i_obs_test)))
    print('\n')
    print('The test data about r:\n', str(np.transpose(r_obs_test)))
    print('\n')
    print('The test data about d:\n', str(np.transpose(d_obs_test)))
    print('\n')

    DNN_tools.log_string('The test data about s:\n%s\n' % str(np.transpose(s_obs_test)), log_fileout)
    DNN_tools.log_string('The test data about i:\n%s\n' % str(np.transpose(i_obs_test)), log_fileout)
    DNN_tools.log_string('The test data about r:\n%s\n' % str(np.transpose(r_obs_test)), log_fileout)
    DNN_tools.log_string('The test data about d:\n%s\n' % str(np.transpose(d_obs_test)), log_fileout)

    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True                        # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True                            # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        tmp_lr = init_lr
        for i_epoch in range(R['max_epoch'] + 1):
            t_batch, s_obs, i_obs, r_obs, d_obs = \
                DNN_data.randSample_Normalize_5existData(
                    train_date, train_data2s, train_data2i, train_data2r, train_data2d, batchsize=batchSize_train,
                    normalFactor=1.0, sampling_opt=R['opt2sample'])
            n_obs = np.reshape(nbatch2train, [batchSize_train, 1])
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
            else:
                temp_penalty_pt = pt_penalty_init

            _, loss_s, loss_i, loss_r, loss_d, loss_n, loss, pwb2s, pwb2i, pwb2r, pwb2d, pwb2beta, pwb2gamma, pwb2mu = \
                sess.run([train_Losses, Loss2S, Loss2I, Loss2R, Loss2D, Loss2N, Loss, PWB2S, PWB2I, PWB2R, PWB2D,
                          PWB2Beta, PWB2Gamma, PWB2Mu], feed_dict={T_it: t_batch, S_observe: s_obs, I_observe: i_obs,
                                                                   R_observe: r_obs, D_observe: d_obs, N_observe: n_obs,
                                                                   in_learning_rate: tmp_lr,
                                                                   predict_true_penalty: temp_penalty_pt})

            loss_s_all.append(loss_s)
            loss_i_all.append(loss_i)
            loss_r_all.append(loss_r)
            loss_d_all.append(loss_d)
            loss_n_all.append(loss_n)
            loss_all.append(loss)

            if i_epoch % 1000 == 0:
                print_and_log2train(i_epoch, time.time() - t0, tmp_lr, temp_penalty_pt, pwb2s, pwb2i, pwb2r, pwb2d,
                                    pwb2beta, pwb2gamma, pwb2mu, loss_s, loss_i, loss_r, loss_d, loss_n, loss,
                                    log_out=log_fileout)

                # 以下代码为输出训练过程中 S_NN, I_NN, R_NN, D_NN, beta, gamma, mu 的训练结果
                s_nn2train, i_nn2train, r_nn2train, d_nn2train, beta2train, gamma2train, mu2train = sess.run(
                    [S_NN, I_NN, R_NN, D_NN, beta, gamma, mu], feed_dict={T_it: np.reshape(train_date, [-1, 1])})

                # 以下代码为输出训练过程中 S_NN, I_NN, R_NN, beta, gamma 的测试结果
                test_epoch.append(i_epoch / 1000)
                s_nn2test, i_nn2test, r_nn2test, d_nn2test, beta2test, gamma2test, mu2test = sess.run(
                    [S_NN, I_NN, R_NN, D_NN, beta, gamma, mu], feed_dict={T_it: test_t_bach})

                test_mse2S = np.mean(np.square(s_nn2test - s_obs_test))
                test_mse2S_all.append(test_mse2S)
                test_rel2S = test_mse2S / np.mean(np.square(s_obs_test))
                test_rel2S_all.append(test_rel2S)

                test_mse2I = np.mean(np.square(i_nn2test - i_obs_test))
                test_mse2I_all.append(test_mse2I)
                test_rel2I = test_mse2I / np.mean(np.square(i_obs_test))
                test_rel2I_all.append(test_rel2I)

                test_mse2R = np.mean(np.square(r_nn2test - r_obs_test))
                test_mse2R_all.append(test_mse2R)
                test_rel2R = test_mse2I / np.mean(np.square(r_obs_test))
                test_rel2R_all.append(test_rel2R)

                test_mse2D = np.mean(np.square(d_nn2test - d_obs_test))
                test_mse2D_all.append(test_mse2D)
                test_rel2D = test_mse2D / np.mean(np.square(d_obs_test))
                test_rel2D_all.append(test_rel2D)

                print_and_log_test_one_epoch(test_mse2S, test_rel2S, test_mse2I, test_rel2I, test_mse2R, test_rel2R,
                                             test_mse2D, test_rel2D, log_out=log_fileout)

        saveData.save_trainSolu2mat_Covid(s_nn2train, name2solus='s2train', outPath=R['FolderName'])
        saveData.save_trainSolu2mat_Covid(i_nn2train, name2solus='i2train', outPath=R['FolderName'])
        saveData.save_trainSolu2mat_Covid(r_nn2train, name2solus='r2train', outPath=R['FolderName'])
        saveData.save_trainSolu2mat_Covid(d_nn2train, name2solus='d2train', outPath=R['FolderName'])

        saveData.save_trainParas2mat_Covid(beta2train, name2para='beta2train', outPath=R['FolderName'])
        saveData.save_trainParas2mat_Covid(gamma2train, name2para='gamma2train', outPath=R['FolderName'])
        saveData.save_trainParas2mat_Covid(mu2train, name2para='mu2train', outPath=R['FolderName'])

        plotData.plot_Solus2convid(np.reshape(train_data2s, [-1, 1]), s_nn2train, name2file='s2train',
                                   name2solu1='s_true', name2solu2='s_train',
                                   coord_points2test=np.reshape(train_date, [-1, 1]), outPath=R['FolderName'])
        plotData.plot_Solus2convid(np.reshape(train_data2i, [-1, 1]), i_nn2train, name2file='i2train',
                                   name2solu1='i_true', name2solu2='i_train',
                                   coord_points2test=np.reshape(train_date, [-1, 1]), outPath=R['FolderName'])
        plotData.plot_Solus2convid(np.reshape(train_data2r, [-1, 1]), r_nn2train, name2file='r2train',
                                   name2solu1='r_true', name2solu2='r_train',
                                   coord_points2test=np.reshape(train_date, [-1, 1]), outPath=R['FolderName'])
        plotData.plot_Solus2convid(np.reshape(train_data2d, [-1, 1]), d_nn2train, name2file='d2train',
                                   name2solu1='d_true', name2solu2='d_train',
                                   coord_points2test=np.reshape(train_date, [-1, 1]), outPath=R['FolderName'])

        plotData.plot_Para2convid(beta2train, name2para='beta_train', coord_points2test=np.reshape(train_date, [-1, 1]),
                                  outPath=R['FolderName'])
        plotData.plot_Para2convid(gamma2train, name2para='gamma_train', coord_points2test=np.reshape(train_date, [-1, 1]),
                                  outPath=R['FolderName'])
        plotData.plot_Para2convid(mu2train, name2para='mu_train', coord_points2test=np.reshape(train_date, [-1, 1]),
                                  outPath=R['FolderName'])

        saveData.save_SIRD_trainLoss2mat_no_N(loss_s_all, loss_i_all, loss_r_all, loss_d_all, actName=act_func2SIRD,
                                              outPath=R['FolderName'])

        plotData.plotTrain_loss_1act_func(loss_s_all, lossType='loss2s', seedNo=R['seed'], outPath=R['FolderName'],
                                          yaxis_scale=True)
        plotData.plotTrain_loss_1act_func(loss_i_all, lossType='loss2i', seedNo=R['seed'], outPath=R['FolderName'],
                                          yaxis_scale=True)
        plotData.plotTrain_loss_1act_func(loss_r_all, lossType='loss2r', seedNo=R['seed'], outPath=R['FolderName'],
                                          yaxis_scale=True)
        plotData.plotTrain_loss_1act_func(loss_d_all, lossType='loss2d', seedNo=R['seed'], outPath=R['FolderName'],
                                          yaxis_scale=True)

        saveData.true_value2convid(s_obs_test, name2Array='s2test', outPath=R['FolderName'])
        saveData.true_value2convid(i_obs_test, name2Array='i2test', outPath=R['FolderName'])
        saveData.true_value2convid(r_obs_test, name2Array='r2test', outPath=R['FolderName'])
        saveData.true_value2convid(d_obs_test, name2Array='d2test', outPath=R['FolderName'])

        saveData.save_testMSE_REL2mat(test_mse2S_all, test_rel2S_all, actName='Susceptible', outPath=R['FolderName'])
        saveData.save_testMSE_REL2mat(test_mse2I_all, test_rel2I_all, actName='Infected', outPath=R['FolderName'])
        saveData.save_testMSE_REL2mat(test_mse2R_all, test_rel2R_all, actName='Recover', outPath=R['FolderName'])
        saveData.save_testMSE_REL2mat(test_mse2D_all, test_rel2D_all, actName='Death', outPath=R['FolderName'])

        plotData.plotTest_MSE_REL(test_mse2S_all, test_rel2S_all, test_epoch, actName='Susceptible', seedNo=R['seed'],
                                  outPath=R['FolderName'], yaxis_scale=True)
        plotData.plotTest_MSE_REL(test_mse2I_all, test_rel2I_all, test_epoch, actName='Infected', seedNo=R['seed'],
                                  outPath=R['FolderName'], yaxis_scale=True)
        plotData.plotTest_MSE_REL(test_mse2R_all, test_rel2R_all, test_epoch, actName='Recover', seedNo=R['seed'],
                                  outPath=R['FolderName'], yaxis_scale=True)
        plotData.plotTest_MSE_REL(test_mse2D_all, test_rel2D_all, test_epoch, actName='Death', seedNo=R['seed'],
                                  outPath=R['FolderName'], yaxis_scale=True)

        saveData.save_SIRD_testSolus2mat(s_nn2test, i_nn2test, r_nn2test, d_nn2test, name2solus1='snn2test',
                                         name2solus2='inn2test', name2solus3='rnn2test', name2solus4='dnn2test',
                                         outPath=R['FolderName'])
        saveData.save_SIRD_testParas2mat(beta2test, gamma2test, mu2test, name2para1='beta2test',
                                         name2para2='gamma2test', name2para3='mu2test', outPath=R['FolderName'])

        plotData.plot_Solus2convid(s_obs_test, s_nn2test, name2file='s2test', name2solu1='s_true', name2solu2='s_test',
                                   coord_points2test=test_t_bach, seedNo=R['seed'], outPath=R['FolderName'])
        plotData.plot_Solus2convid(i_obs_test, i_nn2test, name2file='i2test', name2solu1='i_true', name2solu2='i_test',
                                   coord_points2test=test_t_bach, seedNo=R['seed'], outPath=R['FolderName'])
        plotData.plot_Solus2convid(r_obs_test, r_nn2test, name2file='r2test', name2solu1='r_true', name2solu2='r_test',
                                   coord_points2test=test_t_bach, seedNo=R['seed'], outPath=R['FolderName'])
        plotData.plot_Solus2convid(d_obs_test, d_nn2test, name2file='d2test', name2solu1='d_true', name2solu2='d_test',
                                   coord_points2test=test_t_bach, seedNo=R['seed'], outPath=R['FolderName'])

        plotData.plot_Para2convid(beta2test, name2para='beta_test', coord_points2test=test_t_bach,
                                  outPath=R['FolderName'])
        plotData.plot_Para2convid(gamma2test, name2para='gamma_test', coord_points2test=test_t_bach,
                                  outPath=R['FolderName'])
        plotData.plot_Para2convid(mu2test, name2para='mu_test', coord_points2test=test_t_bach,
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
    # step_stop_flag = input('please input an  integer number to activate step-stop----0:no---!0:yes--:')
    # R['activate_stop'] = int(step_stop_flag)
    R['activate_stop'] = int(0)
    # if the value of step_stop_flag is not 0, it will activate stop condition of step to kill program
    R['max_epoch'] = 60000
    if 0 != R['activate_stop']:
        epoch_stop = input('please input a stop epoch:')
        R['max_epoch'] = int(epoch_stop)

    # ----------------------------------------- Convid 设置 ---------------------------------
    R['eqs_name'] = 'SIRD'
    R['input_dim'] = 1                       # 输入维数，即问题的维数(几元问题)
    R['output_dim'] = 1                      # 输出维数
    R['total_population'] = 3450000          # 总的“人口”数量

    R['normalize_population'] = 3450000      # 归一化时使用的“人口”数值
    # R['normalize_population'] = 100000
    # R['normalize_population'] = 1

    # ------------------------------------  神经网络的设置  ----------------------------------------
    R['size2train'] = 250                    # 训练集的大小
    R['batch_size2train'] = 30              # 训练数据的批大小
    R['batch_size2test'] = 50               # 测试数据的批大小
    # R['opt2sample'] = 'random_sample'     # 训练集的选取方式--随机采样
    # R['opt2sample'] = 'rand_sample_sort'    # 训练集的选取方式--随机采样后按时间排序
    R['opt2sample'] = 'windows_rand_sample'  # 训练集的选取方式--随机窗口采样(以随机点为基准，然后滑动窗口采样)

    R['init_penalty2predict_true'] = 100     # Regularization parameter for boundary conditions
    # R['activate_stage_penalty'] = 0         # 是否开启阶段调整惩罚项，0 代表不调整，非 0 代表调整
    R['activate_stage_penalty'] = 1         # 是否开启阶段调整惩罚项，0 代表不调整，非 0 代表调整
    if R['activate_stage_penalty'] == 1 or R['activate_stage_penalty'] == 2:
        # R['init_penalty2predict_true'] = 1000
        # R['init_penalty2predict_true'] = 100
        # R['init_penalty2predict_true'] = 50
        # R['init_penalty2predict_true'] = 20
        R['init_penalty2predict_true'] = 10
        # R['init_penalty2predict_true'] = 1

    # R['regular_weight_model'] = 'L0'
    # R['regular_weight_model'] = 'L1'
    R['regular_weight_model'] = 'L2'          # The model of regular weights and biases

    # R['regular_weight2ODE'] = 0.001             # Regularization parameter for weights
    # R['regular_weight2ODE'] = 0.0005            # Regularization parameter for weights
    R['regular_weight2ODE'] = 0.0001            # Regularization parameter for weights
    # R['regular_weight2ODE'] = 0.00005             # Regularization parameter for weights
    # R['regular_weight2ODE'] = 0.00001           # Regularization parameter for weights

    R['regular_weight2Paras'] = 0.0001

    R['optimizer_name'] = 'Adam'              # 优化器
    R['loss_function'] = 'L2_loss'            # 损失函数的类型
    # R['loss_function'] = 'lncosh_loss'      # 损失函数的类型

    # R['scale_up'] = 0                         # scale_up 用来控制湿粉扑对数值进行尺度提升，如1e-6量级提升到1e-2量级。不为 0 代表开启提升
    R['scale_up'] = 1                         # scale_up 用来控制湿粉扑对数值进行尺度提升，如1e-6量级提升到1e-2量级。不为 0 代表开启提升
    R['scale_factor'] = 100                   # scale_factor 用来对数值进行尺度提升，如1e-6量级提升到1e-2量级

    R['train_model'] = 'train_group'        # 训练模式:各个不同的loss捆绑打包训练
    # R['train_model'] = 'train_union_loss'     # 训练模式:各个不同的loss累加在一起，训练

    if 50000 < R['max_epoch']:
        R['learning_rate'] = 2e-3             # 学习率
        R['lr_decay'] = 1e-4                  # 学习率 decay
        # R['learning_rate'] = 2e-4           # 学习率
        # R['lr_decay'] = 5e-5                # 学习率 decay
    elif (20000 < R['max_epoch'] and 50000 >= R['max_epoch']):
        R['learning_rate'] = 2e-3           # 学习率
        R['lr_decay'] = 5e-4                # 学习率 decay
        # R['learning_rate'] = 2e-4           # 学习率
        # R['lr_decay'] = 1e-4                # 学习率 decay
        # R['learning_rate'] = 1e-4             # 学习率
        # R['lr_decay'] = 5e-5                  # 学习率 decay
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
        R['freq2SIRD'] = np.concatenate(([1], np.arange(1, 25)), axis=0)
    if R['model2paras'] != 'DNN':
        R['freq2paras'] = np.concatenate(([1], np.arange(1, 25)), axis=0)

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

