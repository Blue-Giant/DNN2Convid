"""
@author: xi'an Li
Benchmark Code of SIRD model using Euler iteration and DNN
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

# Reference paper: A flexible rolling regression framework for the identification of time-varying SIRD models


def act_gauss(input):
    # out = tf.exp(-0.25*tf.multiply(input, input))
    out = tf.exp(-0.5 * tf.multiply(input, input))
    return out


# 记录字典中的一些设置
def dictionary_out2file(R_dic, log_fileout):
    DNN_tools.log_string('Equation name for problem: %s\n' % (R_dic['eqs_name']), log_fileout)
    DNN_tools.log_string('Network model of dealing with parameters: %s\n' % str(R_dic['model2paras']), log_fileout)

    if str.upper(R_dic['model2paras']) == 'DNN_FOURIERBASE':
        DNN_tools.log_string('The input activate function for parameter: %s\n' % '[sin;cos]', log_fileout)
    else:
        DNN_tools.log_string('The input activate function for parameter: %s\n' % str(R_dic['actIn_Name2paras']), log_fileout)

    DNN_tools.log_string('The hidden-layer activate function for parameter: %s\n' % str(R_dic['act_Name2paras']), log_fileout)

    DNN_tools.log_string('hidden layers for parameters: %s\n' % str(R_dic['hidden2para']), log_fileout)

    if str.upper(R_dic['model2paras']) != 'DNN':
        DNN_tools.log_string('The scale for frequency to SIR NN: %s\n' % str(R_dic['freq2paras']), log_fileout)
        DNN_tools.log_string('Repeat the high-frequency scale or not for para-NN: %s\n' % str(R_dic['if_repeat_High_freq2paras']), log_fileout)

    DNN_tools.log_string('The training model for all networks: %s\n' % str(R_dic['train_model']), log_fileout)

    if (R_dic['optimizer_name']).title() == 'Adam':
        DNN_tools.log_string('optimizer:%s\n' % str(R_dic['optimizer_name']), log_fileout)
    else:
        DNN_tools.log_string('optimizer:%s  with momentum=%f\n' % (R_dic['optimizer_name'], R_dic['momentum']),
                             log_fileout)
    DNN_tools.log_string('Init learning rate: %s\n' % str(R_dic['learning_rate']), log_fileout)
    DNN_tools.log_string('Decay to learning rate: %s\n' % str(R_dic['lr_decay']), log_fileout)
    DNN_tools.log_string('The type for Loss function: %s\n' % str(R_dic['loss_function']), log_fileout)

    if R_dic['activate_stop'] != 0:
        DNN_tools.log_string('activate the stop_step and given_step= %s\n' % str(R_dic['max_epoch']), log_fileout)
    else:
        DNN_tools.log_string('no activate the stop_step and given_step = default: %s\n' % str(R_dic['max_epoch']), log_fileout)

    DNN_tools.log_string('The model of regular weights and biases: %s\n' % str(R_dic['regular_weight_model']), log_fileout)

    DNN_tools.log_string('Regularization parameter for weights and biases: %s\n' % str(R_dic['regular_weight']), log_fileout)

    DNN_tools.log_string('Size 2 training set: %s\n' % str(R_dic['size2train']), log_fileout)

    DNN_tools.log_string('Batch-size 2 training: %s\n' % str(R_dic['batch_size2train']), log_fileout)

    DNN_tools.log_string('Batch-size 2 testing: %s\n' % str(R_dic['batch_size2test']), log_fileout)


def print_and_log2train(i_epoch, run_time, tmp_lr, penalty_wb2beta, penalty_wb2gamma, penalty_wb2mu, loss_s, loss_i,
                        loss_r, loss_d, loss_all, log_out=None):
    print('train epoch: %d, time: %.3f' % (i_epoch, run_time))
    print('learning rate: %.10f' % tmp_lr)
    print('penalty weights and biases for Beta: %.16f' % penalty_wb2beta)
    print('penalty weights and biases for Gamma: %.16f' % penalty_wb2gamma)
    print('penalty weights and biases for Mu: %.16f' % penalty_wb2mu)
    print('loss for S: %.16f' % loss_s)
    print('loss for I: %.16f' % loss_i)
    print('loss for R: %.16f' % loss_r)
    print('loss for D: %.16f' % loss_d)
    print('total loss: %.16f\n' % loss_all)

    DNN_tools.log_string('train epoch: %d,time: %.3f' % (i_epoch, run_time), log_out)
    DNN_tools.log_string('learning rate: %.10f' % tmp_lr, log_out)
    DNN_tools.log_string('penalty weights and biases for Beta: %.16f' % penalty_wb2beta, log_out)
    DNN_tools.log_string('penalty weights and biases for Gamma: %.16f' % penalty_wb2gamma, log_out)
    DNN_tools.log_string('penalty weights and biases for Mu: %.16f' % penalty_wb2mu, log_out)
    DNN_tools.log_string('loss for S: %.16f' % loss_s, log_out)
    DNN_tools.log_string('loss for I: %.16f' % loss_i, log_out)
    DNN_tools.log_string('loss for R: %.16f' % loss_r, log_out)
    DNN_tools.log_string('loss for D: %.16f' % loss_d, log_out)
    DNN_tools.log_string('total loss: %.16f \n\n' % loss_all, log_out)


# 使用拟欧拉方法，归一化数据是不可行的。
def solve_SIRD2COVID(R):
    log_out_path = R['FolderName']        # 将路径从字典 R 中提取出来
    if not os.path.exists(log_out_path):  # 判断路径是否已经存在
        os.mkdir(log_out_path)            # 无 log_out_path 路径，创建一个 log_out_path 路径
    log_fileout = open(os.path.join(log_out_path, 'log_train.txt'), 'w')  # 在这个路径下创建并打开一个可写的 log_train.txt文件
    dictionary_out2file(R, log_fileout)

    trainSet_szie = R['size2train']                   # 训练集大小,给定一个数据集，拆分训练集和测试集时，需要多大规模的训练集
    batchSize_train = R['batch_size2train']           # 训练批量的大小,该值远小于训练集大小
    batchSize_test = R['batch_size2test']             # 测试批量的大小,该值小于等于测试集大小
    wb_penalty = R['regular_weight']                  # 神经网络参数的惩罚因子
    lr_decay = R['lr_decay']                          # 学习率额衰减
    init_lr = R['learning_rate']                      # 初始学习率

    act_func2paras = R['act_Name2paras']              # 参数网络的隐藏层激活函数

    input_dim = R['input_dim']                        # 输入维度
    out_dim = R['output_dim']                         # 输出维度

    flag2beta = 'WB2beta'
    flag2gamma = 'WB2gamma'
    flag2mu = 'WB2mu'
    hidden_para = R['hidden2para']

    AI = tf.eye(batchSize_train, dtype=tf.float32) * (-2)
    Ones_mat = tf.ones([batchSize_train, batchSize_train], dtype=tf.float32)
    A_diag = tf.linalg.band_part(Ones_mat, 0, 1)
    Amat = AI + A_diag

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
            T_train = tf.compat.v1.placeholder(tf.float32, name='T_train', shape=[batchSize_train, out_dim])
            S_observe = tf.compat.v1.placeholder(tf.float32, name='S_observe', shape=[batchSize_train, out_dim])
            I_observe = tf.compat.v1.placeholder(tf.float32, name='I_observe', shape=[batchSize_train, out_dim])
            R_observe = tf.compat.v1.placeholder(tf.float32, name='R_observe', shape=[batchSize_train, out_dim])
            D_observe = tf.compat.v1.placeholder(tf.float32, name='D_observe', shape=[batchSize_train, out_dim])

            T_train2test = tf.compat.v1.placeholder(tf.float32, name='T_train2test', shape=[trainSet_szie, out_dim])
            T_test = tf.compat.v1.placeholder(tf.float32, name='T_test', shape=[batchSize_test, out_dim])

            in_learning_rate = tf.compat.v1.placeholder_with_default(input=1e-5, shape=[], name='lr')

            if 'DNN' == str.upper(R['model2paras']):
                in_beta2train = DNN_base.DNN(T_train, Weight2beta, Bias2beta, hidden_para,
                                             activateIn_name=R['actIn_Name2paras'], activate_name=R['act_Name2paras'])
                in_gamma2train = DNN_base.DNN(T_train, Weight2gamma, Bias2gamma, hidden_para,
                                              activateIn_name=R['actIn_Name2paras'], activate_name=R['act_Name2paras'])
                in_mu2train = DNN_base.DNN(T_train, Weight2mu, Bias2mu, hidden_para,
                                           activateIn_name=R['actIn_Name2paras'], activate_name=R['act_Name2paras'])
                in_beta2train_test = DNN_base.DNN(T_train2test, Weight2beta, Bias2beta, hidden_para,
                                                  activateIn_name=R['actIn_Name2paras'],
                                                  activate_name=R['act_Name2paras'])
                in_gamma2train_test = DNN_base.DNN(T_train2test, Weight2gamma, Bias2gamma, hidden_para,
                                                   activateIn_name=R['actIn_Name2paras'],
                                                   activate_name=R['act_Name2paras'])
                in_mu2train_test = DNN_base.DNN(T_train2test, Weight2mu, Bias2mu, hidden_para,
                                                activateIn_name=R['actIn_Name2paras'],
                                                activate_name=R['act_Name2paras'])

                in_beta2test = DNN_base.DNN(T_test, Weight2beta, Bias2beta, hidden_para,
                                            activateIn_name=R['actIn_Name2paras'], activate_name=R['act_Name2paras'])
                in_gamma2test = DNN_base.DNN(T_test, Weight2gamma, Bias2gamma, hidden_para,
                                             activateIn_name=R['actIn_Name2paras'], activate_name=R['act_Name2paras'])
                in_mu2test = DNN_base.DNN(T_test, Weight2mu, Bias2mu, hidden_para,
                                          activateIn_name=R['actIn_Name2paras'], activate_name=R['act_Name2paras'])
            elif 'DNN_SCALE' == str.upper(R['model2paras']):
                freq2paras = R['freq2paras']
                in_beta2train = DNN_base.DNN_scale(T_train, Weight2beta, Bias2beta, hidden_para, freq2paras,
                                                   activateIn_name=R['actIn_Name2paras'],
                                                   activate_name=R['act_Name2paras'])
                in_gamma2train = DNN_base.DNN_scale(T_train, Weight2gamma, Bias2gamma, hidden_para, freq2paras,
                                                    activateIn_name=R['actIn_Name2paras'],
                                                    activate_name=R['act_Name2paras'])
                in_mu2train = DNN_base.DNN_scale(T_train, Weight2mu, Bias2mu, hidden_para, freq2paras,
                                                 activateIn_name=R['actIn_Name2paras'],
                                                 activate_name=R['act_Name2paras'])
                in_beta2train_test = DNN_base.DNN_scale(T_train2test, Weight2beta, Bias2beta, hidden_para,
                                                        activateIn_name=R['actIn_Name2paras'],
                                                        activate_name=R['act_Name2paras'])
                in_gamma2train_test = DNN_base.DNN_scale(T_train2test, Weight2gamma, Bias2gamma, hidden_para,
                                                         activateIn_name=R['actIn_Name2paras'],
                                                         activate_name=R['act_Name2paras'])
                in_mu2train_test = DNN_base.DNN_scale(T_train2test, Weight2mu, Bias2mu, hidden_para,
                                                      activateIn_name=R['actIn_Name2paras'],
                                                      activate_name=R['act_Name2paras'])

                in_beta2test = DNN_base.DNN_scale(T_test, Weight2beta, Bias2beta, hidden_para, freq2paras,
                                                  activateIn_name=R['actIn_Name2paras'],
                                                  activate_name=R['act_Name2paras'])
                in_gamma2test = DNN_base.DNN_scale(T_test, Weight2gamma, Bias2gamma, hidden_para, freq2paras,
                                                   activateIn_name=R['actIn_Name2paras'],
                                                   activate_name=R['act_Name2paras'])
                in_mu2test = DNN_base.DNN_scale(T_test, Weight2mu, Bias2mu, hidden_para, freq2paras,
                                                activateIn_name=R['actIn_Name2paras'],
                                                activate_name=R['act_Name2paras'])
            elif str.upper(R['model2paras']) == 'DNN_FOURIERBASE':
                freq2paras = R['freq2paras']
                in_beta2train = DNN_base.DNN_FourierBase(T_train, Weight2beta, Bias2beta, hidden_para, freq2paras,
                                                         activate_name=R['act_Name2paras'], sFourier=1.0)
                in_gamma2train = DNN_base.DNN_FourierBase(T_train, Weight2gamma, Bias2gamma, hidden_para, freq2paras,
                                                          activate_name=R['act_Name2paras'], sFourier=1.0)
                in_mu2train = DNN_base.DNN_FourierBase(T_train, Weight2mu, Bias2mu, hidden_para, freq2paras,
                                                       activate_name=R['act_Name2paras'], sFourier=1.0)

                in_beta2train_test = DNN_base.DNN_FourierBase(T_train2test, Weight2beta, Bias2beta, hidden_para,
                                                              freq2paras, activate_name=R['act_Name2paras'],
                                                              sFourier=1.0)
                in_gamma2train_test = DNN_base.DNN_FourierBase(T_train2test, Weight2gamma, Bias2gamma, hidden_para,
                                                               freq2paras, activate_name=R['act_Name2paras'],
                                                               sFourier=1.0)
                in_mu2train_test = DNN_base.DNN_FourierBase(T_train2test, Weight2mu, Bias2mu, hidden_para,
                                                            freq2paras, activate_name=R['act_Name2paras'],
                                                            sFourier=1.0)

                in_beta2test = DNN_base.DNN_FourierBase(T_test, Weight2beta, Bias2beta, hidden_para, freq2paras,
                                                        activate_name=R['act_Name2paras'], sFourier=1.0)
                in_gamma2test = DNN_base.DNN_FourierBase(T_test, Weight2gamma, Bias2gamma, hidden_para, freq2paras,
                                                         activate_name=R['act_Name2paras'], sFourier=1.0)
                in_mu2test = DNN_base.DNN_FourierBase(T_test, Weight2mu, Bias2mu, hidden_para, freq2paras,
                                                      activate_name=R['act_Name2paras'], sFourier=1.0)

            # Remark: beta, gamma,S_NN.I_NN,R_NN都应该是正的. beta.1--15之间，gamma在(0,1）使用归一化的话S_NN.I_NN,R_NN都在[0,1)范围内
            betaNN2train = tf.nn.sigmoid(in_beta2train)
            gammaNN2train = tf.nn.sigmoid(in_gamma2train)
            muNN2train = 0.01*tf.nn.sigmoid(in_mu2train)
            # muNN2train = 0.05 * tf.nn.sigmoid(in_mu2train)
            # muNN2train = 0.1 * tf.nn.sigmoid(in_mu2train)
            #
            betaNN2train_test = tf.nn.sigmoid(in_beta2train_test)
            gammaNN2train_test = tf.nn.sigmoid(in_gamma2train_test)
            muNN2train_test = 0.01 * tf.nn.sigmoid(in_mu2train_test)
            # muNN2train_test = 0.05 * tf.nn.sigmoid(in_mu2train_test)
            # muNN2train_test = 0.1 * tf.nn.sigmoid(in_mu2train_test)
            #
            betaNN2test = tf.nn.sigmoid(in_beta2test)
            gammaNN2test = tf.nn.sigmoid(in_gamma2test)
            muNN2test = 0.01*tf.nn.sigmoid(in_mu2test)
            # muNN2test = 0.05 * tf.nn.sigmoid(in_mu2test)
            # muNN2test = 0.1 * tf.nn.sigmoid(in_mu2test)

            # betaNN2train = act_gauss(in_beta2train)
            # gammaNN2train = act_gauss(in_gamma2train)
            # muNN2train = 0.01*act_gauss(in_mu2train)
            # # muNN2train = 0.05 * act_gauss(in_mu2train)
            #
            # betaNN2train_test = act_gauss(in_beta2train_test)
            # gammaNN2train_test = act_gauss(in_gamma2train_test)
            # muNN2train_test = 0.01 * act_gauss(in_mu2train_test)
            # # muNN2train_test = 0.05 * act_gauss(in_mu2train_test)
            #
            # betaNN2test = act_gauss(in_beta2test)
            # gammaNN2test = act_gauss(in_gamma2test)
            # # muNN2test = 0.01*act_gauss(in_mu2test)
            # # muNN2test = 0.05 * act_gauss(in_mu2test)
            # muNN2test = 0.01 * act_gauss(in_mu2test)

            dS2dt = tf.matmul(Amat[0:-1, :], S_observe)
            dI2dt = tf.matmul(Amat[0:-1, :], I_observe)
            dR2dt = tf.matmul(Amat[0:-1, :], R_observe)
            dD2dt = tf.matmul(Amat[0:-1, :], D_observe)

            temp_s2t = -betaNN2train[0:-1, 0] * S_observe[0:-1, 0] * I_observe[0:-1, 0]/(S_observe[0:-1, 0] + I_observe[0:-1, 0])
            temp_i2t = betaNN2train[0:-1, 0] * S_observe[0:-1, 0] * I_observe[0:-1, 0]/(S_observe[0:-1, 0] + I_observe[0:-1, 0]) - \
                       gammaNN2train[0:-1, 0] * I_observe[0:-1, 0] - muNN2train[0:-1, 0] * I_observe[0:-1, 0]
            temp_r2t = gammaNN2train[0:-1, 0] * I_observe[0:-1, 0]
            temp_d2t = muNN2train[0:-1, 0] * I_observe[0:-1, 0]

            if str.lower(R['loss_function']) == 'l2_loss':
                Loss2dS = tf.reduce_mean(tf.square(dS2dt - tf.reshape(temp_s2t, shape=[-1, 1])))
                Loss2dI = tf.reduce_mean(tf.square(dI2dt - tf.reshape(temp_i2t, shape=[-1, 1])))
                Loss2dR = tf.reduce_mean(tf.square(dR2dt - tf.reshape(temp_r2t, shape=[-1, 1])))
                Loss2dD = tf.reduce_mean(tf.square(dD2dt - tf.reshape(temp_d2t, shape=[-1, 1])))
            elif str.lower(R['loss_function']) == 'lncosh_loss':
                Loss2dS = tf.reduce_mean(tf.log(tf.cosh(dS2dt - tf.reshape(temp_s2t, shape=[-1, 1]))))
                Loss2dI = tf.reduce_mean(tf.log(tf.cosh(dI2dt - tf.reshape(temp_i2t, shape=[-1, 1]))))
                Loss2dR = tf.reduce_mean(tf.log(tf.cosh(dR2dt - tf.reshape(temp_r2t, shape=[-1, 1]))))
                Loss2dD = tf.reduce_mean(tf.log(tf.cosh(dD2dt - tf.reshape(temp_d2t, shape=[-1, 1]))))

            if R['regular_weight_model'] == 'L1':
                regular_WB2Beta = DNN_base.regular_weights_biases_L1(Weight2beta, Bias2beta)
                regular_WB2Gamma = DNN_base.regular_weights_biases_L1(Weight2gamma, Bias2gamma)
                regular_WB2Mu = DNN_base.regular_weights_biases_L1(Weight2mu, Bias2mu)
            elif R['regular_weight_model'] == 'L2':
                regular_WB2Beta = DNN_base.regular_weights_biases_L2(Weight2beta, Bias2beta)
                regular_WB2Gamma = DNN_base.regular_weights_biases_L2(Weight2gamma, Bias2gamma)
                regular_WB2Mu = DNN_base.regular_weights_biases_L2(Weight2mu, Bias2mu)
            else:
                regular_WB2Beta = tf.constant(0.0)
                regular_WB2Gamma = tf.constant(0.0)
                regular_WB2Mu = tf.constant(0.0)

            PWB2Beta = wb_penalty * regular_WB2Beta
            PWB2Gamma = wb_penalty * regular_WB2Gamma
            PWB2Mu = wb_penalty * regular_WB2Mu

            Loss = Loss2dS + Loss2dI + Loss2dR + Loss2dD + PWB2Beta + PWB2Gamma + PWB2Mu

            my_optimizer = tf.compat.v1.train.AdamOptimizer(in_learning_rate)
            train_Losses = my_optimizer.minimize(Loss, global_step=global_steps)

    t0 = time.time()
    loss_s_all, loss_i_all, loss_r_all, loss_d_all, loss_all = [], [], [], [], []
    test_epoch = []

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
    elif (R['total_population'] != R['normalize_population']) and R['normalize_population'] != 1:
        # 归一化数据，使用的归一化数值小于总“人口”
        train_date, train_data2s, train_data2i, train_data2r, train_data2d, test_date, test_data2s, test_data2i, \
        test_data2r, test_data2d = DNN_data.split_5csvData2train_test(date, data2S, data2I, data2R, data2D,
                                                                      size2train=trainSet_szie,
                                                                      normalFactor=R['normalize_population'])
    elif (R['total_population'] == R['normalize_population']) and R['normalize_population'] != 1:
        # 归一化数据，使用总“人口”归一化数据
        train_date, train_data2s, train_data2i, train_data2r, train_data2d, test_date, test_data2s, test_data2i, \
        test_data2r, test_data2d = DNN_data.split_5csvData2train_test(date, data2S, data2I, data2R, data2D,
                                                                      size2train=trainSet_szie,
                                                                      normalFactor=R['total_population'])
    # 对于时间数据来说，验证模型的合理性，要用连续的时间数据验证.
    test_t_bach = DNN_data.sample_testDays_serially(test_date, batchSize_test)

    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.compat.v1.ConfigProto(allow_soft_placement=False)  # 创建sess的时候对sess进行参数配置
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
            tmp_lr = tmp_lr * (1 - lr_decay)

            _, loss_s, loss_i, loss_r, loss_d, loss, pwb2beta, pwb2gamma, pwb2mu = sess.run(
                [train_Losses, Loss2dS, Loss2dI, Loss2dR, Loss2dD, Loss, PWB2Beta, PWB2Gamma, PWB2Mu],
                feed_dict={T_train: t_batch, S_observe: s_obs, I_observe: i_obs, R_observe: r_obs, D_observe: d_obs,
                           in_learning_rate: tmp_lr})

            loss_s_all.append(loss_s)
            loss_i_all.append(loss_i)
            loss_r_all.append(loss_r)
            loss_d_all.append(loss_d)
            loss_all.append(loss)

            if i_epoch % 1000 == 0:
                print_and_log2train(i_epoch, time.time() - t0, tmp_lr, pwb2beta, pwb2gamma, pwb2mu, loss_s,
                                    loss_i, loss_r, loss_d, loss, log_out=log_fileout)

                # 以下代码为输出训练过程中 beta, gamma, mu 的训练结果
                test_epoch.append(i_epoch / 1000)
                beta2train, gamma2train, mu2train = sess.run([betaNN2train_test, gammaNN2train_test, muNN2train_test],
                                                             feed_dict={T_train2test: np.reshape(train_date, [-1, 1])})

                # 以下代码为输出 beta, gamma, mu 的测试结果
                beta2test, gamma2test, mu2test = sess.run([betaNN2test, gammaNN2test, muNN2test],
                                                          feed_dict={T_test: test_t_bach})

                # s2t, i2t, r2t, d2t = sess.run([dS2dt, dI2dt, dR2dt, dD2dt],
                #                               feed_dict={T_train: t_batch, S_observe: s_obs, I_observe: i_obs,
                #                                          R_observe: r_obs, D_observe: d_obs})
                #
                # print('dS/dt:', s2t)
                # print('dI/dt:', i2t)
                # print('dR/dt:', r2t)
                # print('dD/dt:', d2t)

        saveData.save_trainParas2mat_Covid(beta2train, name2para='beta2train', outPath=R['FolderName'])
        saveData.save_trainParas2mat_Covid(gamma2train, name2para='gamma2train', outPath=R['FolderName'])
        saveData.save_trainParas2mat_Covid(mu2train, name2para='mu2train', outPath=R['FolderName'])

        plotData.plot_Para2convid(beta2train, name2para='beta_train',
                                  coord_points2test=np.reshape(train_date, [-1, 1]), outPath=R['FolderName'])
        plotData.plot_Para2convid(gamma2train, name2para='gamma_train',
                                  coord_points2test=np.reshape(train_date, [-1, 1]), outPath=R['FolderName'])
        plotData.plot_Para2convid(mu2train, name2para='mu_train',
                                  coord_points2test=np.reshape(train_date, [-1, 1]), outPath=R['FolderName'])

        saveData.save_SIRD_trainLoss2mat_no_N(loss_s_all, loss_i_all, loss_r_all, loss_d_all, actName=act_func2paras,
                                              outPath=R['FolderName'])

        plotData.plotTrain_loss_1act_func(loss_s_all, lossType='loss2s', seedNo=R['seed'], outPath=R['FolderName'],
                                          yaxis_scale=True)
        plotData.plotTrain_loss_1act_func(loss_i_all, lossType='loss2i', seedNo=R['seed'], outPath=R['FolderName'],
                                          yaxis_scale=True)
        plotData.plotTrain_loss_1act_func(loss_r_all, lossType='loss2r', seedNo=R['seed'], outPath=R['FolderName'],
                                          yaxis_scale=True)
        plotData.plotTrain_loss_1act_func(loss_d_all, lossType='loss2d', seedNo=R['seed'], outPath=R['FolderName'],
                                          yaxis_scale=True)

        saveData.save_SIRD_testParas2mat(beta2test, gamma2test, mu2test, name2para1='beta2test',
                                         name2para2='gamma2test', name2para3='mu2test', outPath=R['FolderName'])

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
    store_file = 'EulerSIRD'
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
    R['max_epoch'] = 150000
    if 0 != R['activate_stop']:
        epoch_stop = input('please input a stop epoch:')
        R['max_epoch'] = int(epoch_stop)

    # ----------------------------------------- Convid 设置 ---------------------------------
    R['eqs_name'] = 'SIRD'
    R['input_dim'] = 1                       # 输入维数，即问题的维数(几元问题)
    R['output_dim'] = 1                      # 输出维数
    R['total_population'] = 3450000          # 总的“人口”数量

    # R['normalize_population'] = 3450000      # 归一化时使用的“人口”数值
    # R['normalize_population'] = 100000
    R['normalize_population'] = 1

    # ------------------------------------  神经网络的设置  ----------------------------------------
    R['size2train'] = 250                    # 训练集的大小
    # R['batch_size2train'] = 10               # 训练数据的批大小
    R['batch_size2train'] = 16               # 训练数据的批大小
    # R['batch_size2train'] = 30              # 训练数据的批大小
    R['batch_size2test'] = 50               # 训练数据的批大小
    # R['opt2sample'] = 'random_sample'     # 训练集的选取方式--随机采样
    # R['opt2sample'] = 'rand_sample_sort'    # 训练集的选取方式--随机采样后按时间排序
    R['opt2sample'] = 'windows_rand_sample'  # 训练集的选取方式--随机窗口采样(以随机点为基准，然后滑动窗口采样)

    # R['regular_weight_model'] = 'L0'
    # R['regular_weight_model'] = 'L1'
    R['regular_weight_model'] = 'L2'          # The model of regular weights and biases
    R['regular_weight'] = 0.001             # Regularization parameter for weights
    # R['regular_weight'] = 0.0005            # Regularization parameter for weights
    # R['regular_weight'] = 0.0001            # Regularization parameter for weights
    # R['regular_weight'] = 0.00005             # Regularization parameter for weights
    # R['regular_weight'] = 0.00001           # Regularization parameter for weights

    R['optimizer_name'] = 'Adam'              # 优化器
    R['loss_function'] = 'L2_loss'            # 损失函数的类型
    # R['loss_function'] = 'lncosh_loss'      # 损失函数的类型(Nan 了, 不收敛)

    R['train_model'] = 'train_union_loss'     # 训练模式:各个不同的loss累加在一起，训练

    if 50000 < R['max_epoch']:
        R['learning_rate'] = 1e-2             # 学习率
        R['lr_decay'] = 1e-4                  # 学习率 decay

        # R['learning_rate'] = 2e-3             # 学习率
        # R['lr_decay'] = 1e-4                  # 学习率 decay
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

    # SIRD参数网络模型的选择
    # R['model2paras'] = 'DNN'
    # R['model2paras'] = 'DNN_scale'
    # R['model2paras'] = 'DNN_scaleOut'
    R['model2paras'] = 'DNN_FourierBase'

    # SIRD参数网络模型的隐藏层单元数目
    if R['model2paras'] == 'DNN_FourierBase':
        R['hidden2para'] = (35, 50, 30, 30, 20)  # 1*50+50*50+50*30+30*30+30*20+20*1 = 5570
    else:
        # R['hidden2para'] = (10, 10, 8, 6, 6, 3)       # it is used to debug our work
        R['hidden2para'] = (70, 50, 30, 30, 20)  # 1*50+50*50+50*30+30*30+30*20+20*1 = 5570
        # R['hidden2para'] = (80, 80, 60, 40, 40, 20)   # 80+80*80+80*60+60*40+40*40+40*20+20*1 = 16100
        # R['hidden2para'] = (100, 100, 80, 60, 60, 40)
        # R['hidden2para'] = (200, 100, 100, 80, 50, 50)

    # SIRD参数网络模型的尺度因子
    if R['model2paras'] != 'DNN':
        R['freq2paras'] = np.concatenate(([1], np.arange(1, 25)), axis=0)

    # SIRD参数网络模型为傅里叶网络和尺度网络时，重复高频因子或者低频因子
    if R['model2paras'] == 'DNN_FourierBase' or R['model2paras'] == 'DNN_scale':
        R['if_repeat_High_freq2paras'] = False

    # SIRD参数网络模型的激活函数的选择
    # R['actIn_Name2paras'] = 'relu'
    # R['actIn_Name2paras'] = 'leaky_relu'
    # R['actIn_Name2paras'] = 'sigmoid'
    # R['actIn_Name2paras'] = 'tanh'
    # R['actIn_Name2paras'] = 'srelu'
    # R['actIn_Name2paras'] = 's2relu'
    R['actIn_Name2paras'] = 'sin'
    # R['actIn_Name2paras'] = 'sinAddcod'
    # R['actIn_Name2paras'] = 'elu'
    # R['actIn_Name2paras'] = 'gelu'
    # R['actIn_Name2paras'] = 'mgelu'
    # R['actIn_Name2paras'] = 'linear'

    # R['act_Name2paras'] = 'relu'
    # R['act_Name2paras'] = 'leaky_relu'
    # R['act_Name2paras'] = 'sigmoid'
    # R['act_Name2paras'] = 'tanh'  # 这个激活函数比较s2ReLU合适
    # R['act_Name2paras'] = 'srelu'
    # R['act_Name2paras'] = 's2relu'
    R['act_Name2paras'] = 'sin'
    # R['act_Name2paras'] = 'sinAddcos'
    # R['act_Name2paras'] = 'elu'
    # R['act_Name2paras'] = 'gelu'
    # R['act_Name2paras'] = 'mgelu'
    # R['act_Name2paras'] = 'linear'

    solve_SIRD2COVID(R)

