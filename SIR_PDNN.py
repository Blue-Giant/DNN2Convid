"""
@author: LXA
Benchmark Code of SIR model
2022-06-17
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
import DNN_LogPrint


# 记录字典中的一些设置
def dictionary_out2file(R_dic, log_fileout):
    DNN_tools.log_string('Equation name for problem: %s\n' % (R_dic['eqs_name']), log_fileout)
    DNN_tools.log_string('Network model of dealing with parameters: %s\n' % str(R_dic['model2paras']), log_fileout)

    if str.upper(R_dic['model2paras']) == 'DNN_FOURIERBASE':
        DNN_tools.log_string('The input activate function for parameter: %s\n' % '[sin;cos]', log_fileout)
    else:
        DNN_tools.log_string('The input activate function for parameter: %s\n' % str(R_dic['actIn_Name2paras']), log_fileout)

    if str.upper(R_dic['model2paras']) != 'DNN':
        DNN_tools.log_string('The scale for frequency to SIR NN: %s\n' % str(R_dic['freq2paras']), log_fileout)
        DNN_tools.log_string('Repeat the high-frequency scale or not for para-NN: %s\n' % str(R_dic['if_repeat_High_freq2paras']), log_fileout)

    DNN_tools.log_string('The hidden-layer activate function for parameter: %s\n' % str(R_dic['act_Name2paras']), log_fileout)

    DNN_tools.log_string('hidden layers for parameters: %s\n' % str(R_dic['hidden2para']), log_fileout)

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

    DNN_tools.log_string('The model of regular weights and biases: %s\n' % str(R_dic['regular_weight_model']), log_fileout)

    DNN_tools.log_string('Regularization parameter for weights and biases: %s\n' % str(R_dic['regular_weight']), log_fileout)

    DNN_tools.log_string('Size 2 training set: %s\n' % str(R_dic['size2train']), log_fileout)

    DNN_tools.log_string('Batch-size 2 training: %s\n' % str(R_dic['batch_size2train']), log_fileout)

    DNN_tools.log_string('Batch-size 2 testing: %s\n' % str(R_dic['batch_size2test']), log_fileout)


def print_and_log2train(i_epoch, run_time, tmp_lr, loss_all, loss_s, loss_i, loss_r, penalty_wb2beta,
                        penalty_wb2gamma, log_out=None):
    print('train epoch: %d, time: %.3f' % (i_epoch, run_time))
    print('learning rate: %f' % tmp_lr)
    print('penalty weights and biases for Beta: %f' % penalty_wb2beta)
    print('penalty weights and biases for Gamma: %f' % penalty_wb2gamma)
    print('loss for S: %.16f' % loss_s)
    print('loss for I: %.16f' % loss_i)
    print('loss for R: %.16f' % loss_r)
    print('total loss: %.16f\n' % loss_all)

    DNN_tools.log_string('train epoch: %d,time: %.3f' % (i_epoch, run_time), log_out)
    DNN_tools.log_string('learning rate: %f' % tmp_lr, log_out)
    DNN_tools.log_string('penalty weights and biases for Beta: %f' % penalty_wb2beta, log_out)
    DNN_tools.log_string('penalty weights and biases for Gamma: %f' % penalty_wb2gamma, log_out)
    DNN_tools.log_string('loss for S: %.16f' % loss_s, log_out)
    DNN_tools.log_string('loss for I: %.16f' % loss_i, log_out)
    DNN_tools.log_string('loss for R: %.16f' % loss_r, log_out)
    DNN_tools.log_string('total loss: %.16f \n\n' % loss_all, log_out)


def solve_SIR2COVID(R):
    log_out_path = R['FolderName']        # 将路径从字典 R 中提取出来
    if not os.path.exists(log_out_path):  # 判断路径是否已经存在
        os.mkdir(log_out_path)            # 无 log_out_path 路径，创建一个 log_out_path 路径
    log_fileout = open(os.path.join(log_out_path, 'log_train.txt'), 'w')  # 在这个路径下创建并打开一个可写的 log_train.txt文件
    dictionary_out2file(R, log_fileout)

    log2testParas = open(os.path.join(log_out_path, 'test_Paras.txt'), 'w')  # 在这个路径下创建并打开一个可写的 log_train.txt文件

    trainSet_szie = R['size2train']                    # 训练集大小,给定一个数据集，拆分训练集和测试集时，需要多大规模的训练集
    batchSize_train = R['batch_size2train']            # 训练批量的大小,该值远小于训练集大小
    batchSize_test = R['batch_size2test']              # 测试批量的大小,该值小于等于测试集大小
    wb_penalty = R['regular_weight']                   # 神经网络参数的惩罚因子
    lr_decay = R['lr_decay']                           # 学习率额衰减
    init_lr = R['learning_rate']                       # 初始学习率

    act_func2paras = R['act_Name2paras']               # 参数网络的隐藏层激活函数

    input_dim = R['input_dim']                         # 输入维度
    out_dim = R['output_dim']                          # 输出维度

    flag2beta = 'WB2beta'                              # beta 的网络的变量空间标识
    flag2gamma = 'WB2gamma'                            # gamma 的网络的变量空间标识
    hidden_para = R['hidden2para']

    AI = tf.eye(batchSize_train, dtype=tf.float32) * 2
    Ones_mat = tf.ones([batchSize_train, batchSize_train], dtype=tf.float32)
    A_diag = tf.linalg.band_part(Ones_mat, 0, 1)
    Amat = AI - A_diag

    # 初始化神经网络的参数及其方式
    if str.upper(R['model2paras']) == 'DNN_FOURIERBASE':
        Weight2beta, Bias2beta = DNN_base.Xavier_init_NN_Fourier(input_dim, out_dim, hidden_para, flag2beta)
        Weight2gamma, Bias2gamma = DNN_base.Xavier_init_NN_Fourier(input_dim, out_dim, hidden_para, flag2gamma)
    else:
        Weight2beta, Bias2beta = DNN_base.Xavier_init_NN(input_dim, out_dim, hidden_para, flag2beta)
        Weight2gamma, Bias2gamma = DNN_base.Xavier_init_NN(input_dim, out_dim, hidden_para, flag2gamma)

    global_steps = tf.Variable(0, trainable=False)
    with tf.device('/gpu:%s' % (R['gpuNo'])):
        with tf.compat.v1.variable_scope('vscope', reuse=tf.compat.v1.AUTO_REUSE):
            T_train = tf.compat.v1.placeholder(tf.float32, name='T_train', shape=[batchSize_train, input_dim])
            S_observe = tf.compat.v1.placeholder(tf.float32, name='S_observe', shape=[batchSize_train, input_dim])
            I_observe = tf.compat.v1.placeholder(tf.float32, name='I_observe', shape=[batchSize_train, input_dim])
            R_observe = tf.compat.v1.placeholder(tf.float32, name='R_observe', shape=[batchSize_train, input_dim])
            in_learning_rate = tf.compat.v1.placeholder_with_default(input=1e-5, shape=[], name='lr')
            T_test = tf.compat.v1.placeholder(tf.float32, name='T_test', shape=[batchSize_test, input_dim])

            freq2paras = R['freq2paras']
            if 'DNN' == str.upper(R['model2paras']):
                in_beta2train = DNN_base.DNN(T_train, Weight2beta, Bias2beta, hidden_para,
                                             activateIn_name=R['actIn_Name2paras'], activate_name=act_func2paras)
                in_gamma2train = DNN_base.DNN(T_train, Weight2gamma, Bias2gamma, hidden_para,
                                              activateIn_name=R['actIn_Name2paras'], activate_name=act_func2paras)

                in_beta2test = DNN_base.DNN(T_test, Weight2beta, Bias2beta, hidden_para,
                                            activateIn_name=R['actIn_Name2paras'], activate_name=act_func2paras)
                in_gamma2test = DNN_base.DNN(T_test, Weight2gamma, Bias2gamma, hidden_para,
                                             activateIn_name=R['actIn_Name2paras'], activate_name=act_func2paras)
            elif 'DNN_SCALE' == str.upper(R['model2paras']):
                in_beta2train = DNN_base.DNN_scale(T_train, Weight2beta, Bias2beta, hidden_para, freq2paras,
                                                   activateIn_name=R['actIn_Name2paras'], activate_name=act_func2paras,
                                                   repeat_Highfreq=R['if_repeat_High_freq2paras'])
                in_gamma2train = DNN_base.DNN_scale(T_train, Weight2gamma, Bias2gamma, hidden_para, freq2paras,
                                                    activateIn_name=R['actIn_Name2paras'], activate_name=act_func2paras,
                                                    repeat_Highfreq=R['if_repeat_High_freq2paras'])
                in_beta2test = DNN_base.DNN_scale(T_test, Weight2beta, Bias2beta, hidden_para, freq2paras,
                                                  activateIn_name=R['actIn_Name2paras'], activate_name=act_func2paras,
                                                  repeat_Highfreq=R['if_repeat_High_freq2paras'])
                in_gamma2test = DNN_base.DNN_scale(T_test, Weight2gamma, Bias2gamma, hidden_para, freq2paras,
                                                   activateIn_name=R['actIn_Name2paras'], activate_name=act_func2paras,
                                                   repeat_Highfreq=R['if_repeat_High_freq2paras'])
            elif str.upper(R['model2paras']) == 'DNN_FOURIERBASE':
                in_beta2train = DNN_base.DNN_FourierBase(T_train, Weight2beta, Bias2beta, hidden_para, freq2paras,
                                                         activate_name=act_func2paras, sFourier=1.0,
                                                         repeat_Highfreq=R['if_repeat_High_freq2paras'])
                in_gamma2train = DNN_base.DNN_FourierBase(T_train, Weight2gamma, Bias2gamma, hidden_para, freq2paras,
                                                          activate_name=act_func2paras, sFourier=1.0,
                                                          repeat_Highfreq=R['if_repeat_High_freq2paras'])
                in_beta2test = DNN_base.DNN_FourierBase(T_test, Weight2beta, Bias2beta, hidden_para, freq2paras,
                                                        activate_name=act_func2paras, sFourier=1.0,
                                                        repeat_Highfreq=R['if_repeat_High_freq2paras'])
                in_gamma2test = DNN_base.DNN_FourierBase(T_test, Weight2gamma, Bias2gamma, hidden_para, freq2paras,
                                                         activate_name=act_func2paras, sFourier=1.0,
                                                         repeat_Highfreq=R['if_repeat_High_freq2paras'])

            # Remark: beta, gamma,S_NN.I_NN,R_NN都应该是正的. beta.1--15之间，gamma在(0,1）使用归一化的话S_NN.I_NN,R_NN都在[0,1)范围内
            betaNN2train = tf.square(in_beta2train)
            gammaNN2train = tf.nn.sigmoid(in_gamma2train)

            betaNN2test = tf.square(in_beta2test)
            gammaNN2test = tf.nn.sigmoid(in_gamma2test)

            dS2dt = tf.matmul(Amat[0:-1, :], S_observe)
            dI2dt = tf.matmul(Amat[0:-1, :], I_observe)
            dR2dt = tf.matmul(Amat[0:-1, :], R_observe)

            temp_s2t = -betaNN2train[0:-1, 0]*S_observe[0:-1, 0]*I_observe[0:-1, 0]
            temp_i2t = betaNN2train[0:-1, 0]*S_observe[0:-1, 0]*I_observe[0:-1, 0] - \
                       gammaNN2train[0:-1, 0] * I_observe[0:-1, 0]
            temp_r2t = gammaNN2train[0:-1, 0] * I_observe[0:-1, 0]

            if str.lower(R['loss_function']) == 'l2_loss':
                Loss2dS = tf.reduce_mean(tf.square(dS2dt - tf.reshape(temp_s2t, shape=[-1, 1])))
                Loss2dI = tf.reduce_mean(tf.square(dI2dt - tf.reshape(temp_i2t, shape=[-1, 1])))
                Loss2dR = tf.reduce_mean(tf.square(dR2dt - tf.reshape(temp_r2t, shape=[-1, 1])))
            elif str.lower(R['loss_function']) == 'lncosh_loss':
                Loss2dS = tf.reduce_mean(tf.log(tf.cosh(dS2dt - tf.reshape(temp_s2t, shape=[-1, 1]))))
                Loss2dI = tf.reduce_mean(tf.log(tf.cosh(dI2dt - tf.reshape(temp_i2t, shape=[-1, 1]))))
                Loss2dR = tf.reduce_mean(tf.log(tf.cosh(dR2dt - tf.reshape(temp_r2t, shape=[-1, 1]))))

            if R['regular_weight_model'] == 'L1':
                regular_WB2Beta = DNN_base.regular_weights_biases_L1(Weight2beta, Bias2beta)
                regular_WB2Gamma = DNN_base.regular_weights_biases_L1(Weight2gamma, Bias2gamma)
            elif R['regular_weight_model'] == 'L2':
                regular_WB2Beta = DNN_base.regular_weights_biases_L2(Weight2beta, Bias2beta)
                regular_WB2Gamma = DNN_base.regular_weights_biases_L2(Weight2gamma, Bias2gamma)
            else:
                regular_WB2Beta = tf.constant(0.0)
                regular_WB2Gamma = tf.constant(0.0)

            PWB2Beta = wb_penalty * regular_WB2Beta
            PWB2Gamma = wb_penalty * regular_WB2Gamma

            Loss = Loss2dS + Loss2dI + Loss2dR + PWB2Beta + PWB2Gamma

            my_optimizer = tf.compat.v1.train.AdamOptimizer(in_learning_rate)
            train_Losses = my_optimizer.minimize(Loss, global_step=global_steps)

    t0 = time.time()
    loss_s_all, loss_i_all, loss_r_all, loss_all = [], [], [], []
    test_epoch = []

    # filename = 'data2csv/Wuhan.csv'
    # filename = 'data2csv/Italia_data.csv'
    filename = 'data2csv/Korea_data.csv'
    # filename = 'data2csv/minnesota.csv'

    # date, data2I, data2S = DNN_data.load_2csvData_cal_S(datafile=filename, total_population=R['total_population'])

    date, data2I, data2S, data2R = DNN_data.load_2csvData_cal_S_R(datafile=filename, total_population=R['total_population'])

    assert(trainSet_szie + batchSize_test <= len(data2I))
    if R['normalize_population'] == 1:
        # 不归一化数据
        train_date, train_data2i, train_data2s, test_date, test_data2i, test_data2s = \
            DNN_data.split_3csvData2train_test(date, data2I, data2S, size2train=trainSet_szie, normalFactor=1.0)
        nbatch2train = np.ones(batchSize_train, dtype=np.float32)*float(R['total_population'])

    elif (R['total_population'] != R['normalize_population']) and R['normalize_population'] != 1:
        # 归一化数据，使用的归一化数值小于总“人口”
        train_date, train_data2i, train_data2s, test_date, test_data2i, test_data2s = \
            DNN_data.split_3csvData2train_test(date, data2I, data2S, size2train=trainSet_szie,
                                               normalFactor=R['normalize_population'])
        nbatch2train = np.ones(batchSize_train, dtype=np.float32) * (float(R['total_population'])/float(R['normalize_population']))

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
    r_obs_test = DNN_data.sample_testData_serially(test_data2r, batchSize_test, normalFactor=1.0)

    print('The test data about i:\n', str(np.transpose(i_obs_test)))
    print('\n')
    print('The test data about s:\n', str(np.transpose(s_obs_test)))
    print('\n')
    DNN_tools.log_string('The test data about i:\n%s\n' % str(np.transpose(i_obs_test)), log_fileout)
    DNN_tools.log_string('The test data about s:\n%s\n' % str(np.transpose(s_obs_test)), log_fileout)

    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True              # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True                  # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        tmp_lr = init_lr
        for i_epoch in range(R['max_epoch'] + 1):
            # 由于将数据拆分为训练数据和测试数据时，进行了归一化处理，故这里不用归一化
            t_batch, i_obs, s_obs = \
                DNN_data.randSample_Normalize_3existData(train_date, train_data2i, train_data2s, batchsize=batchSize_train,
                                                         normalFactor=1.0, sampling_opt=R['opt2sample'])
            tmp_lr = tmp_lr * (1 - lr_decay)
            _, loss, loss_s, loss_i, loss_r, pwb2beta, pwb2gamma = sess.run(
                [train_Losses, Loss, Loss2dS, Loss2dI, Loss2dR, PWB2Beta, PWB2Gamma],
                feed_dict={T_train: t_batch, I_observe: i_obs, S_observe: s_obs, R_observe: r_obs, in_learning_rate: tmp_lr})
            loss_all.append(loss)
            loss_s_all.append(loss_s)
            loss_i_all.append(loss_i)
            loss_r_all.append(loss_r)

            if i_epoch % 1000 == 0:
                print_and_log2train(i_epoch, time.time() - t0, tmp_lr, loss, loss_s, loss_i, loss_r, pwb2beta,
                                    pwb2gamma, log_out=log_fileout)

                # 以下代码为输出训练过程中 beta, gamma 的测试结果
                test_epoch.append(i_epoch / 1000)
                test_beta, test_gamma = sess.run([betaNN2test, gammaNN2test], feed_dict={T_test: test_t_bach})

                DNN_tools.log_string('------------------The epoch----------------------: %s\n' % str(i_epoch), log2testParas)
                DNN_tools.log_string('The test result for beta:\n%s\n' % str(np.transpose(test_beta)), log2testParas)
                DNN_tools.log_string('The test result for gamma:\n%s\n' % str(np.transpose(test_gamma)), log2testParas)

                # --------以下代码为输出训练过程中 in_beta, in_gamma 的测试结果-------------
                in_beta_test, in_gamma_test = sess.run([in_beta2test, in_gamma2test], feed_dict={T_test: test_t_bach})

                DNN_tools.log_string('------------------The epoch----------------------: %s\n' % str(i_epoch), log2testParas)
                DNN_tools.log_string('The test result for in_beta:\n%s\n' % str(np.transpose(in_beta_test)), log2testParas)
                DNN_tools.log_string('The test result for in_gamma:\n%s\n' % str(np.transpose(in_gamma_test)), log2testParas)

        # 训练完成，保存最终的结果并画图
        saveData.save_SIR_trainLoss2mat(loss_s_all, loss_i_all, loss_r_all, loss_all, actName=act_func2paras,
                                        outPath=R['FolderName'])

        plotData.plotTrain_loss_1act_func(loss_s_all, lossType='loss2s', seedNo=R['seed'], outPath=R['FolderName'],
                                          yaxis_scale=True)
        plotData.plotTrain_loss_1act_func(loss_i_all, lossType='loss2i', seedNo=R['seed'], outPath=R['FolderName'],
                                          yaxis_scale=True)
        plotData.plotTrain_loss_1act_func(loss_r_all, lossType='loss2r', seedNo=R['seed'], outPath=R['FolderName'],
                                          yaxis_scale=True)
        plotData.plotTrain_loss_1act_func(loss_all, lossType='loss', seedNo=R['seed'], outPath=R['FolderName'],
                                          yaxis_scale=True)

        saveData.save_SIR_testParas2mat_Covid(test_beta, test_gamma, name2para1='beta2test', name2para2='gamma2test',
                                              outPath=R['FolderName'])

        plotData.plot_testSolu2convid(test_beta, name2solu='beta_test', coord_points2test=test_t_bach,
                                      outPath=R['FolderName'])
        plotData.plot_testSolu2convid(test_gamma, name2solu='gamma_test', coord_points2test=test_t_bach,
                                      outPath=R['FolderName'])


if __name__ == "__main__":
    R={}
    R['gpuNo'] = 0  # 默认使用 GPU，这个标记就不要设为-1，设为0,1,2,3,4....n（n指GPU的数目，即电脑有多少块GPU）

    # 文件保存路径设置
    store_file = 'SIR2covid'
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(BASE_DIR)
    OUT_DIR = os.path.join(BASE_DIR, store_file)
    if not os.path.exists(OUT_DIR):
        print('---------------------- OUT_DIR ---------------------:', OUT_DIR)
        os.mkdir(OUT_DIR)

    R['seed'] = np.random.randint(1e5)
    seed_str = str(R['seed'])                     # int 型转为字符串型
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
    R['eqs_name'] = 'SIR'
    R['input_dim'] = 1                    # 输入维数，即问题的维数(几元问题)
    R['output_dim'] = 1                   # 输出维数
    R['total_population'] = 9776000       # 总的“人口”数量(韩国)

    # R['normalize_population'] = 9776000       # 归一化时使用的“人口”数值
    R['normalize_population'] = 100000
    # R['normalize_population'] = 1

    # ------------------------------------  神经网络的设置  ----------------------------------------
    R['size2train'] = 70                       # 训练集的大小
    R['batch_size2train'] = 20                 # 训练数据的批大小
    R['batch_size2test'] = 10                  # 测试数据的批大小
    # R['opt2sample'] = 'random_sample'        # 训练集的选取方式--随机采样
    # R['opt2sample'] = 'rand_sample_sort'       # 训练集的选取方式--随机采样后按时间排序
    R['opt2sample'] = 'windows_rand_sample'    # 训练集的选取方式--随机窗口采样(以随机点为基准，然后滑动窗口采样)

    # R['regular_weight_model'] = 'L0'      # The model of regular weights and biases
    # R['regular_weight'] = 0.000           # Regularization parameter for weights

    # R['regular_weight_model'] = 'L1'
    R['regular_weight_model'] = 'L2'        # The model of regular weights and biases
    # R['regular_weight'] = 0.001           # Regularization parameter for weights
    # R['regular_weight'] = 0.0005          # Regularization parameter for weights
    # R['regular_weight'] = 0.0001          # Regularization parameter for weights
    R['regular_weight'] = 0.00005           # Regularization parameter for weights
    # R['regular_weight'] = 0.00001         # Regularization parameter for weights

    R['optimizer_name'] = 'Adam'           # 优化器
    R['loss_function'] = 'L2_loss'         # 损失函数的类型
    # R['loss_function'] = 'lncosh_loss'   # 损失函数的类型
    R['scale_up'] = 1                      # scale_up 用来控制湿粉扑对数值进行尺度提升，如1e-6量级提升到1e-2量级。不为 0 代表开启提升
    R['scale_factor'] = 100                # scale_factor 用来对数值进行尺度提升，如1e-6量级提升到1e-2量级

    # R['train_model'] = 'train_group'     # 训练模式:各个不同的loss捆绑打包训练
    R['train_model'] = 'train_union_loss'  # 训练模式:各个不同的loss累加在一起，训练

    if 50000 < R['max_epoch']:
        R['learning_rate'] = 2e-3           # 学习率
        R['lr_decay'] = 1e-4                # 学习率 decay
        # R['learning_rate'] = 2e-4         # 学习率
        # R['lr_decay'] = 5e-5              # 学习率 decay
    elif (20000 < R['max_epoch'] and 50000 >= R['max_epoch']):
        # R['learning_rate'] = 1e-3         # 学习率
        # R['lr_decay'] = 1e-4              # 学习率 decay
        # R['learning_rate'] = 2e-4         # 学习率
        # R['lr_decay'] = 1e-4              # 学习率 decay
        R['learning_rate'] = 1e-4           # 学习率
        R['lr_decay'] = 5e-5                # 学习率 decay
    else:
        R['learning_rate'] = 5e-5           # 学习率
        R['lr_decay'] = 1e-5                # 学习率 decay

    # SIR参数网络模型的选择
    # R['model2paras'] = 'DNN'
    # R['model2paras'] = 'DNN_scale'
    # R['model2paras'] = 'DNN_scaleOut'
    R['model2paras'] = 'DNN_FourierBase'

    # SIR和参数网络模型的隐藏层单元数目
    if R['model2paras'] == 'DNN_FourierBase':
        R['hidden2para'] = (35, 50, 30, 30, 20)         # 1*50+50*50+50*30+30*30+30*20+20*1 = 5570
    else:
        # R['hidden2para'] = (10, 10, 8, 6, 6, 3)       # it is used to debug our work
        R['hidden2para'] = (70, 50, 30, 30, 20)         # 1*50+50*50+50*30+30*30+30*20+20*1 = 5570
        # R['hidden2para'] = (80, 80, 60, 40, 40, 20)   # 80+80*80+80*60+60*40+40*40+40*20+20*1 = 16100
        # R['hidden2para'] = (100, 100, 80, 60, 60, 40)
        # R['hidden2para'] = (200, 100, 100, 80, 50, 50)

    # SIR和参数网络模型的尺度因子
    if R['model2paras'] != 'DNN':
        R['freq2paras'] = np.concatenate(([1], np.arange(1, 20)), axis=0)

    # SIR和参数网络模型为傅里叶网络和尺度网络时，重复高频因子或者低频因子
    if R['model2paras'] == 'DNN_FourierBase' or R['model2paras'] == 'DNN_scale':
        R['if_repeat_High_freq2paras'] = False

    # SIR参数网络模型的激活函数的选择
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
    R['act_Name2paras'] = 'tanh'                 # 这个激活函数比较s2ReLU合适
    # R['act_Name2paras'] = 'srelu'
    # R['act_Name2paras'] = 's2relu'
    # R['act_Name2paras'] = 'sin'
    # R['act_Name2paras'] = 'sinAddcos'
    # R['act_Name2paras'] = 'elu'
    # R['act_Name2paras'] = 'gelu'
    # R['act_Name2paras'] = 'mgelu'
    # R['act_Name2paras'] = 'linear'

    solve_SIR2COVID(R)
