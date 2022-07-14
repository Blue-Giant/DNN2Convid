import numpy as np
import csv


def load_2csvData(datafile=None):
    csvdata_list = []
    csvdate_list = []
    icount = 0
    csvreader = csv.reader(open(datafile, 'r'))
    for dataItem2csv in csvreader:
        if str.isnumeric(dataItem2csv[1]):
            csvdata_list.append(int(dataItem2csv[1]))
            csvdate_list.append(icount)
            icount = icount + 1
    csvdate = np.array(csvdate_list)
    csvdata = np.array(csvdata_list)
    return csvdate, csvdata


def load_2csvData_cal_S(datafile=None, total_population=100000):
    csvdata2I_list = []
    csvdata2S_list = []
    csvdate_list = []
    icount = 0
    csvreader = csv.reader(open(datafile, 'r'))
    for dataItem2csv in csvreader:
        if str.isnumeric(dataItem2csv[1]):
            csvdata2I_list.append(int(dataItem2csv[1]))
            csvdata2S_list.append(int(total_population)-int(dataItem2csv[1]))
            csvdate_list.append(icount)
            icount = icount + 1
    csvdate = np.array(csvdate_list)
    csvdata2I = np.array(csvdata2I_list)
    csvdata2S = np.array(csvdata2S_list)
    return csvdate, csvdata2I, csvdata2S


def load_3csvData(datafile=None):
    csvdata1_list = []
    csvdata2_list = []
    csvdate_list = []
    icount = 0
    csvreader = csv.reader(open(datafile, 'r'))
    for dataItem2csv in csvreader:
        if str.isnumeric(dataItem2csv[1]):
            csvdata1_list.append(int(dataItem2csv[1]))
            csvdata2_list.append(int(dataItem2csv[2]))
            csvdate_list.append(icount)
            icount = icount + 1
    csvdate = np.array(csvdate_list)
    csvdata1 = np.array(csvdata1_list)
    csvdata2 = np.array(csvdata2_list)
    return csvdate, csvdata1, csvdata2


def load_3csvData_cal_S(datafile=None, total_population=100000):
    csvdata1_list = []
    csvdata2_list = []
    csvdata3_list = []
    csvdate_list = []
    icount = 0
    csvreader = csv.reader(open(datafile, 'r'))
    for dataItem2csv in csvreader:
        if str.isnumeric(dataItem2csv[1]):
            csvdata1_list.append(int(dataItem2csv[1]))
            csvdata2_list.append(int(dataItem2csv[2]))
            csvdata3_list.append(int(total_population)-int(dataItem2csv[2]))
            csvdate_list.append(icount)
            icount = icount + 1
    csvdate = np.array(csvdate_list)
    csvdata1 = np.array(csvdata1_list)
    csvdata2 = np.array(csvdata2_list)
    csvdata3 = np.array(csvdata2_list)
    return csvdate, csvdata1, csvdata2, csvdata3


def load_4csvData(datafile=None):
    csvdata1_list = []
    csvdata2_list = []
    csvdata3_list = []
    csvdate_list = []
    icount = 0
    csvreader = csv.reader(open(datafile, 'r'))
    for dataItem2csv in csvreader:
        if str.isnumeric(dataItem2csv[1]):
            csvdata1_list.append(int(dataItem2csv[1]))
            csvdata2_list.append(int(dataItem2csv[2]))
            csvdata3_list.append(int(dataItem2csv[3]))
            csvdate_list.append(icount)
            icount = icount + 1
    csvdate = np.array(csvdate_list)
    csvdata1 = np.array(csvdata1_list)
    csvdata2 = np.array(csvdata2_list)
    csvdata3 = np.array(csvdata3_list)
    return csvdate, csvdata1, csvdata2, csvdata3


def load_5csvData(datafile=None):
    csvdata1_list = []
    csvdata2_list = []
    csvdata3_list = []
    csvdata4_list = []
    csvdate_list = []
    icount = 0
    csvreader = csv.reader(open(datafile, 'r'))
    for dataItem2csv in csvreader:
        if str.isnumeric(dataItem2csv[1]):
            csvdata1_list.append(int(dataItem2csv[1]))
            csvdata2_list.append(int(dataItem2csv[2]))
            csvdata3_list.append(int(dataItem2csv[3]))
            csvdata4_list.append(int(dataItem2csv[4]))
            csvdate_list.append(icount)
            icount = icount + 1
    csvdate = np.array(csvdate_list)
    csvdata1 = np.array(csvdata1_list)
    csvdata2 = np.array(csvdata2_list)
    csvdata3 = np.array(csvdata3_list)
    csvdata4 = np.array(csvdata4_list)
    return csvdate, csvdata1, csvdata2, csvdata3, csvdata4


def load_4csvData_cal_S(datafile=None, total_population=3450000):
    csvdata2I_list = []
    csvdata2R_list = []
    csvdata2D_list = []
    csvdata2S_list = []
    csvdate_list = []
    icount = 1
    csvreader = csv.reader(open(datafile, 'r'))
    for dataItem2csv in csvreader:
        if str.isnumeric(dataItem2csv[1]):
            # csvdata2I_list.append(int(dataItem2csv[1]))

            csvdata2I_list.append(int(dataItem2csv[1]) - int(dataItem2csv[2]) - int(dataItem2csv[3]))
            csvdata2R_list.append(int(dataItem2csv[2]))
            csvdata2D_list.append(int(dataItem2csv[3]))
            # csvdata2S_list.append(total_population-int(dataItem2csv[1])-int(dataItem2csv[2])-int(dataItem2csv[3]))
            csvdata2S_list.append(total_population - int(dataItem2csv[1]))
            csvdate_list.append(icount)
            icount = icount + 1
    csvdate = np.array(csvdate_list)
    csvdata2I = np.array(csvdata2I_list)
    csvdata2R = np.array(csvdata2R_list)
    csvdata2D = np.array(csvdata2D_list)
    csvdata2S = np.array(csvdata2S_list)
    return csvdate, csvdata2S, csvdata2I, csvdata2R, csvdata2D


# 将数据集拆分为训练集合测试集
def split_2csvData2train_test(date_data, data, size2train=50, normalFactor=10000):

    date2train = date_data[0:size2train]
    data2train = data[0:size2train]/float(normalFactor)

    date2test = date_data[size2train:-1]
    data2test = data[size2train:-1]/float(normalFactor)
    return date2train, data2train, date2test, data2test


# 将数据集拆分为训练集合测试集
def split_3csvData2train_test(date_data, data1, data2, size2train=50, normalFactor=10000):

    date2train = date_data[0:size2train]
    data1_train = data1[0:size2train]/float(normalFactor)
    data2_train = data2[0:size2train] / float(normalFactor)

    date2test = date_data[size2train:-1]
    data1_test = data1[size2train:-1]/float(normalFactor)
    data2_test = data2[size2train:-1] / float(normalFactor)
    return date2train, data1_train, data2_train, date2test, data1_test, data2_test


# 将数据集拆分为训练集合测试集
def split_4csvData2train_test(date_data, data1, data2, data3, size2train=50, normalFactor=1.0):

    date2train = date_data[0:size2train]
    data1_train = data1[0:size2train]/float(normalFactor)
    data2_train = data2[0:size2train] / float(normalFactor)
    data3_train = data3[0:size2train] / float(normalFactor)

    date2test = date_data[size2train:-1]
    data1_test = data1[size2train:-1]/float(normalFactor)
    data2_test = data2[size2train:-1] / float(normalFactor)
    data3_test = data3[size2train:-1] / float(normalFactor)
    return date2train, data1_train, data2_train, data3_train, date2test, data1_test, data2_test, data3_test


# 将数据集拆分为训练集合测试集
def split_5csvData2train_test(date_data, data1, data2, data3, data4, size2train=50, normalFactor=1.0):

    date2train = date_data[0:size2train]
    data1_train = data1[0:size2train]/float(normalFactor)
    data2_train = data2[0:size2train] / float(normalFactor)
    data3_train = data3[0:size2train] / float(normalFactor)
    data4_train = data4[0:size2train] / float(normalFactor)

    date2test = date_data[size2train:-1]
    data1_test = data1[size2train:-1]/float(normalFactor)
    data2_test = data2[size2train:-1] / float(normalFactor)
    data3_test = data3[size2train:-1] / float(normalFactor)
    data4_test = data4[size2train:-1] / float(normalFactor)
    return date2train, data1_train, data2_train, data3_train, data4_train, date2test, data1_test, data2_test, data3_test, data4_test


def randSample_existData(data1, data2, batchsize=1):
    data1_temp = []
    data2_temp = []
    data_length = len(data1)
    indexes = np.random.randint(data_length, size=batchsize)
    for i_index in indexes:
        data1_temp .append(data1[i_index])
        data2_temp .append(data2[i_index])
    data1_samples = np.array(data1_temp)
    data2_samples = np.array(data2_temp)
    data1_samples = data1_samples.reshape(batchsize, 1)
    data2_samples = data2_samples.reshape(batchsize, 1)
    return data1_samples, data2_samples


def randSample_3existData(data1, data2, data3, batchsize=1):
    data1_temp = []
    data2_temp = []
    data3_temp = []
    data_length = len(data1)
    indexes = np.random.randint(data_length, size=batchsize)
    for i_index in indexes:
        data1_temp .append(data1[i_index])
        data2_temp .append(data2[i_index])
        data3_temp.append(data3[i_index])
    data1_samples = np.array(data1_temp)
    data2_samples = np.array(data2_temp)
    data3_samples = np.array(data3_temp)
    data1_samples = data1_samples.reshape(batchsize, 1)
    data2_samples = data2_samples.reshape(batchsize, 1)
    data3_samples = data3_samples.reshape(batchsize, 1)
    return data1_samples, data2_samples, data3_samples


# 从总体数据集中载入部分数据作为训练集
def randSample_Normalize_existData(date_data, data2, batchsize=1, normalFactor=1000, sampling_opt=None):
    date_temp = []
    data_temp = []
    data_length = len(date_data)
    if str.lower(sampling_opt) == 'random_sample':
        indexes = np.random.randint(data_length, size=batchsize)
    elif str.lower(sampling_opt) == 'rand_sample_sort':
        indexes_temp = np.random.randint(data_length, size=batchsize)
        indexes = np.sort(indexes_temp)
    else:
        index_base = np.random.randint(data_length-batchsize, size=1)
        indexes = np.arange(index_base, index_base+batchsize)
    for i_index in indexes:
        date_temp .append(float(date_data[i_index]))
        data_temp .append(float(data2[i_index])/float(normalFactor))
    date_samples = np.array(date_temp)
    data_samples = np.array(data_temp)
    date_samples = date_samples.reshape(batchsize, 1)
    data_samples = data_samples.reshape(batchsize, 1)
    return date_samples, data_samples


# 从总体数据集中载入部分数据作为训练集
def randSample_Normalize_3existData(date_data, data1, data2, batchsize=1, normalFactor=1000, sampling_opt=None):
    date_temp = []
    data1_temp = []
    data2_temp = []
    data_length = len(date_data)
    if str.lower(sampling_opt) == 'random_sample':
        indexes = np.random.randint(data_length, size=batchsize)
    elif str.lower(sampling_opt) == 'rand_sample_sort':
        indexes_temp = np.random.randint(data_length, size=batchsize)
        indexes = np.sort(indexes_temp)
    else:
        index_base = np.random.randint(data_length-batchsize, size=1)
        indexes = np.arange(index_base, index_base+batchsize)
    for i_index in indexes:
        date_temp .append(float(date_data[i_index]))
        data1_temp.append(float(data1[i_index]) / float(normalFactor))
        data2_temp .append(float(data2[i_index])/float(normalFactor))

    date_samples = np.array(date_temp)
    data1_samples = np.array(data1_temp)
    data2_samples = np.array(data2_temp)

    date_samples = date_samples.reshape(batchsize, 1)
    data1_samples = data1_samples.reshape(batchsize, 1)
    data2_samples = data2_samples.reshape(batchsize, 1)
    return date_samples, data1_samples, data2_samples


# 从总体数据集中载入部分数据作为训练集
def randSample_Normalize_5existData(date_data, data1, data2, data3, data4, batchsize=1, normalFactor=1000, sampling_opt=None):
    date_temp = []
    data1_temp = []
    data2_temp = []
    data3_temp = []
    data4_temp = []
    data_length = len(date_data)
    if str.lower(sampling_opt) == 'random_sample':
        indexes = np.random.randint(data_length, size=batchsize)
    elif str.lower(sampling_opt) == 'rand_sample_sort':
        indexes_temp = np.random.randint(data_length, size=batchsize)
        indexes = np.sort(indexes_temp)
    else:
        index_base = np.random.randint(data_length-batchsize, size=1)
        indexes = np.arange(index_base, index_base+batchsize)
    for i_index in indexes:
        date_temp .append(float(date_data[i_index]))
        data1_temp.append(float(data1[i_index]) / float(normalFactor))
        data2_temp.append(float(data2[i_index])/float(normalFactor))
        data3_temp.append(float(data3[i_index]) / float(normalFactor))
        data4_temp.append(float(data4[i_index]) / float(normalFactor))

    date_samples = np.array(date_temp)
    data1_samples = np.array(data1_temp)
    data2_samples = np.array(data2_temp)
    data3_samples = np.array(data3_temp)
    data4_samples = np.array(data4_temp)

    date_samples = date_samples.reshape(batchsize, 1)
    data1_samples = data1_samples.reshape(batchsize, 1)
    data2_samples = data2_samples.reshape(batchsize, 1)
    data3_samples = data3_samples.reshape(batchsize, 1)
    data4_samples = data4_samples.reshape(batchsize, 1)
    return date_samples, data1_samples, data2_samples, data3_samples, data4_samples


# 对于时间数据来说，验证模型的合理性，要用连续的时间数据验证
def sample_testDays_serially(test_date, batch_size):
    day_it = test_date[0:batch_size]
    day_it = np.reshape(day_it, newshape=(batch_size, 1))
    return day_it


# 对于时间数据来说，验证模型的合理性，要用连续的时间数据验证
def sample_testData_serially(test_data, batch_size, normalFactor=1000):
    data_it = test_data[0:batch_size]
    data_it = data_it.astype(np.float32)
    data_it = np.reshape(data_it, newshape=(batch_size, 1))
    data_it = data_it/float(normalFactor)
    return data_it
