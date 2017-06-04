from data_process import *
import sys
import numpy as np
from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad, Adam
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Masking
from keras.layers.recurrent import LSTM, GRU
from keras.models import Model
from keras.losses import mean_squared_error
from keras.layers import Input, Dense, concatenate, normalization
from keras.regularizers import l1, l2
from keras import initializers
from keras.models import load_model
import os


def train(trainjson, epoch, batch_size, netq, neta, netfull, activate, drop_out, modelname, reg_flag, normal_flag,
          optim):
    data = loadfromjson(trainjson)
    # get label
    taglist = []
    for index, item in enumerate(data['datalist']):
        if item[0] == '0':
            taglist.append(0)
        else:
            if item[0] == '1':
                taglist.append(1)
            else:
                print('EiRROR\n')
                print(index)
                taglist.append(0)
    # get answer vectors and question vectors
    xq = np.zeros((len(data['vectorlist1']), netq[0], 60), dtype='float32')
    xa = np.zeros((len(data['vectorlist1']), neta[0], 60), dtype='float32')
    for index1, items in enumerate(data['vectorlist1']):
        for index2, item2 in enumerate(items):
            if index2 == netq[0]:
                break
            xq[index1][index2] = item2
    for index1, items in enumerate(data['vectorlist2']):
        for index2, item2 in enumerate(items):
            if index2 == neta[0]:
                break
            xa[index1][index2] = item2
    ya = np.array(taglist)
    trueya = []
    truexa = []
    truexq = []
    for index, label in enumerate(taglist):
        if label == 1:
            trueya.append(label)
            truexa.append(xa[index])
            truexq.append(xq[index])
    assert (len(trueya) != 0)
    print(len(truexq))
    truexa = np.repeat(truexa, 2, axis=0)
    truexq = np.repeat(truexq, 2, axis=0)
    trueya = np.repeat(trueya, 2, axis=0)
    ya = np.concatenate((ya, np.array(trueya)))
    # print(xa.shape,truexa.shape)
    xa = np.concatenate((xa, np.array(truexa)))
    xq = np.concatenate((xq, np.array(truexq)))

    print('Build model...')
    # regularizer param
    if reg_flag == 'None':
        reg = None
    else:
        reg_rate = float(reg_flag.split('_')[1])
        if reg_flag.split('_')[0] == 'l1':
            reg = l1(reg_rate)
        else:
            reg = l2(reg_rate)

    if not os.path.isfile(modelname):
        # seperate LSTM for qustion and answers
        question_vector_input = Input(shape=(netq[0], 60), dtype="float32", name='question_vector_input')
        question_vector_mask = Masking(mask_value=0.0)(question_vector_input)
        question_features = LSTM(output_dim=netq[1],
                                 kernel_initializer=initializers.truncated_normal(stddev=0.01))(question_vector_mask)
        answer_vector_input = Input(shape=(neta[0], 60), dtype="float32", name='answer_vector_input')
        answer_vector_mask = Masking(mask_value=0.0)(answer_vector_input)
        answer_features = LSTM(output_dim=neta[1],
                               kernel_initializer=initializers.truncated_normal(stddev=0.01))(answer_vector_mask)
        # merge two LSTMs
        question_features = normalization.BatchNormalization()(question_features)
        answer_features = normalization.BatchNormalization()(answer_features)
        features = concatenate([answer_features, question_features])
        # features = Activation(activate)(features)
        # full connected layer
        for dim in netfull:
            features = Dense(dim, kernel_regularizer=reg, use_bias=True,
                             kernel_initializer=initializers.truncated_normal(stddev=0.01))(features)
            # using normalization or not
            if (normal_flag == 'true'):
                features = normalization.BatchNormalization()(features)
            features = Activation(activate)(features)
        # drop out layer
        final_layer = Dropout(drop_out)(features)
        # sigmoid to 0-1
        main_output = Dense(1, activation='sigmoid', name='main_output')(final_layer)
        # finish model
        model = Model(inputs=[question_vector_input, answer_vector_input], outputs=[main_output])

        opt = optim.split('_')[0]

        if opt == 'rmsprop':
            opt = RMSprop((float)(optim.split('_')[1]))
        else:
            opt = Adam((float)(optim.split('_')[1]))

        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    else:
        print('load previous model')
        model = load_model(modelname)
    best_score = 0
    best_epoch = 0
    if not os.path.isfile('../json_data/quicktest.json'):
        savetojson('../raw_data/dev.txt', '../json_data/quicktest.json', 8000)
    test_q, test_a = getvalid(netq[0], neta[0], '../json_data/quicktest.json')
    quelist, answerlist, datalist = getdata('../raw_data/dev.txt', 8000)

    for i in range(epoch):
        model.fit([xq, xa], [ya], batch_size=batch_size, nb_epoch=1)  # 训练时间为若干个小时
        cur_score = valid(model, test_q, test_a, quelist, answerlist)
        print('In epoch', i + 1, 'MRR', cur_score)
        if cur_score > best_score:
            best_score = cur_score
            best_epoch = i + 1
        model.save(modelname[:-3] + str(epoch) + '.h5')
    print('best epoch', best_epoch, 'best MRR', best_score)
    f = open('../result/' + modelname[9:-3] + '.txt', 'w')
    f.write('best epoch' + str(best_epoch) + 'best MRR' + str(best_score))
    f.close()


def getvalid(q_in, a_in, testfile):
    data = loadfromjson(testfile)
    taglist = []
    for index, item in enumerate(data['datalist']):
        if item[0] == '0':
            taglist.append(0)
        else:
            if item[0] == '1':
                taglist.append(1)
            else:
                print('EiRROR\n')
                print(index)
                taglist.append(0)
    xq = np.zeros((len(data['vectorlist1']), q_in, 60), dtype='float32')
    xa = np.zeros((len(data['vectorlist1']), a_in, 60), dtype='float32')
    for index1, items in enumerate(data['vectorlist1']):
        for index2, item2 in enumerate(items):
            if index2 == q_in:
                break
            xq[index1][index2] = item2
    for index1, items in enumerate(data['vectorlist2']):
        for index2, item2 in enumerate(items):
            if index2 == a_in:
                break
            xa[index1][index2] = item2

    return xq, xa


def valid(model, xq, xa, quelist, answerlist):
    score = model.predict([xq, xa])
    sum = 0
    for cnt, question in enumerate(quelist):
        tmp = []
        for i in question:
            tmp.append(score[i])
        tmp.sort()
        if answerlist[cnt] == []:
            pass
        else:
            tmp.reverse()
            sum = sum + 1 / (1 + tmp.index(score[answerlist[cnt]]))
    return sum / len(quelist)


if __name__ == '__main__':
    raw_data_path = '../raw_data/'
    json_data_path = '../json_data/'
    model_path = '../model/'
    if len(sys.argv) == 13:
        if (sys.argv[1] == 'new'):
            savetojson(raw_data_path + 'train.txt', json_data_path + 'train.json', int(sys.argv[9]))
        epoch = int(sys.argv[2])
        batch_size = int(sys.argv[3])
        net_lstm_q = [int(x) for x in sys.argv[4].split('_')]
        net_lstm_a = [int(x) for x in sys.argv[5].split('_')]
        net_full = [int(x) for x in sys.argv[6].split('_')]
        activate = sys.argv[7]
        drop_out = float(sys.argv[8])
        modelname = model_path + 'model'
        for arg in sys.argv[2:]:
            modelname = modelname + '_' + arg
        modelname = modelname + '.h5'
        train(json_data_path + 'train.json', epoch, batch_size,
              net_lstm_q, net_lstm_a, net_full, activate, drop_out, modelname, sys.argv[10], sys.argv[11], sys.argv[12])
    else:
        print(
            'Usage: python train.py [jsonfile] [epoch]  [batch_size] [net_LSTM_question] [net_LSTM_answer] '
            '[net_full] [net_full activate] [drop_out] [train_dara_size] [regular] [batch normalization] [optimizer]')
        print('#jsonfile new/exist')
        print('#net_LSTM_question 16_64')
        print('#net_LSTM_answer 32_64')
        print('#net_full 128_128')
        print('#net_full_activate relu/sigmoid')
        print('#regular l2_0.01/l1_0.01/None')
        print('#batch normalization true/false')
        print('#optimizer adam_0.001/rmsprop_0.001')
        print('Usage example:python train.py exist 15 64 16_32 32_64 64 sigmoid 0.75 30000 l1_0.001 false adam_0.001')

    exit(0)
