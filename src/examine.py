import numpy as np
from data_process import *
import sys
from keras.layers.recurrent import LSTM, GRU
from keras.models import load_model


def calculateMRR(quelist, answerlist, datalist, scorefile):
    f = open(scorefile, 'r', encoding='utf-8')
    r = f.readlines()
    score = []
    for i in r:
        score.append(float(i))
    sum = 0
    print(len(r), len(quelist), len(datalist), len(answerlist))
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
    outfile = open(scorefile[0:-4] + '_result.txt', 'w')
    outfile.write(str(sum / len(quelist)))


def output(model_name, testfile, scorefile):
    model = load_model(model_name)
    param=model_name.split('_')
    q_in=int(param[3])
    a_in=int(param[5])
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

    result = model.predict([xq, xa])
    print(result)
    print(len(result))
    f = open(scorefile, 'w', encoding='utf-8')
    for i in result:
        f.write(str((i[0])))
        f.write('\n')
    f.close()


if __name__ == '__main__':
    json_data_path = '../json_data/'
    model_path = '../model/'
    result_path = '../result/'
    raw_data_path = '../raw_data/'
    if len(sys.argv) == 3:
        if (sys.argv[2] == 'new'):
            savetojson(raw_data_path + 'dev.txt', json_data_path + 'test.json',90000)
        modelname = model_path + sys.argv[1] + '.h5'
        json_data = json_data_path + 'test.json'
        output(modelname, json_data, result_path + sys.argv[1] + '_score.txt')
        quelist, answerlist, datalist = getdata(raw_data_path + 'dev.txt',90000)
        calculateMRR(quelist, answerlist, datalist, result_path + sys.argv[1] + '_score.txt')
    else:
        print(
            'Usage: python examine.py [model_name] [test_json]')
        print('#model_name using model name (see directory model)')
        print('#test_json new/exist')
    exit(0)
