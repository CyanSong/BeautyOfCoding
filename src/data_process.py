from gensim.models import Word2Vec
import json
import re
import jieba
import sys
import math


def getdata(filename,maxnum):
    f = open(filename, 'r', encoding='utf-8')
    r = f.readlines()
    tmpquestion = " "
    quecnt = -1
    quelist = []  # question as index, point to the number whose answer is related to this question
    answerlist = []  # question as index, point to the correct answer's number
    datalist = []  # datalist is the list of origin data, datalist[i][0] is the correctness of question-answer pair i. datalist[i][1]is the question ,data;ist[i][2] is the answer
    print('Loading data\n', 'expect', len(r), 'lines')
    for cnt, lines in enumerate(r):
        l = re.sub(r'\s+', ' ', lines)  # Replace the multy space with one space
        l = l.split(' ', 2)  # l contains 3 parts after this operation
        if l[0] != '0' and l[0] != '1':
            l[0] = '0'
        datalist.append(l)
        if l[1] != tmpquestion:
            quecnt = quecnt + 1
            tmpquestion = l[1]
            quelist.append([])
            answerlist.append([])
        if l[0] == '1':
            answerlist[quecnt] = cnt
        quelist[quecnt].append(cnt)
        if cnt + 1 ==maxnum:
            break
        progressbar(cnt, len(r))
    f.close()
    return quelist, answerlist, datalist

def progressbar(cur, total):
    percent = '{:.2%}'.format(cur / total)
    sys.stdout.write('\r')
    sys.stdout.write("[%-50s] %s" % (
        '=' * int(math.floor(cur * 50 / total)),
        percent))
    sys.stdout.flush()


def spiltword(datalist):
    wordlist1 = []
    wordlist2 = []
    print('\nSplitting word\n')
    for index, storage in enumerate(datalist):
        wordsplit1 = list(jieba.cut(storage[1], cut_all=False))
        wordsplit2 = list(jieba.cut(storage[2], cut_all=False))
        wordlist1.append(wordsplit1)
        wordlist2.append(wordsplit2)
        progressbar(index, len(datalist))
    return wordlist1, wordlist2


def turnwordtovector(wordlist1, wordlist2):
    print('\nLoading word2vec model...\n')
    model = Word2Vec.load('../tools/Word60.model')
    print('\nTurning word to vector\n')
    vectorlist1 = []
    vectorlist2 = []
    for index, data in enumerate(wordlist1):
        tmp = []
        for item in data:
            if item in model.vocab:
                tmp.append(model[item].tolist())
        vectorlist1.append(tmp)
        progressbar(index, len(wordlist1))

    for index, data in enumerate(wordlist2):
        tmp = []
        for item in data:
            if item in model.vocab:
                tmp.append(model[item].tolist())
        vectorlist2.append(tmp)
        progressbar(index, len(wordlist2))
    return vectorlist1, vectorlist2


def savetojson(datafile, jsonfile,num):
    f = open(jsonfile, "w", encoding='utf-8')
    quelist, answerlist, datalist = getdata(datafile,num)
    wordlist1, wordlist2 = spiltword(datalist)
    vectorlist1, vectorlist2 = turnwordtovector(wordlist1, wordlist2)
    python2json = {}
    python2json["quelist"] = quelist
    python2json["answerlist"] = answerlist
    python2json["datalist"] = datalist
    python2json["wordlist1"] = wordlist1
    python2json["wordlist2"] = wordlist2
    python2json["vectorlist1"] = vectorlist1
    python2json["vectorlist2"] = vectorlist2
    jsonstr = json.dumps(python2json)
    f.write(jsonstr)
    f.close()


def loadfromjson(filename):
    f = open(filename, 'r', encoding='utf-8')
    data = json.load(f)
    f.close()
    return data
