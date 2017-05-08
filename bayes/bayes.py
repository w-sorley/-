#!/usr/bin/env python3
# -*-coding:utf-8-*-

__Author__="ShouLi_Wang"

'''
the  bates classifier
'''

from numpy import *
import re

def loadDataSet():
    postingList=[['my','dog','flea','problem','help','please'],
    ['maybe','not','take','him','to','dog','park','stupid'],
    ['my','dalmation','is','so','cute','i','love','him'],
    ['stop','posting','stupid','woorthless','garbage'],
    ['mr','licks','ate','my','steak','how','to','stop','him'],
    ['quit','buying','woorthless','dog','food','stupid']
    ]
    classVec=[0,1,0,1,0,1]
    return postingList,classVec

def createVocabList(dataSet):       #利用集合的并操作，由数据集创建词列表（集合），忽略出现的次数
    vocabSet=set([])
    for document in dataSet:
        vocabSet=vocabSet|set(document)
    return list(vocabSet)

def setOfWords2Vec(vocabList,inputSet):
        returnVec=[0]*len(vocabList)   #乘以符号×，返回n个相同元素的集合
        for word in inputSet:
            if word in vocabList:        #利用输入数据中的单词，在判别字典中查找，若包含则对应位置1
                returnVec[vocabList.index(word)]=1
            else:
                print("the word :%s is not in my vocabList"%word)
        return returnVec                  #返回的结果的长度等于判别字典的长度（所含的单词数）


def trainNB0(trainMatrix,trainCategory):
    numTrainDocs=len(trainMatrix)         #文档数
    numWords=len(trainMatrix[0])           #每篇文档的单词数
    pAbusive=sum(trainCategory)/float(numTrainDocs)        #probibity the labal is 1 (1D/sum(D))
    p0num=ones(numWords)
    p1num=ones(numWords)
    p0Denom=2.0
    p1Denom=2.0
    for i in range(numTrainDocs):
        if trainCategory[i]==1:
            p1num+=trainMatrix[i]                 # sum num of every words in label 1
            p1Denom+=sum(trainMatrix[i])    #sum the num of words of doc in vocabulary
        else:
            p0num+=trainMatrix[i]
            p0Denom+=sum(trainMatrix[i])
    p1Vec=p1num/p1Denom                    #every word / all the words in vocabulary
    p0Vec=p0num/p0Denom
    return log(p0Vec),log(p1Vec),pAbusive            #p(v|0)  p(v|1) p(1)

def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1=sum(vec2Classify*p1Vec)+log(pClass1)         #sum(P(v|1)*p(1))
    p0=sum(vec2Classify*p0Vec)+log(1.0-pClass1)
    if p1>p0:
        return 1
    else:
        return 0

def testingNB():
    listOpost,listClasses=loadDataSet()
    myvocablist=createVocabList(listOpost)
    trainmat=[]
    for postinDoc in listOpost:
        trainmat.append(setOfWords2Vec(myvocablist,postinDoc))
    p0V,p1V,pAb=trainNB0(array(trainmat),array(listClasses))
    testEntry=['love','my','dalmation']
    thisDoc=array(setOfWords2Vec(myvocablist,testEntry))
    print(testEntry,'classified as:',classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry=['stupid','garbage']
    thisDoc=array(setOfWords2Vec(myvocablist,testEntry))
    print(testEntry,'classified as:',classifyNB(thisDoc,p0V,p1V,pAb))

def bagOfWords2VecMN(vocabList,inputSet):
        returnVec=[0]*len(vocabList)
        for word in inputSet:
            if word in vocabList:
                returnVec[vocabList.index(word)]+=1
        return returnVec

def textParse(bigString):
    regEx=re.compile('\\W*')
    listOfTokes=regEx.split(bigString)
    return [tok.lower() for tok in listOfTokes if len(tok)>2]

def spamTest():
    docList=[]
    classList=[]
    fullText=[]
                #import the email data and  build the fullText ,docList
    for i in range(1,26):
        wordList=textParse(open('email/spam/%d.txt'%i).read())
        docList.append(wordList)   #append zhuijia
        fullText.extend(wordList)    #extend kuozhan
        classList.append(1)
        wordList=textParse(open('email/ham/%d.txt'%i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)

    vocabList=createVocabList(docList)
    trainingSet=range(50)
    testSet=[]
                  #suiji xuanze trainSet and TestSet
    for i in range(10):
        randIndex=int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(list(trainingSet)[randIndex])

         # train the classify model
    trainMat=[]
    trainClasses=[]
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam=trainNB0(array(trainMat),array(trainClasses))
    errorCount=0

              #test the model error rate
    for docIndex in testSet:
        wordVector=setOfWords2Vec(vocabList,docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam)!=classList[docIndex]:
            errorCount+=1
    print('the error rate is :',float(errorCount)/len(testSet))


#spamTest()
#spamTest()



# open('email/ham/23.txt').read()

# mySent='This book is the best book  on python or M.L .I have ever laid  eyes upon'
# print(listOfTokes)
# print(mySent.split())


# emailText=open('email/ham/6.txt').read()
# listOfTokens=regEx.split(emailText)






testingNB()

# listOpost,listClasses=loadDataSet()
# print(listOpost,'\n \n',listClasses,'\n \n')
# myvocablist=createVocabList(listOpost)
# print(myvocablist)

# trainmat=[]
# for postinDoc in listOpost:
#     trainmat.append(setOfWords2Vec(myvocablist,postinDoc))

# print(trainmat)

# p0v,p1v,pAb=trainNB0(trainmat,listClasses)
# print("p0v=",p0v,"\n","p1v=",p1v,"pAb=",pAb)

    # print(postinDoc)

# print(p0V,p1V,pAb)

# print(vec)


