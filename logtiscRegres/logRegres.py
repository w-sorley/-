#!/usr/bin/env python3
# -*-coding:utf-8 -*-

__Author__="Shouli_Wang"

'''
the machine learning algrothm  logistisc regres
'''

from numpy import *
import matplotlib.pyplot as plt

def loadDataSet():
    dataMat=[]
    labelMat=[]
    fr=open('testSet.txt')
    for line in fr.readlines():
        lineArr=line.strip().split()     #strip()  delete the blank space
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

def sigmoid(inx):
    return 1.0/(1+exp(-inx))

def gradAscet(dataMatIn,classLabels):
    dataMatrix=mat(dataMatIn)
    labelMat=mat(classLabels).transpose()
    m,n=shape(dataMatrix)
    alpha=0.001
    maxCycles=500
    weights=ones((n,1))
    for k in range(maxCycles): #iteration that update the weight use the w(i+1)=w(i)+al*data*weight
        h=sigmoid(dataMatrix*weights)
        error=(labelMat-h)
        weights=weights+alpha*dataMatrix.transpose()*error
    return weights

def plotBestFit(wei):
    # weights=wei.getA()
    dataMat,labelMat=loadDataSet()
    dataArr=array(dataMat)
    n=shape(dataArr)[0]
    xcode1=[]
    ycode1=[]
    xcode2=[]
    ycode2=[]
    for i in range(n):
        if(int(labelMat[i]))==1:
            xcode1.append(dataArr[i,1])
            ycode1.append(dataArr[i,2])
        else:
            xcode2.append(dataArr[i,1])
            ycode2.append(dataArr[i,2])
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcode1,ycode1,s=30,c='red',marker='s')
    ax.scatter(xcode2,ycode2,s=30,c='green')
    x=arange(-3.0,3.0,0.1)
    y=(-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

def stocGradAscent0(dataMatrix,classLabels):
    m,n=shape(dataMatrix)
    alpha=0.01
    weights=ones(n)
    for i in range(m):
        h=sigmoid(sum(dataMatrix[i]*weights))
        error=classLabels[i]-h
        print(error)
        print(dataMatrix[i])               #sui ji tidu shangsheng  ;accroding to every sample to update
        weights=weights+alpha*error*dataMatrix[i]
    return weights



def stocGradAscent1(dataMatrix,classLabels,numIter=150):
    m,n=shape(dataMatrix)
    weights=ones(n)
    for j in range(numIter):
        dataIndex=range(m)
        for i in range(m):                          #gaijin ban suiji tidu shangsheng ;random chose the alpha
            alpha=4/(1.0+i+j)+0.01
            randIndex=int(random.uniform(0,len(dataIndex)))
            h=sigmoid(sum(dataMatrix[i]*weights))
            error=classLabels[i]-h
            weights=weights+alpha*error*dataMatrix[i]
            del(list(dataIndex)[randIndex])
    return weights

def classifyVector(inX,weights):
    prob=sigmoid(sum(inX*weights))
    if (prob>0.5):
        return 1.0
    else:
        return 0.0

def colicTest():
                            # load the dateSet
    frTrain=open('horseColicTraining.txt')
    frTest=open('horseColicTest.txt')
    trainingSet=[]
    trainingLabels=[]
           #init the trainSet and train the model
    for line in frTrain.readlines():
        currLine=line.strip().split('\t')
        lineArr=[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights=stocGradAscent1(array(trainingSet),trainingLabels,500)

                             #test the  TrainModel  use TestSet
    errorCount=0
    numTestVec=0
    for line in frTest.readlines():
        numTestVec+=1.0
        currLine=line.strip().split('\t')
        lineArr=[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr),trainWeights))!=int(currLine[21]):
            errorCount+=1
    errorRate=(float(errorCount)/numTestVec)
    print("the error rate of this test is :%f "%errorRate)
    return errorRate

def multiTest():
    numTests=10
    errorSum=0.0
    for k in range(numTests):
        errorSum+=colicTest()
    print("after %d iteration the average error rate is:%f"%(numTests,errorSum/float(numTests)))





# dataArr,labelMat=loadDataSet()
# weights=gradAscet(dataArr,labelMat)
# weights=stocGradAscent1(array(dataArr),labelMat)
# plotBestFit(weights)
multiTest()
