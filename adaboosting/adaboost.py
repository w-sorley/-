#!/usr/bin/env python3.4
# -*- coding:utf-8 -*-

__Author__="Shouli_Wang"

'''
the  machine learning algrothm    adapting boosting
'''


from numpy import *
im                                         #load the dataset
def loadSimpData():
    dataMat=matrix([[1.,2.1],[2.,1.1],[1.3,1.],[1.,1.],[2.,1.]])
    classLabels=[1.0,1.0,-1.0,-1.0,1.0]
    return dataMat,classLabels
              # dimen chose the feature,threshlneq chose the bijiao fangshi
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
                  #init: build the return mat
    retArray=ones((shape(dataMatrix)[0],1))
    if(threshIneq=='lt'):
           # the filter if duiying wei is true make it as -1.0
        retArray[dataMatrix[:,dimen]<=threshVal]=-1.0
    else:
        retArray[dataMatrix[:,dimen]>threshVal]=-1.0
    return retArray

def buildStump(dataArr,classLabels,D):
              #init the dataMat and labelMat
    dataMatrix=mat(dataArr)
    labelMat=mat(classLabels).T
    m,n=shape(dataMatrix)
    numSteps=10.0
    bestStump={}
    bestClassEst=mat(zeros((m,1)))
    minError=inf
                 #the first for on the all feature
    for i in range(n):
        rangeMin=dataMatrix[i,:].min()
        rangeMax=dataMatrix[i,:].max()
        stepSize=(rangeMax-rangeMin)/numSteps
                                  #the second for on the feature value
        for j in range(-1,int(numSteps)+1):
            for inequal in ['lt','gt']:
                threshVal=(rangeMin+float(j)*stepSize)    #try the different threshold value
                predictedVals=stumpClassify(dataMatrix,i,threshVal,inequal)
                   #return the result of all the sample
                errArr=mat(ones((m,1)))
                errArr[predictedVals==labelMat]=0
                weightedError=D.T*errArr
                print('spilt:dim %d thresh %.2f  thresh ineqal: %s   the weighted error is :\
                                                     %.3f'%(i,threshVal,inequal,weightedError))

       #lookup the best way and store the things obout it
                if weightedError<minError:
                    minError=weightedError
                    bestClassEst=predictedVals.copy()
                    bestStump['dim']=1
                    bestStump['thresh']=threshVal
                    bestStump['ineq']=inequal
    return bestStump,minError,bestClassEst

def adaBoostTrainDS(dataArr,classLabels,numIt=40):
                    #init:
    weakClassArr=[]
    m=shape(dataArr)[0]
    D=mat(ones((m,1))/m)
    aggClassEst=mat(zeros((m,1)))

    for i in range(numIt):
        bestStump,error,classEst=buildStump(dataArr,classLabels,D)  #get the best single tree
        print('D:',D.T)
        alpha=float(0.5*log((1.0-error)/max(error,1e-16)))  #calculate the alpha use the gongshi
        bestStump['alpha']=alpha
        weakClassArr.append(bestStump)   #store the things about the best single tree
        print('classEst:',classEst.T)
                        #update the value of D
        expon=multiply(-1*alpha*mat(classLabels).T,classEst)
        D=multiply(D,exp(expon))
        D=D/D.sum()
                        #calculate the leiji error
        aggClassEst+=alpha*classEst
        print('aggClassEst:',aggClassEst.T)
        aggErrors=multiply(sign(aggClassEst)!=mat(classLabels).T,ones((m,1)))
        errorRate=aggErrors.sum()/m
        print('total error:',errorRate,'\n')
        if (errorRate==0.0):
            break
        return weakClassArr,aggClassEst
    # return weakClassArr
                                         #classify the dataset  use the many suo classify
def adaClassify(datToclass,classifierArr):
    dataMatrix=mat(datToclass)
    m=shape(dataMatrix)[0]
    aggClassEst=mat(zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst=stumpClassify(dataMatrix,classifierArr[i]['dim'],\
                                           classifierArr[i]['thresh'],classifierArr[i]['ineq'])
        aggClassEst+=classifierArr[i]['alpha']*classEst
        print(aggClassEst)
    return sign(aggClassEst)

def loadDataSet(fileName):
    numFeat=len(open(fileName).readline().split('\t'))
    dataMat=[]
    labelMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        lineArr=[]
        curLine=line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def plotROC(predStrengths,classLabels):
    import matplotlib.pyplot as plt
    cur=(1.0,1.0)
    ySum=0
    numPosClas=sum(array(classLabels)==1.0)
    yStep=1/float(numPosClas)
    xStep=1/float(len(classLabels)-numPosClas)
    sortedIndicies=predStrengths.argsort()
    fig=plt.figure()
    fig.clf()
    ax=plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index]==1.0:
            delX=0
            delY=yStep
        else:
            delX=xStep
            delY=0
            ySum+=cur[1]
        ax.plot([0,1],[0,1],'b--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve for Adaboost Horse Colic Detection System')
        ax.axis([0,1,0,1])
        plt.show()
        print('the Area Under the curve is :',ySum*xStep)


datmat,classlabels=loadSimpData()
D=mat(ones((5,1))/5)
buildStump(datmat,classlabels,D)

#dataArr,labelsArr=loadDataSet('horseColicTraining2.txt')
#classifierArray,aggClassEst=adaBoostTrainDS(dataArr,labelsArr,10)
#plotROC(aggClassEst.T,labelsArr)

# testArr,testLabelArr=loadDataSet('horseColicTest2.txt')
# prediction10=adaClassify(testArr,classifierArray)
# errArr=mat(ones((67,1)))
# print(errArr[prediction10!=mat(testLabelArr).T].sum()/67)
# D=mat(ones((5,1))/5)
# dataMat,classLabels=loadSimpData()
# print(buildStump(dataMat,classLabels,D))
# classifierArray=adaBoostTrainDS(dataMat,classLabels,30)
# print(adaClassify([0,0],classifierArray))
