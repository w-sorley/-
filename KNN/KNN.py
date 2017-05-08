#!/usr/bin/env python3
# -*- coding=utf-8 -*-

__Author__="Shouli_Wang"

'''
the machinelearning agrogrith KNN
'''



from numpy import *
import matplotlib
import matplotlib.pyplot as plt
import operator
from os import listdir

def creatDataSet():
    group=array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels

def classify0(inx,dataSet,labels,k):
    dataSetSize=dataSet.shape[0]                  #shape return(row, column)
    diffMat=tile(inx,(dataSetSize,1))-dataSet   #tile(inx,n) inx*n  tile(inx,(row,column))
                                                                      #return matrix which's elements is inx
    sqDiffMat=diffMat**2
    sqDistance=sqDiffMat.sum(axis=1)  # sum(axis=?) axis=None sum all the elements return
                                                              #a number,axis=0,sum every row of matrix return a vector
                                                              #axis=1,sum every column of matrix return a vector
    distance=sqDistance**0.5
    sortedDistIndicies=distance.argsort()      #将距离从小到大排序
    classCount={}
    for i in range(k):             #找到距离最近的前k个，并将对应的标签（label）计数
        voteIlabel=labels[sortedDistIndicies[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1   #若存在加1，否则初始为0

    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)    #将对距离最近的前k个距离对应标签加后排序
    return sortedClassCount[0][0]   #返回在前k个最近的数据样本中，对应标签数最多的标签
                                    #（类似投票，最相近的前k个样本具有投票权，每人的投票为自己对应的标签，得票最多的标签为结果）

def file2matrix(filename):                                    
    fr=open(filename)                               #分别对每行进行操作，每行的前3的元素赋给样本的数据，每行的最后一个元素赋给样本的标签，并一index对样本进行计数标记
    arrayOLines=fr.readlines()
    numberOfLines=len(arrayOLines)
    returnMat=zeros((numberOfLines,3))
    classLabelVector=[] 
    index=0
    for line in arrayOLines:
        line=line.strip()
        listFromLine=line.split('\t')
        returnMat[index,:]=listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index+=1
    return returnMat,classLabelVector

def autoNorm(dataSet):
    minVals=dataSet.min(0)
    maxVals=dataSet.max(0)
    ranges=maxVals-minVals
    normDataset=zeros(shape(dataSet))
    m=dataSet.shape[0]                                    #对数据进行规则化，（原数据-最小值）/（最大值-最小值）：将数据规则化为大小{0到1}的范围内
    normDataset=dataSet-tile(minVals,(m,1))
    normDataset=normDataset/tile(ranges,(m,1))
    return normDataset,ranges,minVals

def datingClassTest():
    hoRatio=0.10
    datingDataMat,datingDtalaLabels=file2matrix('datingTestSet2.txt')       #对算法的正确率进行检验，hoRatio控制验证样本的比例，如hoRatio=0.1,对前百分之10的样本进行预测
    normMat,ranges,minVals=autoNorm(datingDataMat)                        #将后面百分之90的样本作为比较集和，最后将预测的标签值与样本真实的标签值进行比较，若不相等错误率加1
    m=normMat.shape[0]
    numTestVecs=int(m*hoRatio)
    errorCount=0.0
    for i in range(numTestVecs):
        classifierResult=classify0(normMat[i,:],normMat[numTestVecs:m,:],datingDtalaLabels[numTestVecs:m],3)
        print('the classifier came back with:%d,the real answer is :%d'%(classifierResult,datingDtalaLabels[i]))
        if(classifierResult!=datingDtalaLabels[i]):
            errorCount+=1.0
    print("the total error rate is :%f"%(errorCount/float(numTestVecs)))

def classifyPerson():
    resultList=['not at all','in small does','in large does']                     #输入某人做各种事情的花费时间的占比，然后与已有的样本数据进行比较，得到预测结果
    percentTats=float(input("percentage of time spend playing video game?"))
    ffMiles=float(input("frequent filier miles earned per yea?"))
    iceCream=float(input("liters of ice cream consumed per year?"))
    datingDataMat,datingDtalaLabels=file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals=autoNorm(datingDataMat)
    inArr=array([ffMiles,percentTats,iceCream])
    classifierResult=classify0((inArr-minVals)/ranges,normMat,datingDtalaLabels,3)
    print("you will probably like this person:",resultList[classifierResult-1])


# handing writing number classifier

def img2vector(filename):
    returnVect=zeros((1,1024))
    fr=open(filename)                      #将32x32的单通道图像转换为向量（32x32的矩阵转换为一行的向量）
    for i in range(32):                       #先按行遍历所有行，后按列遍历一行的所有列，其中矩阵i行j列元素对应向量的第32×i+j个元素
        lineStr=fr.readline()
        for j in range(32):
            returnVect[0,32*i+j]=int(lineStr[j])
    return returnVect






def handwritingClassTest():
    hwLabels=[]
    trainingFileList=listdir('digits/trainingDigits')
    m=len(trainingFileList)
    trainingMat=zeros((m,1024))
    for i in range(m):                                                    #构建训练集，文件名为hwLabels_index.txt,将每个图片转换为一行向量作为矩阵的一行
        filenamestr=trainingFileList[i]
        filestr=filenamestr.split('.')[0]
        classNumStr=int(filestr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:]=img2vector('digits/trainingDigits/%s'%filenamestr)
    testFileList=listdir('digits/testDigits')
    errorCount=0.0
    mTest=len(testFileList)                            #读取每个测试图转换为向量，利用KNN计算与给个训练图的距离，预测它的标签，并与真实标签值比较，若不等错误率加1
    for i in range(mTest):
        filenamestr=testFileList[i]
        filestr=filenamestr.split('.')[0]
        classNumStr=int(filestr.split('_')[0])
        vectorUnderTest=img2vector('digits/testDigits/%s'%filenamestr)
        classifierResult=classify0(vectorUnderTest,trainingMat,hwLabels,3)
        print("the classifier come back with %d, the real answer is %d"%(classifierResult,classNumStr))
        if(classifierResult!=classNumStr): errorCount+=1.0
    print("\n the total number of error is : %d"%errorCount)
    print("\nthe total error rate is :%f"%(errorCount/float(mTest)))



handwritingClassTest()




# classifyPerson()




# datingClassTest()

# fig=plt.figure()
# ax=fig.add_subplot(111)
# ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15.0*array(datingDtalaLabels),15.0*array(datingDtalaLabels))
# plt.show()
