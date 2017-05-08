#!/usr/bin/env python3
# -*- coding=utf-8 -*-

__Author__='Shouli_Wang'

'''
the machine learning algorithm  dicide trees
'''

from math import log
import operator
import treePlotter

def createDataSet():
    dataSet=[[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
    labels=['no surfacing','flippers']
    return dataSet,labels


def calcShannonEnt(dataSet):
    numEntries=len(dataSet)
    labelCounts={}
    for featVec in dataSet:                                 #遍历数据集，对每个标签对应的数据样本的个数进行计数，
        currentLabel=featVec[-1]                              #标签对应的样本个数/总的样本个数=取本标签的概率
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=0
        labelCounts[currentLabel]+=1
    shannonEnt=0.0
    for key in labelCounts:
        prob=float(labelCounts[key])/numEntries
        shannonEnt-=prob*log(prob,2)                   #熵的计算公式：shannonEnt=sum(-概率×以2为底概率的对数)
    return shannonEnt                                  #样本的熵对应样本所包含信息量的大小

def spiltDataSet(dataSet,axis,value):
    retDataSet=[]
    for featVec in dataSet:                             #份割依据样本的第axis个特征是否为value的值
        if featVec[axis]==value:                          #返回的分割子集包含所有第axis个特征为value的样本，且样本与原来相比，去掉第axis个特征
            reduceFeatVec=featVec[:axis]
            reduceFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reduceFeatVec)
    return retDataSet


def chooseBestFeatureToSpilt(dataSet):
    numFeature=len(dataSet[0])-1                    #特征的数目=每个样本的长度减掉一个标签
    baseEntropy=calcShannonEnt(dataSet)              #将原来待分割样本的熵，作为初始的最好分割后获得的信息量
    bestInfoGain=0.0;bestfeature=-1
    for i  in range(numFeature):
        featList=[example[i] for example in dataSet]
        uniqueVals=set(featList)
        newEntropy=0.0
        for value in uniqueVals:
            subDataSet=spiltDataSet(dataSet,i,value)   #按照第i个特征是否为value进行分类
            prob=len(subDataSet)/float(len(dataSet))
            newEntropy+=prob*calcShannonEnt(subDataSet)     #遍历所有样本第i个特征包含的无重复所有特征值value，并分别按期分类，
                                                       #得到的newEntropy为：prop=分割后子集的样本数/总的样本数，newEntropy=sum(prob*子集的熵)
        infoGain=baseEntropy-newEntropy              #原来的待分割数据集的熵减掉newEntropy为此次分割的可获得信息量？？？？？？？？？？？？？？？？？？？？？？？？？？？？
        if(infoGain>bestInfoGain):
            bestInfoGain=infoGain                #分别选举第i个特征作为分割特征，选择出信息获得量最大的特征对应的序号i
            bestfeature=i
    return bestfeature

def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classList.keys():
            classList[vote]=0;
        classList[vote]+=1                                #对标签列表内的标签进行计数，返回数量最大的标签
    sortedClassCount=sorted(classCount.iteritems())
    key=operator.itemgetter(1,reverse=True)
    return sortedClassCount[0][0]                                  # 为什么返回[0][0]而不是[0],sort()返回的为向量还是矩阵？？？？？？？？？？？     ？？

def createTree(dataSet,labels):
    classList=[example[-1] for example in dataSet]                  #统计所有class
    if classList.count(classList[0])==len(classList):     #若所有样本对应class相同，则返回class结束
        return classList[0]
    if len(dataSet[0])==1:                               #若样本仅包含class，不含其他特征，则统计所有样本出现次数最多的一个class返回
        return majorityCnt(classList)
    bestFeat=chooseBestFeatureToSpilt(dataSet)
    bestFeatLabel=labels[bestFeat]            #选择最好分割的特征序号，并得到对应的label （label的分布？？？）
    myTree={bestFeatLabel:{}}
    del(labels[bestFeat])               #del()  用法
    featValues=[example[bestFeat] for example in dataSet]          #获得所有样本的第i个特征值
    uniqueVals=set(featValues)
    for value in uniqueVals:
        subLabels=labels[:]
        myTree[bestFeatLabel][value]=createTree(spiltDataSet(dataSet,bestFeat,value),subLabels)   #遍历所有第i个所有不重复特征值，按其分类，{best：{value1:{},value2:{}......}}
    return myTree

def classify(inputTree,featLabels,textVec):
    firstStr=list(inputTree.keys())[0]i                 #获取跟节点的名称
    secondDict=inputTree[firstStr]                          #获取下级节点
    featIndex=featLabels.index(firstStr)
    for key in secondDict.keys():
        if textVec[featIndex]==key:
            if type(secondDict[key]).__name__=='dict':
                classLabel=classify(secondDict[key],featLabels,textVec)     #如果下级节点为字典节点则继续递归分类，如果下级节点为叶子节点则返回标签分类结果
            else:
                classLabel=secondDict[key]
    return classLabel

def storeTree(inputTree,filename):
    import pickle
    fw=open(filename,'w')
    pickle.dump(inputTree,fw)            #pickle moudle for serializing and de-serializing a Python object structure
    fw.close()

def grabTree(filename):
    import pickle
    fr=open(filename)
    return pickle.load(fr)

fr=open('lenses.txt')
lenses=[inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels=['age','prescript','astigmatic','tearRate']
lensesTree=createTree(lenses,lensesLabels)
print(lensesTree)
treePlotter.createPlot(lensesTree)


# dataSet,labels=createDataSet()
# print(labels)
# myTree=treePlotter.retrieveTree(0)
# print(myTree)
# print(classify(myTree,labels,[1,0]))
# print(classify(myTree,labels,[1,1]))

# storeTree(myTree,'classifierStorage.txt')
# print(grabTree('classifierStorage.txt'))


# retDataSet1=spiltDataSet(dataSet,0,1)
# print(retDataSet1)
# retDataSet2=spiltDataSet(dataSet,0,0)
# print(retDataSet2)

# shannonEnt=calcShannonEnt(dataSet)
# print(shannonEnt)

# print(chooseBestFeatureToSpilt(dataSet))
# myTree=createTree(dataSet,labels)
# print(myTree)


