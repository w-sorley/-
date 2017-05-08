#!/usr/bin/env python3.4
# -*-coding:utf-8 -*-

from numpy import *

   #导入数据集
def loadDataSet(filename):
    dataSet=[]
    fr=open(filename)
    for line in fr.readlines():
        currLine=line.strip().split('\t')
        fltLine=list(map(float,currLine))
        dataSet.append(fltLine)
    return dataSet

#给定某个特征和相对应的特征值将数据集进行二分类
def binSplitDataSet(dataSet,feature,value):
    mat0=dataSet[nonzero(dataSet[:,feature]>value)[0],:]#[0]  #nonzero()返回非零元素的索引，array（i，。。。）array（j，。。）第i行的第j个元素非0
    mat1=dataSet[nonzero(dataSet[:,feature]<=value)[0],:]#[0]
    return mat0,mat1

#当满足停车继续对子树进行划分的条件时，利用regleaf对叶节点进行建模
def regLeaf(dataSet):
    return mean(dataSet[:,-1])

def regErr(dataSet):
    return var(dataSet[:,-1])*shape(dataSet)[0]  #var（）均方差函数


#创建树结构，leaftype：建立叶节点的函数，errtype：误差计算函数，ops：包含树构建的其他参数
def createTree(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    feat,val=chooseBestSplit(dataSet,leafType,errType,ops)  #选择当前划分后最好的特征和对应特征值（尽量使划分后的熵最大）
    if feat==None:    #当满足停止条件时，特征选择函数返回None和其他参数值（常数或线性方程）
        return val
    retTree={}
    retTree['spInd']=feat
    retTree['spVal']=val
    lSet,rSet=binSplitDataSet(dataSet,feat,val)
    retTree['left']=createTree(lSet,leafType,errType,ops)      #而后对左子树和右子树进行递归划分
    retTree['right']=createTree(rSet,leafType,errType,ops)
    return retTree


#给定数据集，选择某个特征和特征值使划分后的树最好（尽量使划分后的树的熵最大）
def chooseBestSplit(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    tolS=ops[0]
    tolN=ops[1]
    if len(set(dataSet[:,-1].T.tolist()[0]))==1:   #set进行去重，当所有样本的类别相同时，停止迭代，即停止继续划分
        return None, leafType(dataSet)
    m,n=shape(dataSet)
    s=errType(dataSet)
    bestS=inf
    bestIndex=0
    bestValue=0
    #两层for循环，没别对每个特征及每个特征的所有特征值进行遍历
    for featIndex in range(n-1):
        for splitVal in dataSet[:,featIndex]:
            mat0,mat1=binSplitDataSet(dataSet,featIndex,splitVal)
            if (shape(mat0)[0]<tolN) or (shape(mat1)[0]<tolN):
                continue
            newS=errType(mat0)+errType(mat1)
            if newS<bestS:               #计算最小误差，并记录最小误差对应的特征以及特征直
                bestIndex=featIndex
                bestValue=splitVal
                bestS=newS
    if (s-bestS)<tolS:   #tolS记录最小误差，当达到最小误差时停止迭代
        return None ,leafType(dataSet)
    mat0,mat1=binSplitDataSet(dataSet,bestIndex,bestValue)
    if (shape(mat0)[0]<tolN) or (shape(mat1)[0]<tolN):   #tolN记录子树的最小样本数，当低于最小样本数时停止迭代
        return None,leafType(dataSet)
    return bestIndex,bestValue



########################################对树进行剪枝操作##############################################33

#判断是否为树类型
def isTree(obj):
    return (type(obj).__name__=='dict')

#对树进行从上到下的遍历，直到叶子节点，返回两者的平均值，以便对树进行塌陷操作
def getMean(tree):
    if isTree(tree['right']):
        tree['right']=getMean(tree['right'])
    if isTree(tree['left']):
        tree['left']=getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0

def prune(tree,testData):
    if shape(testData)[0]==0:
        return getMean(tree)
    if(isTree(tree['right']) or isTree(tree['left'])):   #如果左或右节点为子树类型，利用当前的训练时使用的划分的特征和特征值对测试数据集进行划分
        lSet,rSet=binSplitDataSet(testData,tree['spInd'],tree['spVal'])
    if isTree(tree['left']):
        tree['left']=prune(tree['left'],lSet)  #如果左节点或友节点为子树类型，利用递归继续对左或右子树进行塌陷处理
    if isTree(tree['right']):
        tree['right']=prune(tree['right'],rSet)
    if not isTree(tree['left']) and not isTree(tree['right']):  #只有当遇到左右节点均为叶子节点时，根据合并前后的误差，判断是否对左右叶子节点是否进行合并操作
        lSet,rSet=binSplitDataSet(testData,tree['spInd'],tree['spVal'])
        errorNoMerge=sum(power(lSet[:,-1]-tree['left'],2))+sum(power(rSet[:,-1]-tree['right'],2))
        treeMean=(tree['left']+tree['right'])/2.0
        errorMerge=sum(power(testData[:,-1]-treeMean,2))
        if errorMerge<errorNoMerge:
            print('merging')
            return treeMean
        else:
            return tree
    else:
        return tree


#对数据进行格式化
def linearSolve(dataSet):
    m,n=shape(dataSet)
    X=mat(ones((m,n)))
    Y=mat(ones((m,1)))
    X[:,1:n]=dataSet[:,0:n-1]   #X为样本特征值矩阵（在原矩阵上增加一列），Y为样本label矩阵
    Y=dataSet[:,-1]
    xTx=X.T*X
    if linalg.det(xTx)==0.0:   #判断X是否为奇异矩阵，逆是否存在
        raise NameError('This matrix is singular ,can not do inverse,\n\
                                             try increasing the second value of ops')
    ws=xTx.I*(X.T*Y)  #？？matrix.I返回逆矩阵
    return ws,X,Y

#负责生成叶子节点的线性模型
def modelLeaf(dataSet):
    ws,X,Y=linearSolve(dataSet)
    return ws

#计算模型误差
def modelErr(dataSet):
    ws,X,Y=linearSolve(dataSet)
    yHat=X*ws
    return sum(power(Y-yHat,2))

#若叶子节点为回归书节点，进行预测返回模型浮点数
def regTreeEval(model,inDat):
    return float(model)

#若叶子节点为现行模型树，根据线性回归返回线性模型预测值
def modelTreeEval(model,inDat):
    n=shape(inDat)[1]
    X=mat(ones((1,n+1)))
    X[:,1:n+1]=inDat
    return float(X*model)

#根据训练好的树模型对数据进行预测
def treeForeCast(tree,inDat,modelEval=regTreeEval):
    if not isTree(tree):    #如果遇到遇到叶子节点，根据节点类型（回归树，或线性模型）返回预测值
        return modelEval(tree,inDat)
    if inDat[tree['spInd']]>tree['spVal']:   #根据分割特征和对应模型原特征值与对应的输入数据的特征值判断属于左或右节点
        if isTree(tree['left']):
            return treeForeCast(tree['left'],inDat,modelEval)      #若左或右节点为树节点继续划分
        else:
            return modelEval(tree['left'],inDat)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'],inDat,modelEval)
        else:
            return modelEval(tree['right'],inDat)

#对测试集的每条样本数据进行预测，返回预测值的向量
def createForeCast(tree,testData,modelEval=regTreeEval):
    m=len(testData)
    yHat=mat(zeros((m,1)))
    for i in range(m):
        yHat[i,0]=treeForeCast(tree,mat(testData[i]),modelEval)
    return yHat


# trainMat=mat(loadDataSet('bikeSpeedVsIq_train.txt'))
# testMat=mat(loadDataSet('bikeSpeedVsIq_test.txt'))
# myTree=createTree(trainMat,ops=(1,20))
# yHat=createForeCast(myTree,testMat[:,0])
# a=corrcoef(yHat,testMat[:,1],rowvar=0)
# print(a[0])

# myTree2=createTree(trainMat,modelLeaf,modelErr,(1,20))
# yHat=createForeCast(myTree,testMat[:,0])
# b=corrcoef(yHat,testMat[:,1],rowvar=0)
# print(b[0])

# ws,X,Y=linearSolve(trainMat)
# print(ws)
# for i in range(shape(testMat)[0]):
#     yHat[i]=testMat[i,0]*ws[1,0]+ws[0,0]
# c=corrcoef(yHat,testMat[:,1],rowvar=0)
# print(c[0])
# testMat=mat(eye(4))
# mat0,mat1=binSplitDataSet(testMat,1,0.5)
# print(mat0)
# print(mat1)

# myData=loadDataSet('exp2.txt')
# myMat=mat(myData)
# tree=createTree(myMat,modelLeaf,modelErr,(1,10))
# print(tree)

# myDataTest=loadDataSet('ex2test.txt')
# myMatTest=mat(myDataTest)
# tree2=prune(tree,myMatTest)
# print(tree2)
