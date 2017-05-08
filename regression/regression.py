#!/usr/bin/env python3
# -*-coding:utf-8 -*-

__Author__="Shouli_Wang"

'''
the machine learning algrothm regression
'''

from numpy import *
import matplotlib.pyplot as plt


def loadDataSet(filename):
    numFeat=len(open(filename).readline().split('\t'))-1
    dataMat=[]
    labelMat=[]
    fr=open(filename)
    for line in fr.readlines():
        lineArr=[]
        currLine=line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(currLine[i]))
        dataMat.append(lineArr)
        labelMat.append(lineArr[-1])
    return dataMat,labelMat

                          #return the (xtx)^-1*(xt*y)
def standRegres(xArr,yArr):
    xMat=mat(xArr)
    yMat=mat(yArr).T
    xTx=xMat.T*xMat
    if linalg.det(xTx)==0.0:
        print('this matrix is singular can not do inverse')
        return
    ws=xTx.I*(xMat.T*yMat)
    return ws

def lwlr(testPoint,xArr,yArr,k=1.0):
             #init
    xMat=mat(xArr)
    yMat=mat(yArr).T
    m=shape(xMat)[0]
    weights=mat(eye((m)))
             #calculate the weight on every point use the distance to testpoint
    for j in range(m):
        diffMat=testPoint - xMat[j,:]
        weights[j,j]=exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx=xMat.T*(weights*xMat)
    if linalg.det(xTx)==0.0:
        print('this matrix is singular can not do inverse')
        return
    ws=xTx.I*(xMat.T*(weights*yMat))
    return testPoint*ws         #return the predict value y^= W*X

                    #tesy the algorithm  lwlr  jubujiaquanhuigui
def lwlrTest(testArr,xArr,yArr,k=1.0):
    m=shape(testArr)[0]
    yHat=zeros(m)
    for i in range(m):
        yHat[i]=lwlr(testArr[i],xArr,yArr,k)
    return yHat

def rssError(yArr,yHatArr):
    return ((yArr-yHatArr)**2).sum()

             #linghuigui :xTx=xTx+I*lam  make it is not singular
def ridgeRegres(xMat,yMat,lam=0.2):
    xTx=xMat.T*xMat
    denom=xTx+eye(shape(xMat)[1])*lam
    if linalg.det(xTx)==0.0:
        print('this matrix is singular can not do inverse')
        return
    ws=denom.I*(xMat.T*yMat)
    return ws

def ridgeTest(xArr,yArr):
    xMat=mat(xArr)
    yMat=mat(yArr).T
    yMean=mean(yMat,0)
               #regular the yMat and xMat   the yMat is different :yMat=yMat-Ymean
    yMat=yMat-yMean
    xMeans=mean(xMat,0)
    xVar=var(xMat,0)
    xMat=(xMat-xMeans)/xVar

    numTestPts=30
    wMat=zeros((numTestPts,shape(xMat)[1]))
            #use the different lam (xTx=xTx+I*lam) calculate the ws
    for i in range(numTestPts):
        ws=ridgeRegres(xMat,yMat,exp(i-10))
        wMat[i,:]=ws.T
    return wMat

def regularize(xMat):#regularize by columns
    inMat = xMat.copy()
    inMeans = mean(inMat,0)   #calc mean then subtract it off
    inVar = var(inMat,0)      #calc variance of Xi then divide by it
    inMat = (inMat - inMeans)/inVar
    return inMat


  #qianxiang zubu xianxing regress  :eps the update step distance
def stageWise(xArr,yArr,eps=0.01,numIt=100):
           #init and regularize the dataSet
    xMat=mat(xArr)
    yMat=mat(yArr).T
    yMean=mean(yMat,0)
    yMat=yMat-yMean
    xMat=regularize(xMat)
    m,n=shape(xMat)
    returnMat=zeros((numIt,n))
    ws=zeros((n,1))
    wsTest=ws.copy()
    wsMax=ws.copy()
           #iteration  on every feature
    for i in range(numIt):
        print(ws.T)
        lowestError=inf
        for j in range(n):
            for sign in [-1,1]:    #update two time on every feature add an sub(jianshao)
                wsTest=ws.copy()
                wsTest[j]+=eps*sign    #update the ws
                yTest=xMat*wsTest
                rssE=rssError(yMat.A,yTest.A)    #calculate the error with the new ws
                if rssE<lowestError:     #if the new error is lower than the lowest error, update the wsmax
                    lowestError=rssE     #and the lowest error
                    wsMax=wsTest
            ws=wsMax.copy()
            returnMat[i,:]=ws.T
    return returnMat


from time import sleep
import json
import urllib2
def searchForSet(retX,retY,setNum,yr,numPce,origPrc):
    sleep(10)
    myApiStr='get from code.google.com'
    searchURL='http://www.google.com/shopping/search/vl/public/productions?/key='
    pg=urllib2.urlopen(searchURL)
    retDict=json.loads(pg.read())
    for i in range(len(retDict['items'])):
        try:
            currItem=retDict['items'][i]
            if currItem['product']['condiction']=='new':
                newFlag=1
            else:
                newFlag=0
            listOfInv=currItem['product']['inventories']
            for item in listOfInv:
                sellinfPrice=item['price']
                if sellinfPrice>origPrc*0.5:
                    print('%d\t%d\t%d\t%f\t%f\t'%(yr,numPce,newFlag,origPrc,sellinfPrice))
                retX.append([yr,numPce,newFlag,origPrc])
                retY.append(sellinfPrice)
        except:
            print('problem with item %d'%i)

def setDataCollect(retX,retY):
    searchForSet(retX,retY,8288,2006,800,49.99)
    searchForSet(retX,retY,10030,2002,3096,269.99)
    searchForSet(retX,retY,10179,2007,5195,499.99)
    searchForSet(retX,retY,10181,2007,3428,199.99)
    searchForSet(retX,retY,10189,2008,5922,299.99)
    searchForSet(retX,retY,10196,2009,3263,249.99)


  #jiaocha yanzheng linghuigui
def crossValidation(xArr,yArr,numVal=10):
    #init the dateset
    m=len(yArr)
    indexList=range(m)
    errorMat=zeros((numVal,30))
    #build the trainDataSet and testDataSet
    for i in range(numVal):
        trainX=[]
        trainY=[]
        testX=[]
        testY=[]
        random.shuffle(indexList)
        for j in range(m):
            if j<m*0.9:
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
    #train the model and test and  regularize the variable
    wMat=ridgeTest(trainX,trainY)
    for k in range(30):
        matTestX=mat(testX)
        matTrainX=mat(trainX)
        meanTrain=mean(matTrainX,0)
        varTrain=var(matTrainX,0)
        matTestX=(matTestX-meanTrain)/varTrain
        yEst=matTestX*mat(wMat[k,:]).T+mean(trainY)
        errorMat[i,k]=rssError(yEst.T.A,array(testY))
    meanError=mean(errorMat,0)
    minMean=float(min(meanError))
    bestWeight=wMat(nonzero(meanError==minMean))
    xMat=mat(xArr)
    yMat=mat(yArr).T
    meanX=mean(xMat,0)
    varX=var(xMat,0)
    unReg=bestWeight/varX
    print('the best  model from Ridge regression is :\n ',unReg)
    print('with constant term :',-1*sum(multiply(meanX,unReg))+mean(yMat))






# xArr,yArr=loadDataSet('abalone.txt')
# stageWise(xArr,yArr,0.001,5000)
# xMat=mat(xArr)
# yMat=mat(yArr).T
# xMat=regularize(xMat)
# yM=mean(yMat,0)
# yMat=yMat-yM
# print(standRegres(xMat,yMat.T).T)



# abx,aby=loadDataSet('abalone.txt')
# ridgeWeight=ridgeTest(abx,aby)
# flg=plt.figure()
# ax=flg.add_subplot(111)
# ax.plot(ridgeWeight)
# plt.show()



# abx,aby=loadDataSet('abalone.txt')
# yHat01=lwlrTest(abx[0:99],abx[0:99],aby[0:99],0.1)
# yHat1=lwlrTest(abx[0:99],abx[0:99],aby[0:99],1.0)
# yHat10=lwlrTest(abx[0:99],abx[0:99],aby[0:99],10.0)
# print(rssError(aby[0:99],yHat01.T))
# print(rssError(aby[0:99],yHat1.T))
# print(rssError(aby[0:99],yHat10.T))


# yHat01=lwlrTest(abx[100:199],abx[0:99],aby[0:99],0.1)
# yHat1=lwlrTest(abx[100:199],abx[0:99],aby[0:99],1.0)
# yHat10=lwlrTest(abx[100:199],abx[0:99],aby[0:99],10.0)
# print(rssError(aby[100:199],yHat01.T))
# print(rssError(aby[100:199],yHat1.T))
# print(rssError(aby[100:199],yHat10.T))



# xArr,yArr=loadDataSet('ex0.txt')
# print(yArr[0])
# print(lwlr(xArr[0],xArr,yArr,1.0))
# print(lwlr(xArr[0],xArr,yArr,0.001))
# yHat=lwlrTest(xArr,xArr,yArr,0.003)
# xMat=mat(xArr)
# srtInd=xMat[:,1].argsort(0)
# xSort=xMat[srtInd][:,0,:]

# flg=plt.figure()
# ax=flg.add_subplot(111)
# ax.plot(xSort[:,1],yHat[srtInd])
# ax.scatter(xMat[:,1].flatten().A[0],mat(yArr).T.flatten().A[0],s=2,c='red')
# plt.show()




# ws=standRegres(xArr,yArr)
# xMat=mat(xArr)
# yMat=mat(yArr)
# yHat=xMat*ws
# print(corrcoef(yHat.T,yMat))

# flg=plt.figure()
# ax=flg.add_subplot(111)
# ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0])
# xCopy=xMat.copy()
# xCopy.sort(0)
# yHat=xCopy*ws
# ax.plot(xCopy[:,1],yHat)
# plt.show()
