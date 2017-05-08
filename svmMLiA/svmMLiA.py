#!/usr/bin/env python3
# -*-coding:utf-8 -*-

__Author__="Shouli_Wang"

'''
the machine learning  algorithm  SVM(SMO)
'''
from numpy import *

def loadDataSet(filename):
    dataSet=[]
    labelMat=[]
    fr=open(filename)
    for line in fr.readlines():
        lineArr=line.strip().split()
        dataSet.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataSet,labelMat

def selectJrand(i,m):     #random update the alpha
    j=i
    while (j==i):
        j=int(random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):      #ensure the input btweem H and  L
    if aj<H:
        aj=H
    if L>aj:
        aj=L
    return aj

def smoSimple(dataSetIn,classLabels,C,toler,maxIter):
                    #load the dataSet  and init the varible
    dataMatrix=mat(dataSetIn)
    labelMat=mat(classLabels).transpose()
    b=0
    m,n=shape(dataMatrix)
    alphas=mat(zeros((m,1)))
    iter=0
        # iteration  until maxiter
    while (iter<maxIter):
        alphaPairsChanged=0     #the flag if the alpha is youhua
                        #iteration on every data
        for i in range(m):
            fXi=float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T))+b  #predict value
            Ei=fXi-float(labelMat[i])         #the error of the predict value
            if ((labelMat[i]*Ei<-toler)and(alphas[i]<C))or((labelMat[i]*Ei>toler)and(alphas[i]>0)):
                j=selectJrand(i,m)             #if the error is not in kejieshou fanwei ,then jinxing youhua
                fXj=float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T))+b
                Ej=fXj-float(labelMat[j])
                alphaIold=alphas[i].copy()
                alphaJold=alphas[j].copy()
                if(labelMat[i]!=labelMat[j]):
                    L=max(0,alphas[j]-alphas[i])
                    H=min(C,C+alphas[j]-alphas[i])
                else:
                    L=max(0,alphas[j]+alphas[i]-C)
                    H=min(C,alphas[i]+alphas[j])
                if(L==H):
                    print("L==H")
                    continue                        #every youhua de gaijin value,if it enough small ,the shoulian
                eta=2.0*dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*\
                                                    dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T
                if eta>0:
                    print("eta>0 ")
                    continue
                alphas[j]-=labelMat[j]*(Ei-Ej)/eta
                alphas[j]=clipAlpha(alphas[j],H,L)
                if(abs(alphas[j]-alphaJold)<0.00001):
                    print("j not moving enough")
                    continue
                alphas[i]+=labelMat[j]*labelMat[j]*(alphaJold-alphas[j])
                b1=b-Ei-labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*\
                                      dataMatrix[i,:].T-labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                b2=b-Ej-labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*\
                                      dataMatrix[j,:].T-labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if(0<alphas[i])and(C>alphas[i]):
                    b=b1
                elif(0<alphas[j])and(C>alphas[j]):
                    b=b2
                else:
                    b=(b1+b2)/2.0
                alphaPairsChanged+=1
                print("iter: %d i : %d ,pairs changed %d"%(iter,i,alphaPairsChanged))
        if(alphaPairsChanged==0):
            iter+=1
        else:
            iter=0
        print("iteration number : %d "%iter)
    return b,alphas


#









class optStruct:
    def __init__(self,dataMatIn,classLabels,C,toler):
        self.X=dataMatIn
        self.labelMat=classLabels
        self.C=C
        self.tol=toler
        self.m=shape(dataMatIn)[0]
        self.alphas=mat(zeros((self.m,1)))
        self.b=0
        self.eCache=mat(zeros((self.m,2)))

def calcEk(oS,k):
    fXk=float(multiply(oS.alphas,oS.labelMat).T*(oS.X*oS.X[k,:].T))+oS.b
    Ek=fXk-float(oS.labelMat[k])
    return Ek

def selectJ(i,oS,Ei):
    maxK=-1
    maxDeltaE=0
    Ej=0
    oS.eCache[i]=[1,Ei]
    validEcacheList=nonzero(oS.eCache[:,0].A)[0]
    if(len(validEcacheList)>1):
        for k in validEcacheList:
            if (k==i):
                continue
            Ek=calcEk(oS,k)
            daltaE=abs(Ei-Ek)
            if(daltaE>maxDeltaE):
                maxK=k
                maxDeltaE=daltaE
                Ej=Ek
        return maxK,Ej
    else:
        j=selectJrand(i,oS.m)
        Ej=calcEk(oS,j)
    return j,Ej

def updateEk(oS,k):
    Ek=calcEk(oS,k)
    oS.eCache[k]=[1,Ek]


def innerL(i,oS):
    Ei=calcEk(oS,i)
    if((oS.labelMat[i]*Ei<-oS.tol)and(oS.alphas[i]<oS.C))or((oS.labelMat[i]*Ei>oS.tol)and(oS.alphas[i]>0)):
        j,Ej=selectJ(i,oS,Ei)
        alphaIold=oS.alphas[i].copy()
        alphaJold=oS.alphas[j].copy()
        if(oS.labelMat[i]!=oS.labelMat[j]):
            L=max(0,oS.alphas[i]-oS.alphas[j])
            H=min(oS.C,oS.C+oS.alphas[j]-oS.alphas[j])
        else:
            L=max(0,oS.alphas[j]+oS.alphas[i]-oS.C)
            H=min(oS.C,oS.alphas[j]+oS.alphas[i])
        if(L==H):
            print("L+H")
            return 0
        eta=2.0*oS.X[i,:]*oS.X[j,:].T-oS.X[i,:]*oS.X[i,:].T-oS.X[j,:]*oS.X[j,:].T
        if(eta>0):
            print("eta>0")
            return 0
        oS.alphas[j]=oS.labelMat[j]*(Ei-Ej)/eta
        oS.alphas[j]=clipAlpha(oS.alphas[j],H,L)
        updateEk(oS,j)
        if(abs(oS.alphas[j]-oS.alphas[i])>0.00001):
            print("j not  moving enough")
            return 0
        oS.alphas[i]+=oS.alphas[j]*oS.alphas[i]*(alphaJold-oS.alphas[j])
        updateEk(oS,i)
        b1=oS.b-Ei-oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[i,:].T-\
                                   oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[i,:]*oS.X[j,:].T
        b2=oS.b-Ej-oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[j,:].T-\
                                  oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[j,:]*oS.X[j,:].T
        if(0<oS.alphas[i])and(oS.C>oS.alphas[i]):
            oS.b=b1
        elif(0<oS.alphas[j])and(oS.C<oS.alphas[j]):
            oS.b=b2
        else:
            oS.b=(b1+b2)/2.0
        return 1
    else:
        return 0

def smoP(dataMatIn,classLabels,C,toler,maxIter,kTup=('lin',0)):
    oS=optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler)
    iter=0
    entireSet=True
    alphaPairsChanged=0
    while (iter<maxIter)and((alphaPairsChanged>0)or(entireSet)):
        alphaPairsChanged=0
        if(entireSet):
            for i in range(oS.m):
                alphaPairsChanged+=innerL(i,oS)
            print("fullset ,iter:%d i :%d ,pairs changed: %d"%(iter,i,alphaPairsChanged))
            iter+=1
        else:
            nonBoundIs=nonzero((oS.alphas.A>0)*(oS.alphas.A<C))[0]
            for i in nonBoundIs:
                alphaPairsChanged+=innerL(i,oS)
            print("non-bound, iter :%d i: %d,pairs: changed:%d "%(iter,i,alphaPairsChanged))
            iter+=1
        if entireSet:
            entireSet=False
        elif(alphaPairsChanged==0):
            entireSet=True
        print("iteration number : %d"%iter)
    return oS.b,oS.alphas

def calcWs(alphas,dataArr,classLabels):
    X=mat(dataArr)
    labelMat=mat(classLabels).transpose()
    m,n=shape(X)
    w=zeros((n,1))
    for i in range(m):
        w+=multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w

def kernelTrans(X,A,kTup):
    m,n=shape(X)
    K=mat(zeros((m,1)))
    if(kTup[0]=='Lin'):
        K=X*A.T
    elif kTup[0]=='rbf':
        for j in range(m):
            deltaRow=X[j,:]-A
            K[j]=deltaRow*deltaRow.T
        K=exp(K/(-1*kTup[1]**2))
    else: raise NameError('Houston We have a Problem--  that kernel is not recognized ')
    return K


class optStruct2:
    def __init__(self,dataMatIn,classLabels,C,toler,kTup):
        self.X=dataMatIn
        self.labelMat=classLabels
        self.C=C
        self.tol=toler
        self.m=shape(dataMatIn)[0]
        self.alphas=mat(zeros((self.m,1)))
        self.b=0
        self.eCache=mat(zeros((self.m,2)))
        self.K=mat(zeros((self.m,self.m)))
        for i in range(self.m):
            self.k[:,i]=kernelTrans(self.X,self.X[i,:],kTup)

def innerL2(i,oS):
    Ei=calcEk(oS,i)
    if((oS.labelMat[i]*Ei<-oS.tol)and(oS.alphas[i]<oS.C))or((oS.labelMat[i]*Ei>oS.tol)and(oS.alphas[i]>0)):
        j,Ej=selectJ(i,oS,Ei)
        alphaIold=oS.alphas[i].copy()
        alphaJold=oS.alphas[j].copy()
        if(oS.labelMat[i]!=oS.labelMat[j]):
            L=max(0,oS.alphas[i]-oS.alphas[j])
            H=min(oS.C,oS.C+oS.alphas[j]-oS.alphas[j])
        else:
            L=max(0,oS.alphas[j]+oS.alphas[i]-oS.C)
            H=min(oS.C,oS.alphas[j]+oS.alphas[i])
        if(L==H):
            print("L+H")
            return 0
        eta=2.0*oS.K[i,j]-oS.K[i,i]-oS.K[j,j]
        if(eta>0):
            print("eta>0")
            return 0
        oS.alphas[j]=oS.labelMat[j]*(Ei-Ej)/eta
        oS.alphas[j]=clipAlpha(oS.alphas[j],H,L)
        updateEk(oS,j)
        if(abs(oS.alphas[j]-oS.alphas[i])>0.00001):
            print("j not  moving enough")
            return 0
        oS.alphas[i]+=oS.alphas[j]*oS.alphas[i]*(alphaJold-oS.alphas[j])
        updateEk(oS,i)
        b1=oS.b-Ei-oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i]-\
                                   oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
        b2=oS.b-Ej-oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j]-\
                                  oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
        if(0<oS.alphas[i])and(oS.C>oS.alphas[i]):
            oS.b=b1
        elif(0<oS.alphas[j])and(oS.C<oS.alphas[j]):
            oS.b=b2
        else:
            oS.b=(b1+b2)/2.0
        return 1
    else:
        return 0

def calcEk2(oS,k):
    fXk=float(multiply(oS.alphas,oS.labelMat).T*oS.K[:,k]+oS.b)
    Ek=fXk-float(oS.labelMat[k])
    return fXk

def testRbf(k1=1.3):
    dataArr,labelArr=loadDataSet('testSetRBF.txt')
    b,alphas=smoP(dataArr,labelArr,200,0.0001,10000,('rbf',k1))
    dataMat=mat(dataArr)
    labelMat=mat(labelArr).transpose()
    svInd=nonzero(alphas.A>0)[0]
    sVs=dataMat[svInd]
    labelSV=labelMat[svInd]
    print('there are %d Support Vectors'%shape(sVs)[0])
    m,n=shape(dataMat)
    errorCount=0
    for i in range(m):
        kernelEval=kernelTrans(sVs,dataMat[i,:],('rbf',k1))
        predict=kernelEval.T*multiply(labelSV,alphas[svInd])+b
        if(sign(predict))!=sign(labelArr[i]):
            errorCount+=1
    print('the training error rate is %f '%(float(errorCount/m)))
    dataArr,labelArr=loadDataSet('testSetRBF2.txt')
    errorCount=0
    dataMat=mat(dataArr)
    labelMat=mat(labelArr).transpose()
    m,n=shape(dataMat)
    for i in range(m):
        kernelEval=kernelTrans(sVs,dataMat[i,:],('rbf',k1))
        predict=kernelEval.T*multiply(labelSV,alphas[svInd])+b
        if(sign(predict)!=sign(labelArr[i])):
            errorCount+=1
    print('the test error rate is %f '%(float(errorCount/m)))

def loadImage(dirName):
    from os import listdir
    hwLabels=[]
    trainingFileList=listdir(dirName)
    m=len(trainingFileList)
    trainingMat=zeros((m,1024))
    for i in range(m):
        fileNameStr=trainingFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileStr.split('_')[0])
        if(classNumStr==9):
            hwLabels.append(-1)
        else:
            hwLabels.append(1)
        trainingMat[i,:]=image2vector('%s%s'%(dirName,fileNameStr))
    return trainingMat,hwLabels

def testDigits(kTup=('rbf',10)):
    dataArr,labelArr=loadImage('trainingDigits')
    b,alphas=smoP(dataArr,labelArr,200,0.0001,10000,kTup)
    dataMat=mat(dataArra)
    labelMat=mat(labelArr).transpose()
    svInd=nonzero(alphas.A>0)[0]
    sVs=dataMat[svInd]
    labelSV=labelMat[svInd]
    print('there are %d Support vector '%shape(sVs)[0])
    m,n=shape(dataMat)
    errorCount=0
    for i in range(m):
        kernelEval=kernelTrans(sVs,dataMat[i,:],kTup)
        predict=kernelEval.T*multiply(labelSV,alphas[svInd])+b
        if sign(predict)!=sign(labelArr[i]):
            errorCount+=1
    print('the training error rate is %f'%(float(errorCount)/m))
    dataArr,labelArr=loadImage('testDigits')
    errorCount=0
    for i in range(m):
        kernelEval=kernelTrans(sVs,dataMat[i,:],kTup)
        predict=kernelEval.T*multiply(labelSV,alphas[svInd])+b
        if sign(predict)!=sign(labelArr[i]):
            errorCount+=1
    print('the test error rate is %f'%(float(errorCount)/m))
 dataArr,labelArr=loadDataSet('testSet.txt')
b,alphas=smoP(dataArr,labelArr,0.6,0.001,40)
# ws=calcWs(alphas,dataArr,labelArr)
# print(ws)
# dataMat=mat(dataArr)
# print(dataMat[4]*mat(ws)+b)
# print(labelArr[4])

