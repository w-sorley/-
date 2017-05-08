# /usr/bin/env python3
# -*-coding:utf-8-*-

__Author__="Shouli_Wang"

'''
tne julie  machinelearning  algrothm KMean
'''

from numpy import *
import urllib
import json
from time import sleep
import matplotlib
import matplotlib.pyplot as plt


#导入文件数据
def loadDataSet(filename):
    dateMat=[]
    fr=open(filename)
    for line in fr.readlines():
        currLine=line.strip().split('\t')
        fltLine=list(map(float,currLine))
        dateMat.append(fltLine)
    return dateMat


#给定两个数据样本向量，返回两者的欧氏距离
def distEclud(vecA,vecB):
    return sqrt(sum(power(vecA-vecB,2)))

#根据样本点选择聚类簇的质心，返回包含K个随机质心的集合（随机质心的每个维度的取值必须在样本给定的范围内）
def randCent(dateMat,k):
    n=shape(dateMat)[1]
    centroids=mat(zeros((k,n)))
    for j in range(n):
        minJ=min(dateMat[:,j])
        rangeJ=float(max(dateMat[:,j])-minJ)       #对于每个特征利用最大值减掉最小值得到该特征的取值范围
        centroids[:,j]=minJ+rangeJ*random.rand(k,1)  #随机数范围0到1  
    return centroids

def kMeans(dataSet,k,distMeas=distEclud,createCent=randCent):
    m=shape(dataSet)[0]
    clusterAssment=mat(zeros((m,2)))
    centroids=createCent(dataSet,k)  #初始化随机质心
    
    clusterChanged=True
    while clusterChanged:
        clusterChanged=False
        for i in range(m):     #对每个样本进行遍历计算其与每个质心的距离
            minDist=inf
            minIndex=-1
            for j in range(k):  #对每个质心进行遍历
                distJI=distMeas(centroids[j,:],dataSet[i,:])  #计算距离
                if distJI<minDist:  #若小于最小距离，更新最小距离，保存质心编号
                    minDist=distJI
                    minIndex=j
            if clusterAssment[i,0]!=minIndex:  #如果质心不再变化
                clusterChanged=True
            clusterAssment[i,:]=(minIndex,minDist**2)
        print (centroids)
        for cent in range(k):  #遍历所有质心更新他们的取值
            ptsInClust=dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]
            centroids[cent,:]=mean(ptsInClust,axis=0)
    return centroids,clusterAssment


#二分K-均值聚类算法
def biKmeans(dataSet,k,distMeas=distEclud):
    m=shape(dataSet)[0]
    clusterAssment=mat(zeros((m,2)))
    centroid0=mean(dataSet,axis=0).tolist()[0]  #利用均值，计算数据集的质心
    centList=[centroid0]
    for j in range(m):        #遍历计算每条样本与质心的距离
        clusterAssment[j,1]=distMeas(mat(centroid0),dataSet[j,:])**2
    while (len(centList)<k):  #如果质心的数目小于K
        lowestSSE=inf
        for i in range(len(centList)):  #在每个质心上遍历，决定最佳簇进行划分
            ptsInCurrCluster=dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]  #过滤出属于该簇的所有点，作为子数据集
            centroidMat,splitClustAss=kMeans(ptsInCurrCluster,2,distMeas)      #对属于该簇的子数据集进行划分二分
            sseSplit=sum(splitClustAss[:,1])   #划分后的SSE
            sseNotSplit=sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])  #不进行划分的SSE
            print("sseSplit,and notSplit:",sseSplit,sseNotSplit)
            if(sseSplit+sseNotSplit)<lowestSSE:     #如果划分后剩余数据集的SSE+划分后的SSE加和小于最小SSE，记录最佳簇
                bestCentToSplit=i
                bestNewCents=centroidMat  #划分后均值
                bestClustAss=splitClustAss.copy()
                lowestSSE=sseNotSplit+sseSplit           #选择最佳划分簇后进行划分，将上面划分后的簇的信息进行修改
        bestClustAss[nonzero(bestClustAss[:,0].A==1)[0],0]=len(centList)
        bestClustAss[nonzero(bestClustAss[:,0].A==0)[0],0]=bestCentToSplit
        print("the best cent to split is :",bestCentToSplit)
        print("the len of best cent cluster ass is ",len(bestClustAss))
        centList[bestCentToSplit]=bestNewCents[0,:]
        centList.append(bestNewCents[1,:])
        clusterAssment[nonzero(clusterAssment[:,0].A==bestCentToSplit)[0],:]=bestClustAss
    return mat(centList),clusterAssment

def geoGrab(stAddress,city):
    apiStem='http://where.yahooapis.com/geocode?'
    params={}
    params['flags']='J'
    params['appid']='ppp68N8t'
    params['location']='%s %s'%(stAddress,city)
    url_params=urllib.urlencode(params)
    yahooApi=apiStem+url_params
    print(yahooApi)
    c=urllib.urlopen(yahooApi)
    return json.loads(c.read())

def massPlaceFind(filename):
    fw=open('places.txt','w')
    for line in open(filename).readlines():
        line=line.strip()
        lineArr=line.split('\t')
        retDict=geoGrab(lineArr[1],lineArr[2])
        if retDict['ResultSet']['Error']==0:
            lat=float(retDict['ResultSet']['Results'][0]['latitude'])
            lng=float(retDict['ResultSet']['Results'][0]['longitude'])
            print("%s\t%f\t%f"%(lineArr[0],lat,lng))
            fw.write("%s\t%f\t%f\n"%(line,lat,lng))
        else:
            print("error fetching")
        sleep(1)
    fw.close()

def distSLC(vecA,vecB):
    a=sin(vecA[0,1]*pi/180)*sin(vecB[0,1]*pi/180)
    b=cos(vecA[0,1]*pi/180)*cos(vecB[0,1]*pi/180)*\
                                          cos(pi*(vecB[0,0]-vecA[0,0])/180)
    return arccos(a+b)*6371.0

def clusterClubs(numClust=5):
    datList=[]
    for line in open('places.txt').readlines():
        lineArr=line.split('\t')
        datList.append([float(lineArr[4]),float(lineArr[3])])
    datMat=mat(datList)
    myCentroids.clusterAssing=biKmeans(datMat,numClust,distMeas=distSLC)
    fig=plt.figure()
    rect=[0.1,0.1,0.8,0.8]
    scatterMarkers=['s','o','^','8','p','d','v','h','>','<']
    axprops=dict(xticks=[],yticks=[])
    ax0=fig.add_axes(rect,label='ax0',**axprops)
    imP=plt.imread('Portland.png')
    ax0.imshow(imP)
    ax1=fig.add_axes(rect,label='ax1',frameon=False)
    for i in range(numClust):
        ptsInCurrCluster=datMat[nonzero(clusterAssing[:,0].A==i)[0],:]
        markerStyle=scatterMarkers[i%len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0],\
                              ptsInCurrCluster[:,1].flatten().A[0],marker=markerStyle,s=90)
    ax1.scatter(myCentroids[:,0].flatten().A[0],myCentroids[:,1].flatten().A[0],\
                                                                     marker='+',s=300)
    plt.show()




datMat=loadDataSet('testSet2.txt')
datMat=mat(datMat)
centList,myNewAssments=biKmeans(datMat,3)
# clusterAss=kMeans(datMat,4)

# print(min(datMat[:,0]))
# print(randCent(datMat,2))
# print(distEclud(datMat[0],datMat[1]))
