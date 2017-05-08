# /usr/bin/env python3
# -*- coding:utf-8 -*-


__Author__="Shouli_Wang"
'''
the machine learing algrothm apriori
'''
from numpy import *
from votesmart import votesmart
from time import sleep


#构建数据集
def loadDataSet():
    return [[1,3,4],[2,3,5],[1,2,3,5],[2,5]]

#创建C1包含所有不重复候选类别
def creatC1(dataSet):
    C1=[]
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return map(frozenset,C1)    #为C1中的每个类别创建不变集合


#ck：数据集，D：包含候选集合的列表，minSupport:感兴趣项集的最小支持度
def scanD(D,Ck,minSupport):    
    ssCnt={}
    for tid in D:
        for can in Ck:                   #比较每个候选项集是否每个数据样本的子集，统计每个候选项集为数据样本子集的数目
            if can.issubset(tid):
                if not ssCnt.has_key(can):
                    ssCnt[can]=1
                else:
                    ssCnt[can]+=1
    numItems=float(len(D))
    retList=[]
    supportData={}
    for key in ssCnt:
        support=ssCnt[key]/numItems    #支持度为以候选项为子集的数据样本的个数/数据样本的总个数
        if support>=minSupport:  #记录大于最小支持度的候选项
            retList.insert(0,key)
        supportData[key]=support
    return retList,supportData


#lk:频繁项集列表，k：项集元素个数
def aprioriGen(Lk,k):
    retList=[]
    lenLk=len(Lk)
    for i in range(lenLk):
        for j in range(i+1,lenLk):   #对第i项以后的项进行遍历
            L1=list(Lk[i])[:k-2]
            L2=list(Lk[j])[:k-2]
            L1.sort()              #如果从开始到倒数第二个均相同则将两者进行合并
            L2.sort()
            if L1==L2:
                retList.append(Lk[i]|Lk[j])
    return retList

def apripri(dataSet,minSupport=0.5):
    C1=creatC1(dataSet)     #创建不重复的候选项集，包含数据集的所有项
    D=map(set,dataSet)         #对每个数据样本进行项的去重
    L1,supportData=scanD(D,C1,minSupport)
    L=[L1]
    k=2
    while (len(L[k-2])>0):    #依次增加项目集中的数目
        Ck=aprioriGen(L[k-2],k)
        Lk,supK=scanD(D,Ck,minSupport)
        supportData.update(supK)
        L.append(Lk)
        k+=1
    return L,supportData


#关联规则生成，supportData：包含各频繁项支持度的列表
def generateRules(L,supportData,minConf=0.7):
    bigRuleList=[]
    for i in range(1,len(L)):  #遍历各个频繁项集
        for frepSet in L[i]:    #遍历每个频繁项集的频繁项
            H1=[frozenset([item]) for item in frepSet]
            if(i>1):
                rulesFromConseq(frepSet,H1,supportData,bigRuleList,minConf)
            else:
                calcConf(frepSet,H1,supportData,bigRuleList,minConf)
    return bigRuleList

#对规则进行评估，返回最小可信度的规则列表
def calcConf(frepSet,H,supportData,br1,minConf=0.7):
    prunedH=[]
    for conSeq in H:
        conf=supportData[frepSet]/supportData[frepSet-conSeq]
        if conf>=minConf:                                #遍历H，返回可信度大于最小可信度的项
            print(frepSet-conSeq,'--->',conSeq,'conf:',conf)
            br1.append((frepSet-conSeq,conSeq,conf))
            prunedH.append(conSeq)
    return prunedH

#生成候选规则集合 H,可以出现在规则右部的元素列表
def rulesFromConseq(frepSet,H,supportData,br1,minConf=0.7):
    m=len(H[0])  #频繁项集的大小
    if(len(frepSet)>(m+1)):
        Hmp1=aprioriGen(H,m+1)  #生成不重复的项集
        Hmp1=calcConf(frepSet,Hmp1,supportData,br1,minConf)   #返回大于最小可信度的项集
        if (len(Hmp1)>1):                                  #如果不重复且大于最小可信度的项个数大于一
            rulesFromConseq(frepSet,Hmp1,supportData,br1,minConf)

def getActionIds():
    actionIdList=[]
    billTitleList=[]
    fr=open('recent20bills.txt')
    for line in fr.readlines():
        billNum=int(line.split('\t')[0])
        try:
            billDetail=votesmart.vote.getBill(billNum)
            for action in billDetail.actions:
                if action.level=='House' and(action.stage=='Amendment Vote'\
                                                        or action.stage=='Passage'):
                    actionId=int(action.actionId)
                    print('bill:%d has actionId: %d'%(billNum,actionId))
                    actionIdList.append(actionId)
                    billTitleList.append(line.strip().split('\t')[1])
        except:
            print('problem getting bill %d'%billNum)
        sleep(1)
    return actionIdList,billTitleList

def getTransList(actionIdList,billTitleList):
    itemMeaning=['Republican','Democratic']
    for billTitle in billTitleList:
        itemMeaning.append('%s --Nay'%billTitle)
        itemMeaning.append('%s --Yea'%billTitle)
    transDict={}
    voteCount=2
    for actionId in actionIdList:
        sleep(3)
        print('getting vote for actionId:%d'%actionId)
        try:
            voteList=votesmart.votes.getBillActionVotes(actionId)
            for vote in voteList:
                if not transDict.has_key(vote.candidataName):
                    transDict[vote.candidataName]=[]
                    if vote.officeParties=='Democratic':
                        transDict[vote.candidataName].append(1)
                    elif vote.officeParties=='Republican':
                        transDict[vote.candidataName].append(0)
                if vote.action=='Nay':
                    transDict[vote.candidataName].append[voteCount]
                elif vote.action=='Yea':
                    transDict[vote.candidataName].append(voteCount+1)
        except:
            print('problem getting actionId :%d'%actionId)
        voteCount+=2
    return transDict,itemMeaning



dataSet=loadDataSet()
# print(dataSet)
# C1=creatC1(dataSet)
# print(list(C1))
# D=list(map(set,dataSet))
# print(D)
# L1,suppData0=scanD(D,C1,0.5)
# print(L1)

L,suppDta=apripri(dataSet)
# rules=generateRules(L,suppDta,minConf=0.5)
# print(rules)
# print(L)
# print(aprioriGen(L[0],2))
