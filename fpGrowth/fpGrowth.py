# /usr/bin/env pyhon3
# -*-coding:utf-8 -*-

__Author__='Shouli_Wang'


'''
the machine learning algrothm  FP-Growth
'''


from numpy import *

#FP树的节点类定义
class treeNode:

    def __init__(self,nameValue,numOccur,parentNode):
        self.name=nameValue
        self.Count=numOccur   #计数值
        self.nodeLink=None   #用于链接相似元素
        self.parent=parentNode   #存放父节点
        self.children={}         #存在孩子

    def inc(self,numOccur):    #自增
        self.Count+=numOccur

    def disp(self,ind=1):       #将节点信息以文本方式打印输出
        print(" "*ind,self.name," ",self.Count)
        for child in self.children.values():
            child.disp(ind+1)


#创建FP树
def createTree(dataSet,minSup=1):
    headerTable={}
    for trans in dataSet:
        for item in trans:
            headerTable[item]=headerTable.get(item,0)+dataSet[trans]     #遍历数据集和数据集中的每个样本统计每个项出现的次数
    for k in headerTable.keys():
        if headerTable[k]<minSup:               #如果项集中的项不满足最小支持度，则将其从项集中删除
            del(headerTable[k])
    freqItemSet=set(headerTable.keys())   #去掉重复项，得到频繁项集
   
    if len(freqItemSet)==0:
        return None,None           #如果频繁项集为空，则停止创建
   
    for k in headerTable:   #遍历，扩展每个频繁项的结构，以便保存其他信息
        headerTable[k]=[headerTable[k],None]
    retTree=treeNode("null tree",1,None)    #创建一个包含为空的根节点
  
    for tranSet,Count in dataSet.items():
        localD={}
        for item in tranSet:
            if item in freqItemSet:        #对每个数据样本的每个出现的项，若其在频繁项集中得到他的出现的次数
                localD[item]=headerTable[item][0]
        if len(localD)>0:                           #对数据集中的每个数据样本根据支持度进行排序
            orderedItems=[v[0] for v in sorted(localD.items(),key=lambda p:p[1],reverse=True)]
            updateTree(orderedItems,retTree,headerTable,Count)   #更新树的结构，headertable 头指针表
    return retTree,headerTable

def updateTree(items,inTree,headerTable,count):
    if items[0] in inTree.children:       #若事物的第一个元素作为字节点存在，则子节点自增一
        inTree.children[items[0]].inc(count)
    else:
        inTree.children[items[0]]=treeNode(items[0],count,inTree)    #否则根据事物的元素创建一个新的子节点
        if headerTable[items[0]][1]==None:
            headerTable[items[0]][1]=inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1],inTree.children[items[0]])    #更新头指针表
    if len(items)>1:
        updateTree(items[1::],inTree.children[items[0]],headerTable,count)   #如果项的个数大于一，迭代继续更新树结构

def updateHeader(nodeToTest,targetNode):
    while (nodeToTest.nodeLink!=None):  #根据链接，直到链尾部
        nodeToTest=nodeToTest.nodeLink
    nodeToTest.nodeLink=targetNode   #将目标节点加到链表尾部


def  loadSimpDat():
    simpDat=[['r','z','h','j','p'],
                    ['z','y','x','w','v','u','t','s'],
                    ['z'],
                    ['r','x','n','o','s'],
                    ['y','r','x','z','q','t','p'],
                    ['y','z','x','e','q','s','t','m']]
    return simpDat

#对数据进行格式化处理
def createInitSet(dataSet):
    retDict={}
    for trans in dataSet:
        retDict[frozenset(trans)]=1
    return retDict

def ascendTree(leafNode,prefixPath):
    if leafNode.parent!=None:             #若节点的父节点不为空，保存节点名，迭代遍历继续向上搜索
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent,prefixPath)

#对于给定尾元素项生成条件模式基（包含给定元素项以此结尾或以关联节点结尾的路径）
def findPrefixPath(basePat,treeNode):
    condPats={}
    while treeNode!=None:
        prefixPath=[]
        ascendTree(treeNode,prefixPath)
        
        if len(prefixPath)>1:
            condPats[frozenset(prefixPath[1:])]=treeNode.Count
        treeNode=treeNode.nodeLink   #遍历所有关联节点
    return condPats

#递归查找频繁项集
def mineTree(inTree,headerTable,minSup,preFix,freqItemList):
    bigL=[v[0] for v in sorted(headerTable.items(),key=lambda p:p[1])]   #从头指针表的底端开始
    for basePat in bigL:
        newFreqSet=preFix.copy()                      #从条件模式集来构建条件FP树
        newFreqSet.add(basePat)
        freqItemList.append(newFreqSet)    #将每一个频繁项添加到频繁项列表
        condPattBases=findPrefixPath(basePat,headerTable[basePat][1])   #对头指针表中的项建立前缀路径
        myCondTree,myHead=createTree(condPattBases,minSup)   #条件基被当作新的数据集创建树结构
        if myHead!=None:                           #迭代直到树中元素为空
            # print('condutional tree for:',newFreqSet)
            # myCondTree.disp()
            mineTree(myCondTree,myHead,minSup,newFreqSet,freqItemList)



simpDat=loadSimpDat()
# print(simpDat)
initSet=createInitSet(simpDat)
# print(initSet)
myFPtree,myHeaderTab=createTree(initSet,3)
condPats=findPrefixPath('r',myHeaderTab['r'][1])

freqItems=[]
mineTree(myFPtree,myHeaderTab,3,set([]),freqItems)
print(freqItems)

# print(condPats)


# myFPtree.disp()


# rootNode=treeNode('pyramid',9,None)
# rootNode.children['eye']=treeNode('eye',13,None)
# rootNode.children['phoenix']=treeNode('phoenix',3,None)
# rootNode.disp()
