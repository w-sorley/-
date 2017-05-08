#!/usr/bin/env python3
# -*-coding=utf-8 -*-

__Author__="Shouli_Wang"
'''
plot the graph of tree
'''

import matplotlib.pyplot as plt
                                                  #定义节点和连线的形式
decisionNode=dict(boxstyle="sawtooth",fc="0.8")
leafNode=dict(boxstyle="round4",fc="0.8")
arrow_args=dict(arrowstyle="<-")

def plotNode(nodeTxt,centerPt,parentPt,nodeType):
    createPlot.ax1.annotate(nodeTxt,xy=parentPt,xycoords='axes fraction',
                            xytext=centerPt,textcoords="axes fraction",va="center",
                            ha="center",bbox=nodeType,arrowprops=arrow_args)

def plotMidText(centrPt,parentPt,txtString):
    xMid=(parentPt[0]-centrPt[0])/2.0+centrPt[0]
    yMid=(parentPt[1]-centrPt[1])/2.0+centrPt[1]
    createPlot.ax1.text(xMid,yMid,txtString)

def plotTree(myTree,parentPt,nodeTxt):
    numLeafs=getNumLeafs(myTree)
    depth=getTreeDepth(myTree)
    firstStr=list(myTree.keys())[0]
    centrPt=(plotTree.x0ff+(1.0+float(numLeafs))/2.0/plotTree.totalW,plotTree.y0ff)
    plotMidText(centrPt,parentPt,nodeTxt)
    plotNode(firstStr,centrPt,parentPt,decisionNode)
    secondDict=myTree[firstStr]
    plotTree.y0ff=plotTree.y0ff-1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            plotTree(secondDict[key],centrPt,str(key))
        else:
            plotTree.x0ff=plotTree.x0ff+1.0/plotTree.totalW
            plotNode(secondDict[key],(plotTree.x0ff,plotTree.y0ff),centrPt,leafNode)
            plotMidText((plotTree.x0ff,plotTree.y0ff),centrPt,str(key))
    plotTree.y0ff=plotTree.y0ff+1.0/plotTree.totalD


def createPlot(inTree):
    fig=plt.figure(1,facecolor="white")
    fig.clf()
    axprops=dict(xticks=[],yticks=[])
    createPlot.ax1=plt.subplot(111,frameon=False,**axprops)
    plotTree.totalW=float(getNumLeafs(inTree))
    plotTree.totalD=float(getTreeDepth(inTree))
    plotTree.x0ff=-0.5/plotTree.totalW
    plotTree.y0ff=1.0
    plotTree(inTree,(0.5,1.0),'')
    plt.show()
    # plotNode('a decision node',(0.5,0.1),(0.1,0.5),decisionNode)
    # plotNode('a leaf node',(0.8,0.1),(0.3,0.8),leafNode)
    # plt.show()

def getNumLeafs(myTree):                    #递归获得数的叶子节点数目
    numLeafs=0;
    firstStr=list(myTree.keys())[0]
    secondDict=myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            numLeafs+=getNumLeafs(secondDict[key])
        else:
            numLeafs+=1
    return numLeafs
                                                       #递归获得树的的深度
def getTreeDepth(myTree):
    maxDepth=0
    firstStr=list(myTree.keys())[0]
    secondDict=myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            thisDepth=1+getTreeDepth(secondDict[key])
        else:
            thisDepth=1
        if thisDepth>maxDepth:
            maxDepth=thisDepth
    return maxDepth

def retrieveTree(i):
    listOfTrees=[{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                                         {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head':{0:'no', 1: 'yes'}},1:'no'}}}}]

    return listOfTrees[i]

# myTree=retrieveTree(0)
# print(getNumLeafs(myTree))
# print(getTreeDepth(myTree))
# myTree['no surfacing'][3]='maybe'
# createPlot(myTree)

# print(myTree)
