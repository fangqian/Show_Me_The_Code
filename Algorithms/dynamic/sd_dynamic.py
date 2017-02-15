
# -*- coding:utf-8 -*-   #字符编码的转换

import math;


import sys
reload(sys)
sys.setdefaultencoding('utf8')

#多重背包问题
#背包问题：i表示物品数下标，j表示容量的下标，c表示容器的容量，v[i]表示物品i的价值，w[i]表示物品i的重量，
#res[i][j]表示前i个物品中装入容量为j的背包中物品的最大价值
def bag(n,c,w,v,amount):  
    res=[[0 for j in range(c+1)] for i in range(n+1)]
    count=[[0 for j in range(c+1)] for i in range(n+1)]
  
    for i in range(1,n+1):  
        for j in range(1,c+1):
            res[i][j]=res[i-1][j]
            #print ("i:",i,";j:",j,";res[i][j]:",res[i][j])
            #print ("w[i-1]:",w[i-1],";v[i-1]:",v[i-1],";res[i-1][j-w[i-1]]:",res[i-1][j-w[i-1]])
            for k in xrange(1,min(c/w[i-1],amount[i-1]+1)):
                if j>=k*w[i-1] and res[i][j]<res[i-1][j-k*w[i-1]]+k*v[i-1]:  
                    res[i][j]=res[i-1][j-k*w[i-1]]+k*v[i-1]
                    print res[i][j]
                    
                    count[i][j]=k
                    #j=j-k*w[i-1]
    #print res,count             
    return res,count
  
def show(n,c,w,res,count):
    #print res 
    #print count 
    print(u"最大价值为:%d"%res[n][c])  
    x=[False for i in range(n+1)]  
    for j in xrange(0,c+1): 
        for i in reversed(range(1,n+1)):
            #print ('res[',i,'][',j,']',res[i][j],';res[',i-1,'][',j,']',res[i-1][j])  
            if res[i][j]>res[i-1][j]:  
                x[i]=True  
                j-=count[i][j]*w[i-1]
             
    #print x  
    print(u"选择的物品为:") 
    j=c 
    for i in reversed(range(n+1)):  
        if x[i]:
            #print j
            print(u"第"+str(i)+u"个"+str(count[i][j])+u"件")
            j-=count[i][j]*w[i-1]
            #print(u"第"+str(i)+u"个")
            #print(u"第"+str(i)+u"个")
        elif i>0:
            print(u"第"+str(i)+u"个"+str(0)+u"件")  
    #print('')     
  
if __name__=='__main__':  

#完全背包问题的例子
    '''
    n=5  
    c=10  
    w=[2,2,6,5,4]  
    v=[6,3,5,4,6]
    '''

#多重背包问题的例子
    n=3  
    c=20  
    w=[3,4,5]  
    v=[4,5,6]
    amount=[7,2,2]

    res,count=bag(n,c,w,v,amount)
  
    show(n,c,w,res,count)