# -*- coding:utf-8 -*-

'''
Description : Check the data input 
require     : windows Anaconda-2.3.0
author      : qiangu_fang@163.com
usage  $ python sd_inventory.py -d -a ...
'''

import os
import sys
import logging
import pandas as pd
from itertools import product
from optparse import OptionParser

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PAR_DIR = os.path.dirname(BASE_DIR)

knapsack_logger = logging.getLogger('SD_API.Method.sd_knapsack')
knapsack_logger.setLevel(logging.INFO)
fh = logging.FileHandler(PAR_DIR + os.path.sep + "LOG" + os.path.sep + "SD_Knapsack.log")
fh.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
fh.setFormatter(formatter)
knapsack_logger.addHandler(fh)

# 0-1 knapsack
def knapsack(c, w, v, amount, s):              # Returns solution matrices
    n = len(w)                                 # Number of available items
    m = [[0]*(c+1) for i in range(n+1)]        # Empty max-value matrix
    P = [[False]*(c+1) for i in range(n+1)]    # Empty keep/drop matrix
    for k in range(1,n+1):                     # We can use k first objects
        i = k-1                                # Object under consideration
        for r in range(1,c+1):                 # Every positive capacity
            m[k][r] = drop = m[k-1][r]         # By default: drop the object
            if w[i] > r: continue              # Too heavy? Ignore it
            keep = v[i] + m[k-1][r-w[i]]       # Value of keeping it
            m[k][r] = max(drop, keep)          # Best of dropping and keeping
            P[k][r] = keep > drop              # Did we keep it?

    if s==1:
        k, r, items = len(w), c, list()
        while k >0 and r >0:
            i = k-1
            if P[k][r]:
                items.append(i)
                r -= w[i]
            k -= 1
        #items.reverse()
        for i in items:amount[i] = 1 
        items = amount

    else:
        items = []
        lists = product(xrange(2),repeat=n)

        for x in lists:
            temp = []
            indexs = []
            for i,j in enumerate(x):
                if j != 0: 
                    temp.append(v[i])

            if m[n][c]==sum(temp):
               items.append(x)

    return items, m[n][c]                              

def multiple_knapsack(c,w,v,amount,s):
    n = len(w)  
    amounts = []
    for i in xrange(0,n):
        if amount[i]:
            amounts.append(min(amount[i],c/w[i]))
        else:
            amounts.append(c/w[i])


    res=[[0 for j in range(c+1)] for i in range(n+1)]
    count=[[0 for j in range(c+1)] for i in range(n+1)]
    for i in range(1,n+1):  
        for j in range(0,c+1):  
            res[i][j]=res[i-1][j]
            for k in xrange(0,min(j/w[i-1],amounts[i-1])+1):
                if j>=k*w[i-1] and res[i][j]<=res[i-1][j-k*w[i-1]]+k*v[i-1]:  
                    res[i][j]=res[i-1][j-k*w[i-1]]+k*v[i-1]
                    count[i][j]=k   
    if s == 1:
        items = []
        x=[False for i in range(n+1)]  
        for j in xrange(0,c+1): 
            for i in reversed(range(1,n+1)):
                if res[i][j]>res[i-1][j]:  
                    x[i]=True  
                    j-=count[i][j]*w[i-1]

        j=c 
        for i in reversed(range(n+1)):  
            if x[i]:
                items.append(count[i][j])
                j-=count[i][j]*w[i-1]

            elif i>0:
                items.append(0) 
        items.reverse()

    else:
        items = []
        max_amount=max(amounts)
        lists = product(range(max_amount+1),repeat=n)

        for x in lists:
            temp = []
            weight = []
            for i,j in enumerate(x): 
                weight.append(j*w[i])               
                if j <= amounts[i]: 
                    temp.append(j*v[i])                    
                else:break
            if res[n][c] != sum(temp) or sum(weight)>c:continue

            else:
              items.append(x)

    return items,res[n][c]

def main(f,c,w,v,m,s,result_name):
    item, max_value = f(c, w, v, m, s)
    result = []

    with open(result_name,"w") as f:
            f.write(str(max_value)+'\n')
            f.write("######\n")

    colum = ["选择的物品数量","单位物品重量","单位物品价值","物品的总价值","剩余容量"]

    if s==1:
        capacity = c
        surplus_capacity = []
        for i in range(len(item)):
            capacity -= item[i]*w[i]
            surplus_capacity.append(capacity)

        result = {
                  "选择的物品数量":item,
                  "单位物品重量":w,
                  "单位物品价值":v,
                  "物品的总价值":[item[i]*v[i] for i in range(len(v))],
                  "剩余容量":surplus_capacity,
        }

        results = pd.DataFrame(result,columns=colum)

        knapsack_logger.info("Only one plan")

        results.to_csv(result_name,index = False, mode = "a", header=False, sep="\t")

    elif len(item)>1:
        for x in item:
            capacity = c
            surplus_capacity = []
            for i in range(len(x)):
                capacity -= x[i]*w[i]
                surplus_capacity.append(capacity)

            result = {
                      "选择的物品数量":x,
                      "单位物品重量":w,
                      "单位物品价值":v,
                      "物品的总价值":[x[i]*v[i] for i in range(len(v))],
                      "剩余容量":surplus_capacity,
            }
            
            results = pd.DataFrame(result,columns=colum)

            knapsack_logger.info("Show all plans if have, maybe waste long time")

            results.to_csv(result_name,index = False, mode = "a", header=False, sep="\t")

            with open(result_name,"a") as f:
                f.write("######\n")
    else:
        print("Error, haven't return item")
        knapsack_logger.info("Error, haven't return item")

    return item




if __name__ == "__main__":
    optparser = OptionParser()
    optparser.add_option('-f', '--function',
                         dest='func',
                         help='function name that want to be solved',
                         type='string',
                         default=None)

    optparser.add_option('-c', '--capacity',
                         dest='capacity',
                         help='capacity of each goods',
                         default=None,
                         type='int')
    optparser.add_option('-w', '--weight',
                         dest='weight',
                         help='weight of each goods',
                         default=None,
                         type='string')
    optparser.add_option('-v','--value',
                         dest='value',
                         help='value of each goods',
                         default=None,
                         type="string")

    optparser.add_option('-m','--amount',
                         dest='amount',
                         help='amount of each goods',
                         default=None,
                         type="string")

    optparser.add_option('-s','--number',
                         dest='scheme',
                         help='number of scheme',
                         default=1,
                         type="int")
    (options, args) = optparser.parse_args()

    f = eval(options.func)
    c = options.capacity
    w = eval(options.weight)
    v = eval(options.value)

    if f == knapsack:m = [0]*len(w)
    else:m = eval(options.amount)

    s = options.scheme

    full_name = os.path.dirname(os.path.dirname(os.path.realpath(sys.argv[0])))
    result_name = full_name +os.path.sep+"pythonFiles"+os.path.sep+"sd_knapsack_result.txt"

    knapsack_logger.info("Parameters received done!, start computing")
    main(f,c,w,v,m,s,result_name)

