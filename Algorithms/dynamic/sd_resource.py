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

resource_logger = logging.getLogger('SD_API.Method.sd_knapsack')
resource_logger.setLevel(logging.INFO)
fh = logging.FileHandler(PAR_DIR + os.path.sep + "LOG" + os.path.sep + "SD_Resource.log")
fh.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
fh.setFormatter(formatter)
resource_logger.addHandler(fh)

def sourece_allot(r,v,s):
    r.insert(0,0)
    for x in v:
        x.insert(0,0)
    m = len(r)
    n = len(v)
    f=[[0 for i in range(m)] for j in range(n)]
    p=[[0 for i in range(m)] for j in range(n)]
    for j in range(0,n):
        for i in range(0,m):
            for k in range(0,i+1):
                if f[j][i]<f[j-1][i-k]+v[j][k]:
                    p[j][i]=k
                    f[j][i]=f[j-1][i-k]+v[j][k]
    max_value = f[n-1][m-1]
    items = []
    if s == 1:
    	k=m-1
    	for j in reversed(range(0,n)):
            items.append(r[p[j][k]])
    	    k=k-p[j][k]
        items.reverse()

    else:
        lists =  product(range(m),repeat=n)
        for x in lists:
            if sum([r[i] for i in x]) > max(r):continue

            temp = []
            weight = []
            for i,j in enumerate(x):
                weight.append(j)
                temp.append(v[i][j])                    
            if max_value != sum(temp) or sum(weight)>m:continue

            else:
              items.append(list(x))

    return max_value, items
        


def main(r,v,s,result_name):
    max_value, items =  sourece_allot(r,v,s)

    with open(result_name,"w") as f:
            f.write(str(max_value)+'\n')
            f.write("######\n")

    colum = ["投入资源数", "项目收益", "剩余资源"]

    if  s == 1:
        resource_logger.info("Only one plan")
        residual_resources = []
        total_resource = max(r)
        for i in range(len(items)):
            total_resource -= items[i]
            residual_resources.append(total_resource)

        result = {"投入资源数": items,
                 "项目收益": [ v[i][r.index(items[i])] for i in range(len(items)) ],
                 "剩余资源": residual_resources}

        results = pd.DataFrame(result,columns=colum)
        results.to_csv(result_name,index = False, mode = "a", header=False, sep="\t")

    elif items:
        resource_logger.info("Show all plans if have, maybe waste long time")
        for x in items:
            residual_resources = []
            total_resource = max(r)
            for i in range(len(x)):
                total_resource -= x[i]
                residual_resources.append(total_resource)

            result = {"投入资源数": x,
                     "项目收益": [ v[i][r.index(x[i])] for i in range(len(x)) ],
                     "剩余资源": residual_resources}

            results = pd.DataFrame(result,columns=colum)
            results.to_csv(result_name,index = False, mode = "a", header=False, sep="\t")

            with open(result_name,"a") as f:
                f.write("######\n")


    else:
        print("Error, haven't return items")
        resource_logger.info("Error, haven't return items")

if __name__ == "__main__":
    optparser = OptionParser()
    optparser.add_option('-r','--resource',
                         dest='resource',
                         help='Investment(put into) of resources',
                         default=None,
                         type="string")

    optparser.add_option('-v','--value',
                         dest='value',
                         help='output values of each items',
                         default=None,
                         type="string")

    optparser.add_option('-s','--number',
                         dest='scheme',
                         help='number of scheme',
                         default=1,
                         type="int")
    (options, args) = optparser.parse_args()

    r = eval(options.resource)
    v = eval(options.value)
    s = options.scheme

    full_name = os.path.dirname(os.path.dirname(os.path.realpath(sys.argv[0])))
    result_name = full_name +os.path.sep+"pythonFiles"+os.path.sep+"sd_resource_result.txt"

    resource_logger.info("Parameters received done!, start computing")
    main(r,v,s,result_name)
