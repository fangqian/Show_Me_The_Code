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

# #数据1
n=3  #项目数
c=5  #资源总数
m=5  #可选择的投资策略
r=[0,2,5,6,8]
v=[[0,8,15,30,38],[0,9,20,35,40],[0,10,28,35,43]] #每种投资策略下每个项目可获得的收益
#r=[0,1,2,3,4,5]
#v=[[0,6,7,10,12,15],[0,3,7,9,12,13],[0,5,10,11,11,11],[0,4,6,11,12,12]]


def sourece_allot(n,m,r,v,s):
	f=[[0 for i in range(m)] for j in range(n)]
	p=[[0 for i in range(m)] for j in range(n)]
	for j in range(0,n):
		for i in range(0,m):
			for k in range(0,i+1):
				if f[j][i]<f[j-1][i-k]+v[j][k]:
					p[j][i]=k
					f[j][i]=f[j-1][i-k]+v[j][k]

    max_value = f[n-1][m-1]
    
	k=m-1
	for j in reversed(range(0,n)):
		print ('The project is:',j+1,';The investment is:',r[p[j][k]])
		k=k-p[j][k]


if __name__ == "__main__":
    optparser = OptionParser()
    optparser.add_option('-n', '--items',
                         dest='items',
                         help='number of items',
                         default=None,
                         type='int')

    optparser.add_option('-m', '--strategies',
                         dest='strateiges',
                         help='strategies',
                         default=None,
                         type='int')

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

    n = options.items
    m = options.strategies
    r = eval(options.weight)
    v = eval(options.value)
    s = options.scheme

    full_name = os.path.dirname(os.path.dirname(os.path.realpath(sys.argv[0])))
    result_name = full_name +os.path.sep+"pythonFiles"+os.path.sep+"sd_resource_result.txt"

    knapsack_logger.info("Parameters received done!, start computing")
    main(n,m,r,v,s,result_name)
