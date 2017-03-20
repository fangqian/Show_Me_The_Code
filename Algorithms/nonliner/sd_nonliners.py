# -*- coding:utf-8 -*-

'''
Description : Check the data input 
require     : windows Anaconda-2.3.0
author      : qiangu_fang@163.com
usage  $ python sd_inventory.py -d -a ...
'''

from decimal import *


import os
import sys
import logging
import math
import numpy as np
import pandas as pd
from sympy import *
from optparse import OptionParser
import numpy.linalg as LA
from numba import jit


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PAR_DIR = os.path.dirname(BASE_DIR)

nonliner_logger = logging.getLogger('SD_API.Method.sd_nonliner')
nonliner_logger.setLevel(logging.INFO)
fh = logging.FileHandler(PAR_DIR + os.path.sep + "LOG" + os.path.sep + "SD_Nonliner.log")
fh.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
fh.setFormatter(formatter)
nonliner_logger.addHandler(fh)

# 线搜索算法框架
# 输入 (目标函数，梯度函数，起始点，方向算法，步长算法，方向算法的参数，步长算法的参数，
#       终止条件的 epsilon，终止条件的迭代次数)
# 注意：如果 epsilon 和 iterations 都不填的话，你就安静地坐下来，等待时间的洪流带你去往彼方吧。
# 输出一个优化的搜寻路径
@jit
def line_search(f, fun, grad, init, direction_algo, steplength_algo, \
                direction_params=None, steplength_params=None, epsilon=0, iterations=float('inf'), var=None, numbers=None):
    nonliner_logger.info("Start computing...")
    sequence = [init]         # 起始点
    data = init
    prev_val = float('inf')    # 正无穷
    counter = 0
    abs_fx = [[fun(f,var,data),None]]

    while abs(fun(f,var,data)-prev_val)>epsilon and counter<iterations:
        p = direction_algo(f,grad,data,var)
        s = steplength_algo(f,grad,var,data,numbers)
        prev_val = fun(f,var,data)
        counter += 1
        data = data+float(s)*p
        sequence.append(data)    # 遍历的点坐标
        abs_fx.append([fun(f,var,data), abs(fun(f,var,data)-prev_val)])

    sequence = np.hstack((sequence,abs_fx))
    
    column = []
    for i in range(len(var)):
        column.append(str(var[i]))
    column += ["f(x)","|Δf(x)|"]
    return(pd.DataFrame(sequence, columns = column))
	
# 梯度下降方向算法
# 输入 (目标函数，梯度函数，x)
# 输出在 x 的负梯度向量
def grad_descent(f, grad, data, var):

    return -(grad(f,var,data))/(np.linalg.norm(grad(f,var,data)))
	
# 牛顿方向算法
# 输入 (目标函数，梯度函数，x，(海塞矩阵函数,))
# 输出（如果海塞矩阵是正定的）牛顿方向，或者负梯度
# def newton_method(f, grad, data, hess, var):
#     hess = hess(f,data,var)
#     if is_pos_def(hess):
#         print(-(np.linalg.inv(hess)).dot(grad(f,var,data))) 
#         return -(np.linalg.inv(hess)).dot(grad(f,var,data))
#     else:
#         print(-grad(f,var,data))
#         return -grad(f,var,data)
		
# 判断一个矩阵是否正定
# 输入矩阵
# 输出 Boolean
def is_pos_def(mat):
    return(np.all(np.linalg.eigvals(mat) > 0) and np.all(mat==mat.T))
	

def grad(f,var,data):
    dicts = dict(list(zip (var, data)))
    lists = [float(diff(f,var[i]).subs(dicts)) for i in range(len(var))]
    #return np.array([float(diff(f,x1).subs({x1:data[0],x2:data[1]})), float(diff(f,x2).subs({x1:data[0],x2:data[1]}))])
    #print np.array(lists) 
    return np.array(lists)
# 最佳步长法：
# 梯度 × 梯度的转置 × 梯度的模
#————————————————————————————————
# 梯度 × f(x)的海瑟矩阵 × 梯度的转置
def best_track(f,grad,var,data,numbers):
    tidu = grad(f,var,data)
    tidu.shape = (1,numbers)
 
    return (tidu.dot(np.mat(tidu).T).dot(LA.norm(tidu)))/(tidu.dot(hess(f,data,var)).dot(np.mat(tidu).T))

def fun(f,var,data):
    """return f(x) value at position data([x,y])"""
    dicts = dict(list(zip (var, data)))
    return f.subs(dicts)
       
def hess(f,data,var):
    lists = []
    for i in range(len(var)):
        lists.append([diff(f,var[i],var[j]) for j in range(len(var))])
    #return np.array([[diff(f,x1,x1),diff(f,x1,x2)], [diff(f,x2,x1),diff(f,x2,x2)]])
    return np.array(lists)




if __name__ == "__main__":
    optparser = OptionParser()
    optparser.add_option('-f', '--function',
                         dest='func',
                         help='function name that want to be solved',
                         type='string',
                         default=None)

    optparser.add_option('-e', '--epsilon',
                         dest='epsilon',
                         help='epsilon  allowed',
                         default=0.01,
                         type='float')
    optparser.add_option('-n', '--number',
                         dest='number',
                         help='numbers of independent variable',
                         default=2,
                         type='int')
    optparser.add_option('-s','--start_point',
                         dest='start_point',
                         help='start point coordinates',
                         default=None,
                         type="string")
    
    (options, args) = optparser.parse_args()

    n = options.number
    names = locals()
    var = []
    for i in xrange(1, n+1):
        names['x%s' % i] = symbols(str('x%s' % i))
        
    for i in xrange(1, n+1):
        var.append(eval('x%s' % i))

    f = eval(options.func)
    data = np.array([0.0,0.0])

    e = options.epsilon

    n = options.number

    if options.start_point:
        s = eval(options.start_point)
    else:
        s = [0]*n

    point = np.array([0.0,0.0])

    nonliner_logger.info("Start,getting parameters")

    result = line_search(f, fun, grad, np.array(s), grad_descent, best_track,direction_params=None, steplength_params=None,epsilon=e, var = var, numbers = n)
    nonliner_logger.info("Computing done! Save result to file")

    full_name = os.path.dirname(os.path.dirname(os.path.realpath(sys.argv[0])))
    result_name = full_name +os.path.sep+"pythonFiles"+os.path.sep+"sd_nonliner_result.txt"


    for x in result.columns:
        if len(result[x])>2:
            try:
                result[x][1:]=result[x][1:].map('{:,.4f}'.format)
            except Exception,e:
                print("Error")
                nonliner_logger.info("float format error")
                break
    result.to_csv(result_name,float_format='%.4f',index = False, header=True, sep="\t")
    
