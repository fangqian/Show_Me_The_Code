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
                direction_params=None, steplength_params=None, epsilon=0, iterations=float('inf')):
    nonliner_logger.info("Start computing...")
    sequence = [init]         # 起始点
    data = init
    prev_val = float('inf')    # 正无穷
    counter = 0
    abs_fx = [[fun(f,data),None]]

    while abs(fun(f,data)-prev_val)>epsilon and counter<iterations:
        p = direction_algo(f,grad,data,hess)
        s = steplength_algo(f,grad,data)
        prev_val = fun(f,data)
        counter += 1
        data = data+float(s)*p
        sequence.append(data)    # 遍历的点坐标
        abs_fx.append([fun(f,data), abs(fun(f,data)-prev_val)])

    sequence = np.hstack((sequence,abs_fx))
    return(pd.DataFrame(sequence, columns = ["x","y","f(x)","|Δf(x)|"]))
	
# 梯度下降方向算法
# 输入 (目标函数，梯度函数，x)
# 输出在 x 的负梯度向量
def grad_descent(f, grad, data, hess):
    # print(np.linalg.norm(grad(f,data)))     #print |G(k)|
    return -(grad(f,data))/(np.linalg.norm(grad(f,data)))
	
# 牛顿方向算法
# 输入 (目标函数，梯度函数，x，(海塞矩阵函数,))
# 输出（如果海塞矩阵是正定的）牛顿方向，或者负梯度
def newton_method(f, grad, x, hess):
    hess = hess(f,x)
    if is_pos_def(hess):
        return -(np.linalg.inv(hess)).dot(grad(f,x))
    else:
        return -grad(x)
		
# 判断一个矩阵是否正定
# 输入矩阵
# 输出 Boolean
def is_pos_def(mat):
    return(np.all(np.linalg.eigvals(mat) > 0) and np.all(mat==mat.T))
	

def grad(f,data):
    return np.array([float(diff(f,x).subs({x:data[0],y:data[1]})), float(diff(f,y).subs({x:data[0],y:data[1]}))])

# 最佳步长法：
# 梯度 × 梯度的转置 × 梯度的模
#————————————————————————————————
# 梯度 × f(x)的海瑟矩阵 × 梯度的转置
def best_track(f,grad,data):
    tidu = grad(f,data)
    tidu.shape = (1,2)

    return (tidu.dot(np.mat(tidu).T).dot(LA.norm(tidu)))/(tidu.dot(hess(f,data)).dot(np.mat(tidu).T))

def fun(f,data):
    """return f(x) value at position data([x,y])"""
    return f.subs({x:data[0],y:data[1]})
       
def hess(f,data):
    return np.array([[diff(f,x,x),diff(f,x,y)], [diff(f,y,x),diff(f,y,y)]])




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
                         default=[2,2],
                         type="string")
    
    (options, args) = optparser.parse_args()

    
    x,y = symbols("x y")
    f = eval(options.func)
    data = np.array([0.0,0.0])

    e = options.epsilon

    n = options.number

    s = eval(options.start_point)

    point = np.array([0.0,0.0])

    nonliner_logger.info("Start,getting parameters")

    result = line_search(f, fun, grad, np.array(s), grad_descent, best_track,direction_params=None, steplength_params=None,epsilon=e)
    nonliner_logger.info("Computing done! Save result to file")

    full_name = os.path.dirname(os.path.dirname(os.path.realpath(sys.argv[0])))
    result_name = full_name +os.path.sep+"pythonFiles"+os.path.sep+"sd_nonliner_result.txt"


    for x in result.columns:
        if len(result[x])>2:
            try:
                result[x][1:]=result[x][1:].map('{:,.4f}'.format)
            except Exception,e:
                break
    result.to_csv(result_name,float_format='%.4f',index = False, header=True, sep="\t")
    
