# -*- coding:utf-8 -*-
'''
Description : Check the data input 
require     : windows Anaconda-2.3.0
author      : qiangu_fang@163.com
usage  $ python sd_queue.py -c channel -n number -u service -d channel_cost -e customer_cost --cn customer_number
'''
import os
import sys
import logging
import math
import pandas as pd
from optparse import OptionParser

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PAR_DIR = os.path.dirname(BASE_DIR)

queue_logger = logging.getLogger('SD_API.Method.sd_queue')
queue_logger.setLevel(logging.INFO)
fh = logging.FileHandler(PAR_DIR + os.path.sep + "LOG" + os.path.sep + "SD_Queue.log")
fh.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
fh.setFormatter(formatter)
queue_logger.addHandler(fh)


dicts = {"n":"平均到达率","u":"平均服务率","p":"系统中无顾客的概率","l1":"平均排队的顾客数","l2":"系统中的平均顾客数",
         "w1":"顾客花在排队上的平均等待时间","w2":"顾客在系统中的平均逗留时间","p1":"顾客得不到及时服务必须排队等待的概率",
         "cost":"单位时间总成本","m":"服务时间的标准差"}

def M_M_C(c, n, u, d, e):
    x = 0
    for k in range(c):
        x += (1.0/math.factorial(k)) * ((n/u)**k)
    y = n/(c*u)
    p = 1.0/(x + 1.0/(math.factorial(c)) * (1.0/(1.0-y)) * ((n/u)**c))
    l1 = (n/u)**c * y/(math.factorial(c)*(1.0-y)**2) * p
    l2 = l1 + n/u
    w1 = l1/n
    w2 = w1 + 1.0/u
    p1 = 1.0/math.factorial(c) * (n/u)**c * (c*u/(c*u-n))*p
    
    cost = c * d + l2 * e

    pn = []
    for i in range(11):
        if i<= c:
            pn.append(((n/u)**i)/(math.factorial(i)) * p )
        else:
            pn.append((n/u)**i/(math.factorial(c)*(c**(i-c)))* p)

    return ({"n":n,"u":u,"p":p,"l1":l1,"l2":l2,"w1":w1,"w2":w2,"p1":p1,"cost":cost}, pn)

def M_G_1(n, u, d, e, m):
    p = 1.0 - n/u
    l1 = 0.5*(n**2 * m**2 + (n/u)**2)/(1.0-n/u)
    l2 = l1 +n/u
    w1 = l1/n
    w2 = w1 + 1/u 
    p1 = n/u

    cost = c*d + l2*e

    pn =None

    return ({"n":n,"u":u,"m":m,"p":p,"l1":l1,"l2":l2,"w1":w1,"w2":w2,"p1":p1,"cost":cost},pn)  

def M_G_C_C(c, n, u):
    a = sum([(n/u)**j/math.factorial(j) for j in range(c+1)])
    pn = []
    for i in range(c+1):
        pn.append((n/u)**i/math.factorial(i)/a)

    p = 1.0/a
    l1 = (n/u)/(1.0-(n/u)**c/math.factorial(c)/a)

    return ({"n":n,"u":u,"p":p,"l1":l1}, pn)

def M_M_c_m(c, cn, n, u, d, e): 
   
    x = (cn*n)/(c*u)  

    a = sum([1.0/(math.factorial(k)*math.factorial(cn-k))*(c*x/cn)**k for k in range(c+1)])
    b = c**c/math.factorial(c) * sum([1.0/math.factorial(cn-k)*((x/cn)**k) for k in range(c+1, cn+1)])

    p = 1.0/(math.factorial(cn) *(a+b) )

    def Pn(cn, c, i, n, u, p):
        if  0<=i<=c:
            return math.factorial(cn)/(math.factorial(cn-i)*math.factorial(i)) * (n/u)**i * p
        elif c+1 <=i <= cn:
            return math.factorial(cn)/(math.factorial(cn-i)*math.factorial(c)*c**(i-c)) * (n/u)**i * p

    
    l2 = sum([i * Pn(cn, c, i, n, u,p) for i in range(1, cn+1)])
    l1 = sum([(i-c) * Pn(cn, c, i, n, u,p) for i in range(c+1, cn+1)])

    l = n*(cn - l1)

    w1 = l1/l
    w2 = l2/l

    cost = c*d + l2*e

    p1 = sum([Pn(cn, c, i, n, u,p) for i in range(c,cn+1)])

    pn = [Pn(cn, c, i, n, u,p) for i in range(0, cn+1)]

    return ({"n":n,"u":u,"p":p, "l1":l1, "l2":l2, "w1":w1, "w2":w2, "cost":cost, "p1":p1}, pn)

def M_M_s(cn, c, n, u):
    x = n/(c*u)
    sum_a = sum([(c*x)**k/math.factorial(k) for k in range(c+1)])

    p = 1.0/(sum_a + c**c/math.factorial(c) * (x*(x**c-x**cn)/(1-x)))

    def Pn(c, x, i, p):
        if 0<=i<=c:
            return (c*x)**i/math.factorial(i) * p
        elif c<=i<=cn:
            return c**c/math.factorial(c) * (x**i) * p

    l1 = p*x*(c*x)**c/(math.factorial(c) * (1.0-x)**2) * (1.0 - x**(cn-c) - (cn-c) * x**(cn-c) * (1.0-x))
    l2 = l1 + c*x*(1.0-Pn(c, x, cn, p))

    w1 = l1/n*(1.0-Pn(c, x, cn, p))
    w2 = w1 + 1.0/u
    
    pn = [Pn(c, x, i, p) for i in range(0, cn+1)]

    return ({"n":n,"u":u,"p":p, "l1":l1, "l2":l2, "w1":w1, "w2":w2},pn)

def main(method, c, n, u, d, e, m, cn):
    if method == "M_M_C":
        summary, Pn = M_M_C(c, n, u, d, e)

    elif method == "M_G_1":
        summary, Pn = M_G_1(n, u, d, e, m)

    elif method == "M_G_C_C":
        summary, Pn = M_G_C_C(c, n, u)

    elif method == "M_M_c_m":
        summary, Pn = M_M_c_m(c, cn, n, u, d, e)

    elif method == "M_M_s":
        summary, Pn = M_M_s(cn, c, n, u)

    else:queue_logger.info("Method not find")

    return summary,Pn

if __name__ == "__main__":
    optparser = OptionParser()

    optparser.add_option('-a', '--method',
                         dest='method',
                         help='queue theory method',
                         default=None,
                         type="string")

    optparser.add_option('-c', '--channel',
                         dest='channels',
                         help='number of channels',
                         default=1,
                         type="int")

    optparser.add_option('-n', '--number',
                         dest='arrive',
                         help='average arrive rates',
                         default=None,
                         type="float")


    optparser.add_option('-u', '--service',
                         dest='service',
                         help='average service rates',
                         default=None,
                         type="float")

    optparser.add_option('-d', '--channel_cost',
                         dest='ch_cost',
                         help='Channel costs',
                         default=None,
                         type="float")

    optparser.add_option('-e', '--customer_cost',
                         dest='cu_cost',
                         help='customer cost',
                         default=None,
                         type="float")

    optparser.add_option('-m', '--mean_square',
                     dest='mean',
                     help='mean-square deviation',
                     default=None,
                     type="float")

    optparser.add_option('--cn', '--channel_number',
                     dest='cu_number',
                     help='customer numbers',
                     default=None,
                     type="int")

    (options, args) = optparser.parse_args()
    
    queue_logger.info("Start,getting parameters")
    method = options.method
    c = options.channels
    n = options.arrive
    u = options.service
    d = options.ch_cost
    e = options.cu_cost
    m = options.mean
    cn = options.cu_number

    summary,Pn = main(method, c, n, u, d, e, m, cn)
    
    queue_logger.info("Computing...")
   
    full_name = os.path.dirname(os.path.dirname(os.path.realpath( sys.argv[0])))

    queue_logger.info("Computing done, save data into file")
    result_name = full_name +os.path.sep+"pythonFiles"+os.path.sep+"sd_queue_result.txt"

    f = open(result_name, "w+")
    for x in summary.keys():
        f.write(dicts[x]+"\t"+str(summary[x])+"\n")
    f.close()

    if Pn != None:
        f = open(result_name, "a+")
        f.write("******"+"\n")
        f.close()
        Pn = pd.DataFrame(Pn)
        Pn.to_csv(result_name,index = True, header=None,float_format = "%.6f",mode = "a", sep="\t")
