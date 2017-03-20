# -*- coding:utf-8 -*-

'''
Description : Check the data input 
require     : windows Anaconda-2.3.0
author      : qiangu_fang@163.com
usage  $ python sd_inventory.py -d -a ...
'''

import os
import sys
import copy
import logging
import pandas as pd
from optparse import OptionParser

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PAR_DIR = os.path.dirname(BASE_DIR)

product_logger = logging.getLogger('SD_API.Method.sd_product')
product_logger.setLevel(logging.INFO)
fh = logging.FileHandler(PAR_DIR + os.path.sep + "LOG" + os.path.sep + "SD_Product.log")
fh.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
fh.setFormatter(formatter)
product_logger.addHandler(fh)


def sd_product(d, p, product_fee, s, max_s, pro_fee, init, result_name):
    """
    d:demand,  p:product, product_fee,  s:storage_cost, 
    max_s:max_storage,  pro_fee:stable_fee,  init:original_storage
    """
    n = len(d)
    f=[]
    m=[]
    for i in range(n):
        f.append([0 for j in range(0,min(max_s[i],sum(d[i:]))+1)])
        m.append([0 for j in range(0,min(max_s[i],sum(d[i:]))+1)])
    f.append(copy.deepcopy(f[-1]))
    m.append(copy.deepcopy(m[-1]))

    for i in reversed(range(n)):
        for j in range(0,min(max_s[i],sum(d[i:]))+1):
            f[i][j] = float("inf")  
            for k in range(max(0,d[i]-j), min(p[i], sum(d[i:])-j, max_s[i]-j+d[i])+1):
                if k == 0:
                    w = (j + k - d[i]) * s[i]
                else:
                    w = pro_fee[i] + k * product_fee[i] + (j + k - d[i]) * s[i]
                if f[i][j] > f[i+1][j + k - d[i]] + w:
                    f[i][j] = f[i+1][j + k - d[i]] + w
                    m[i][j] = k

    max_value = f[0][init]
    products = []
    end_storage = []
    storage_fee = []
    product_cost = []
    total_fee = []
    for i in range(n):
    	products.append(m[i][init])
    	init = m[i][init]+init-d[i]
        end_storage.append(init)
        storage_fee.append(init * s[i])
        if products[i] == 0:
            product_cost.append(0)
        else:
            product_cost.append(products[i]*product_fee[i]+pro_fee[i])
        total_fee.append(storage_fee[i]+product_cost[i])

    columns = ["期末存储量", "最优生产量", "生产费用", "存储费用", "单个生产时期总费用"]

    dicts = {"期末存储量":end_storage, "最优生产量":products, "生产费用":product_cost, 
             "存储费用":storage_fee, "单个生产时期总费用":total_fee}

    if result_name:
        product_logger.info("Saving data")
        with open(result_name,"w") as f:
            f.write(str(max_value)+'\n')
            f.write("######\n")
        results = pd.DataFrame(dicts,columns=columns)
        results.to_csv(result_name,index = False, mode = "a", header=False, sep="\t")
        product_logger.info("Done!")

    return(dicts)


if __name__ == "__main__":
    optparser = OptionParser()
    optparser.add_option('-d','--demand',
                         dest='demand',
                         help='demand for each period',
                         default=None,
                         type="string")

    optparser.add_option('-p','--product',
                         dest='product',
                         help='products for each period',
                         default=None,
                         type="string")

    optparser.add_option('-f','--storage_fee',
                         dest='storage_fee',
                         help='per product storage cost for each period',
                         default=None,
                         type="string")

    optparser.add_option('-c','--cost',
                         dest='cost',
                         help='cost for producting a product',
                         default=None,
                         type="string")

    optparser.add_option('-s','--max_storage',
                         dest='max_storage',
                         help='max storage for each period',
                         default=None,
                         type="string")

    optparser.add_option('-t','--stable_cost',
                         dest='stable_cost',
                         help='stable cost  for each period',
                         default=None,
                         type="string")

    optparser.add_option('-o','--original',
                         dest='original',
                         help='original storage before producting',
                         default=None,
                         type="int")
    (options, args) = optparser.parse_args()

    demand = eval(options.demand)
    product = eval(options.product)
    product_fee = eval(options.cost)
    max_storage = eval(options.max_storage)
    storage_cost = eval(options.storage_fee)
    stable_fee = eval(options.stable_cost)
    original_storage = options.original

    full_name = os.path.dirname(os.path.dirname(os.path.realpath(sys.argv[0])))
    result_name = full_name +os.path.sep+"pythonFiles"+os.path.sep+"sd_product_result.txt"

    product_logger.info("Parameters received done!, start computing")
    sd_product(demand, product, product_fee, storage_cost, max_storage, stable_fee, original_storage, result_name)
