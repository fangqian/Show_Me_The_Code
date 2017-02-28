#!/usr/bin/env  Python
#-*- coding=utf-8 -*-
'''
Description : statistic indicator pearson coeff
require     : windows Anaconda-2.3.0
author      : shizhongxian@126.com
usage  $python sd_pearson.py  -f table.txt 
'''

import pandas as pd
import numpy as np
import sys
import logging
import os
from optparse import OptionParser

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PAR_DIR = os.path.dirname(BASE_DIR)

pear_logger = logging.getLogger('SD_API.Method.PEARSON')
pear_logger.setLevel(logging.INFO)
fh = logging.FileHandler(PAR_DIR + os.path.sep + "LOG" + os.path.sep + "SD_PEAR.log")
fh.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
fh.setFormatter(formatter)
pear_logger.addHandler(fh)

def data_set(fname):
    df = pd.read_csv(fname,"\t")
    data = df.rename(columns={'时间':'time','指标':'indicator','数值':'value'})
    pivoted = data.pivot('indicator','time','value')
    #delete empty data
    cleaned_data = pivoted.dropna(axis=0)
    indicators = cleaned_data.index
    indicators = indicators.tolist()

    pear_logger.info("selected indicators:" + " ".join(indicators))

    indexs = []
    for i in range(len(cleaned_data)-1):
        for j in range(i+1,len(cleaned_data)):
            indexs.append([indicators[i],indicators[j]])

    datas = []
    for x in indexs:
        datas.append(cleaned_data.loc[[x[0],x[1]],:].values)

    return datas,indexs

def sd_pearson(fname,result_name):
    result_dict = []
    cl_data,indi_lists = data_set(fname)

    for i in range(len(cl_data)):
        values = cl_data[i]
        indi_list = indi_lists[i]
        #描述统计信息
        descrip_statis = indi_list

        #均值
        mean = values.mean(axis=1)

        #标准差
        # std  = values.std(axis=1)
        # mean_list = mean.tolist()
        # std_list = std.tolist()
        # mean_list = map(lambda x:str(round(x,4)),mean_list)
        # std_list  = map(lambda x:str(round(x,4)),std_list)
        mean = mean.reshape(2,1)

        diff = values - mean
        diff_quadratic = np.square(diff)
        #皮尔森系数
        pearson_corr = sum(diff[0]*diff[1])/np.sqrt(sum(diff_quadratic[0])*sum(diff_quadratic[1]))
        x = descrip_statis
        x.append(round(pearson_corr,6))
        result_dict.append(x)
    result_dict = pd.DataFrame(result_dict, columns=["指标1","指标2","皮尔森系数"])
    
    if result_name:
        try:
            result_dict.to_csv(result_name,float_format='%.4f',index = False, header=True, sep="\t")
        except Exception,e:
            print e
            pear_logger.info("result save error")
    return result_dict


if __name__=="__main__":
    optparser = OptionParser()
    optparser.add_option('-f', '--inputFile',
                         dest='input',
                         help='filename containing csv convert from rec',
                         default=None)
    (options, args) = optparser.parse_args()
    #inFile = None
    
    if options.input is None:
            inFile = sys.stdin
            #inFile = "INTEGRATED-DATASET.csv"
    elif options.input is not None:
            inFile = options.input
    else:
            print 'No dataset filename specified, system with exit\n'
            sys.exit('System will exit')

    #inFile = "Pearson.txt"
    full_name = os.path.realpath(inFile)
    pos = full_name.find(".txt")
    result_name = full_name[:pos] + "_result.txt"
    sd_pearson(full_name,result_name)
