# -*- coding:utf-8 -*-
'''
Description : Check the data input 
require     : windows Anaconda-2.3.0
author      : qiangu_fang@163.com
usage  $python sd_adf.py  -f table.txt --log 1 -d 1 --lags 1 
'''
import os
import sys
import logging
import numpy as np
import pandas as pd
from optparse import OptionParser
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import acf,pacf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PAR_DIR = os.path.dirname(BASE_DIR)

adf_logger = logging.getLogger('SD_API.Method.ADF')
adf_logger.setLevel(logging.INFO)
fh = logging.FileHandler(PAR_DIR + os.path.sep + "LOG" + os.path.sep + "SD_ADF.log")
fh.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
fh.setFormatter(formatter)
adf_logger.addHandler(fh)

def data_set(fname, d, l):
    """
    提取需要检测的数据，按要求(参数)进行处理(如取对数、差分)，返回数据处理
    过程中的结果(包括中间步骤的处理结果)
    """
    df = pd.read_csv(fname,names=["year","value"], skiprows=[0],sep="\t")
    data = df.dropna(axis=0)
    value = [round(x, 5) for x in data["value"]]
    dta = pd.Series(value)
    diff_results = {"original":list(dta)}

    if l:
        dta = np.log(dta)
        diff_results["log"] = list(dta)

    diff_result = []

    if d >= 1:
            diff_results["diff1"] = list(dta.diff(1).dropna(axis=0))

    if d > 0:
        for i in range(d):
            dta = dta.diff(1)
            diff_result.append(list(dta))

    last = []
    if diff_result:
        for x in diff_result:
            last.append(x[-1])
    
    if d > 0:
        last.pop()
        
    dta = dta.dropna(axis=0)

    r,rac,Q = sm.tsa.acf(dta, qstat=True)
    prac = pacf(dta)
    table_data = np.c_[range(1,len(r)), r[1:],rac,prac[1:len(rac)+1],Q]

    table = pd.DataFrame(table_data, columns=['lag', "AC","Q", "PAC", "Prob(>Q)"])
    table.set_index('lag')

    dta = np.array(dta)

    if d >0:
        return (dta, last, diff_results, diff_result[-1],table)
    else:
        return (dta, last, diff_results, diff_result, table)

    

def sd_adf(inFile, d, l, lags):
    
    adf_logger.info("Start data_set(log|diff)")
    (data,last,diff_results,year,table)=data_set(inFile, d, l)
    
    adf_logger.info("Data_set End, Start ADF Check")
    t=sm.tsa.stattools.adfuller(data, maxlag=lags)
    output=pd.DataFrame(index=['Test Statistic Value', "p-value", "Lags Used", "Number of Observations Used","Critical Value(1%)","Critical Value(5%)","Critical Value(10%)"],columns=['value'])

    output['value']['Test Statistic Value'] = t[0]
    output['value']['p-value'] = t[1]
    output['value']['Lags Used'] = t[2]
    output['value']['Number of Observations Used'] = t[3]
    output['value']['Critical Value(1%)'] = t[4]['1%']
    output['value']['Critical Value(5%)'] = t[4]['5%']
    output['value']['Critical Value(10%)'] = t[4]['10%']

    adf_logger.info("Check Over")

    return(output)

# def adf_main():
    
if __name__ == "__main__":
    optparser = OptionParser()
    optparser.add_option('-f', '--inputFile',
                         dest='input',
                         help='filename containing csv convert from rec',
                         default=None)

    optparser.add_option('--log', '--logarithm',
                         dest='if_log',
                         help='Whether take the logarithm',
                         default=0,
                         type='int')

    optparser.add_option('-d', '--difference',
                         dest='diff_time',
                         help='Number of difference, default 0',
                         default=0,
                         type='int')

    optparser.add_option('--lags', '--lagsUsed',
                         dest='lags',
                         help='Lags Used When adf Checking',
                         default=None,
                         type='int')

    (options, args) = optparser.parse_args()
    
    if options.input is None:
            inFile = sys.stdin
            #inFile = 'PCApython2015113092218.txt'
    elif options.input is not None:
            inFile = options.input
    else:
            print 'No dataset filename specified, system with exit\n'
            sys.exit('System will exit')

    l = options.if_log
    d = options.diff_time
    lags = options.lags


    #data,last, diff_results, year = data_set(inFile, d, l)

    result_dict = sd_adf(inFile, d, l, lags)

    full_name = os.path.realpath(inFile)
    pos = full_name.find(".txt")
    result_name = full_name[:pos] + "_result.txt"

    result_dict.to_csv(result_name,float_format='%8.4f')

