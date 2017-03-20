# -*- coding:utf-8 -*-
'''
Description : Check the data input 
require     : windows Anaconda-2.3.0
author      : qiangu_fang@163.com
usage  $python sd_ARIMA.py  -f table.txt --log 1  -p 0 -d 0 -q 0 -x 3 
'''

import os
import sys
import logging
from optparse import OptionParser
from sd_adf import data_set

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import acf,pacf
from statsmodels.tsa.arima_model import ARMA

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PAR_DIR = os.path.dirname(BASE_DIR)

arima_logger = logging.getLogger('SD_API.Method.ARIMA')
arima_logger.setLevel(logging.INFO)
fh = logging.FileHandler(PAR_DIR + os.path.sep + "LOG" + os.path.sep + "SD_ARIMA.log")
fh.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
fh.setFormatter(formatter)
arima_logger.addHandler(fh)


def model_test(dta,p,q):
    if q or p:
        arma_mod = ARMA(dta,(p,d,q)).fit(disp=-1,method='mle')
        summary = (arma_mod.summary2(alpha=.05, float_format="%.8f"))

    else:
        (p, q) = (sm.tsa.arma_order_select_ic(dta,max_ar=6,max_ma=5,ic='aic')['aic_min_order'])
        arma_mod = ARMA(dta,(p,d,q)).fit(disp=-1,method='mle')
        summary = (arma_mod.summary2(alpha=.05, float_format="%.8f"))   

    resid = arma_mod.resid
    y = 1.96/np.sqrt(len(dta))

    r,rac,Q = sm.tsa.acf(resid.values.squeeze(), qstat=True)
    prac = pacf(resid.values.squeeze())
    data = np.c_[range(1,len(r)), r[1:],rac,prac[1:len(rac)+1],Q]

    table = pd.DataFrame(data, columns=['lag', "AC","Q", "PAC", "Prob(>Q)"])
    table.set_index('lag')
    
    return(summary,resid,(p,q), table, y)

def forecasts(data, l, p, d, q, x, last, diff_results):

    arma_model = sm.tsa.ARMA(data,(p,q)).fit(disp=-1,maxiter=100)

    predict_data = arma_model.predict(start=str(2001), end=str(len(data)+2000+x), dynamic = False)

    
    if d == 0:
        result_in = []
        if l:
            for i in range(len(predict_data)):
                result_in.append(np.exp(predict_data[i]))
        else:
            for i in range(len(predict_data)):
                result_in.append(predict_data[i])
        return (predict_data, result_in)

    elif d == 1:
        result_in = []
        predict_data = list(predict_data)       
        predict_in_sample = predict_data[0:len(data)]

        if l:
            for i in range(len(predict_in_sample)):
                result_in.append(np.exp(predict_in_sample[i] + diff_results["log"][i]))
        else:
            for i in range(len(predict_in_sample)):
                result_in.append(predict_in_sample[i] + diff_results["original"][i])

    elif d == 2:
        result_in = []
        predict_data = list(predict_data)
        predict_in_sample = predict_data[0:len(data)]

        if l:
            for i in range(len(predict_in_sample)):
                result_in.append(np.exp(predict_in_sample[i] + diff_results["diff1"][i] + diff_results["log"][i+1]))
        else:
            for i in range(len(predict_in_sample)):
                result_in.append(predict_in_sample[i] + diff_results["diff1"][i] + diff_results["original"][i+1])
    else:
        result_in = []
    if x:
        predict_out_of_sample = arma_model.forecast(x)[0]
        predict_out_of_sample = pd.Series(predict_out_of_sample)
        list1 = list(predict_out_of_sample)
        lists = []

        temp = 0
        if last:
            for i in last:
                temp += i

        for i in range(len(list1)):
            temp += list1[i]
            lists.append(temp)

        result = []
        if l: 
            for i in lists:
                result.append(np.exp(diff_results["log"][-1]+i))
        else:
            for i in lists:
                result.append(diff_results["original"][-1]+i)
    else:
        result = []

    return(predict_data, result_in+result)

 
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

    optparser.add_option('-p', '--acf_lags',
                         dest='acf',
                         help='Lags Used When acf Checking',
                         default=0,
                         type='int')

    optparser.add_option('-q', '--pacf_lags',
                         dest='pacf',
                         help='Lags Used When pacf Checking',
                         default=0,
                         type='int')

    optparser.add_option('-x', '--periods',
                         dest='periods',
                         help='Number of forecast periods',
                         default=0,
                         type='int')

    (options, args) = optparser.parse_args()
    
    if options.input is None:
            inFile = sys.stdin
            #inFile = 'PCApython2015113092218.txt'
    elif options.input is not None:
            inFile = options.input
    else:
            print('No dataset filename specified, system with exit\n')
            sys.exit('System will exit')

    (l, d, p, q, x) = (options.if_log, options.diff_time, options.acf, options.pacf, options.periods)

    arima_logger.info("Strating data set(log/diff)")
    (data,last,diff_results,diff_last,table1) = data_set(inFile, d, l)
    length = len(data)

    data = pd.Series(data)
    data.index = pd.Index(sm.tsa.datetools.dates_from_range(str(2000+1),str(length+2000)))

    arima_logger.info("Get model's Summary and resid table")
    (summary, resid, (p,q), table, y)= model_test(data,p,q)

   
    predict, cover = forecasts(data, l, p, d, q, x, last, diff_results)
    output = {
              "original":diff_results["original"],
              "predict":predict,
              "restore":cover
    }
    output["original"] = diff_results["original"] + [0]*x

    if d >2:
        output["restore"] = [0] * len(diff_results["original"])

    if d:
        output["restore"] = [0]*d + cover
        output["predict"] = [0]*d + predict

    Result = pd.DataFrame(output, columns = ["original","predict","restore"])

    full_name = os.path.realpath(inFile)
    pos = full_name.find(".txt")
    result_name = full_name[:pos] + "_result.txt"

    arima_logger.info("Save data to file")
    Result.to_csv(result_name,index = False, float_format = "%.4f",sep="\t")

    f = open(result_name, "a")
    f.write("######\n")
    f.close()
    table1.to_csv(result_name,index = False, mode = "a",float_format = "%.4f",sep="\t")
    f = open(result_name, "a")

    f = open(result_name, "a")
    f.write("######\n")
    f.close()
    table.to_csv(result_name,index = False, mode = "a",float_format = "%.4f",sep="\t")
    f = open(result_name, "a")
    f.write("######\n")
    f.write(str(p)+"\t"+str(q)+"\n")
    f.close()

    a = summary.tables
    z={
    "summary_para":[a[0][0].values[0], a[0][2].values[3],a[0][2].values[1],
                    a[0][0].values[3],a[0][0].values[7],a[0][2].values[0]],
    "summary_values":[a[0][1].values[0], a[0][3].values[3],a[0][3].values[1],
                    a[0][1].values[3],a[0][1].values[7],a[0][3].values[0]]
    }
    z = pd.DataFrame(z)
    z.to_csv(result_name,index = False, header=None,mode = "a", sep="\t")
    
    f = open(result_name, "a") 
    f.write("######\n")
    f.close()
    a[1].to_csv(result_name, mode = "a", float_format = "%.6f", sep="\t")

