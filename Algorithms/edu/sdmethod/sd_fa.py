#!/usr/bin/env  Python
#-*- coding=utf-8 -*-
'''
Description : statistic indicator area ranks by factor analysis
require     : windows Anaconda-2.3.0
author      : shizhongxian@126.com
usage  $python sd_fa.py  -f table.txt -c 2 
'''

import pandas as pd
import numpy as np
from sklearn.decomposition import FactorAnalysis
from sklearn import preprocessing
from scipy import linalg
from optparse import OptionParser
import sys
import logging
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PAR_DIR = os.path.dirname(BASE_DIR)

fa_logger = logging.getLogger('SD_API.Method.FA')
fa_logger.setLevel(logging.INFO)
fh = logging.FileHandler(PAR_DIR + os.path.sep + "LOG" + os.path.sep + "SD_FA.log")
fh.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
fh.setFormatter(formatter)
fa_logger.addHandler(fh)

def get_factor_weight(data,n_components):
    '''
    get factor weight W 
    '''
    data = preprocessing.scale(data)
    C = np.cov(np.transpose(data))  
    U, s, V = linalg.svd(C, full_matrices=False)  
    eig = s[:n_components]
    eig_v =  U[:,:n_components]
    A = eig_v * np.sqrt(eig)
    A = A**2
    contri = np.sum(A,axis=0)
    contri_ratio = [tmp/np.sum(contri) for tmp in contri ]
    return np.array(contri_ratio)  

def data_set(fname):
    '''
        把数据转化为pandas可处理的二维表格,行表示地区，列表示指标
        删除包含空值的行
    '''
    df = pd.read_csv(fname,"\t")
    #data = df.rename(columns={'月份顺序排序':'m_order','正式指标':'indicator','正式数值':'value'})
    data = df.rename(columns={'地区':'area','正式指标':'indicator','正式数值':'value'})
    pivoted = data.pivot('area','indicator','value')
    #删除空值行,生成清洗后的数据
    cleaned_data = pivoted.dropna(axis=0)
    #原始地区列表
    areas = pivoted.index
    area_list = areas.tolist()
    #原始指标列表
    indicators = pivoted.columns
    indi_list = indicators.tolist()
    #清洗后的地区列表
    cleaned_areas = cleaned_data.index
    cleaned_area_list = cleaned_areas.tolist()
    fa_logger.info("selected area:"+" ".join(area_list))
    fa_logger.info("cleaned  area:"+" ".join(cleaned_area_list))
    deleted_areas = set(area_list) - set(cleaned_area_list)
    if len(deleted_areas) >= 1:
        fa_logger.info("deleted areas:"+" ".join(deleted_areas))
    fa_logger.info("selected indi:"+" ".join(indi_list))
    return cleaned_data,cleaned_area_list

def sd_fa(fname,components,result_name):
    '''
    FA计算
    '''
    fa_logger.info("start sd_fa")
    #Web API 返回结果
    result_dict = {}
    cl_data,area_list = data_set(fname)
    values = cl_data.values
    fa = FactorAnalysis(n_components=components)
    #数据标准化
    values = preprocessing.scale(values)
    try:
        fa.fit(values)
    except Exception,e:
            fa_logger.error("factor analysis fit error")
            sys.exit()
    #print(fa.n_components)
    #print(fa.components_)
    contri_ration = get_factor_weight(values, components)
    
    scores = np.dot(fa.transform(values),contri_ration.T)
    #print scores
    scores_list = scores.tolist()
    result_col =  cl_data.columns
    result_col.name = "指标"
    result_idx = ["因子"+str(i+1) for i in range(components)]
    #输出因子权重
    result_data = pd.DataFrame(fa.components_,columns=result_col,index=result_idx)
    fact_scores = result_data.values.tolist()
    table_1 = []
    table_1.append(['指标']+list(result_col.values))
    for i in range(components):
        table_1.append([result_idx[i]] + fact_scores[i])
    result_dict["table_1"] = table_1
    #result_data = result_data.astype(float)
    #result_data.to_csv("fa_result.txt",sep="\t",float_format='%8.4f')
    if result_name:
        result_data.to_csv(result_name,sep="\t",float_format='%8.4f')
    
    if result_name:
        fout = open(result_name,"a")
        fout.write("\n===============================\n")
    table_2 = []
    table_2.append(['因子','权重'])
    for i in range(len(result_idx)):
        if result_name:
            fout.write("%s,\t %.3f \n" %(result_idx[i],contri_ration[i]))
        table_2.append(["%s" % result_idx[i],"%.3f" % contri_ration[i]])
    if result_name:
        fout.write("\n===============================\n")
    result_dict["table_2"] = table_2
    
    rank_1 = []
    rank_1.append(['地区','综合得分'])
    if len(area_list) == len(scores):
        area_scores_list = zip(area_list,scores_list)
        area_scores_list = sorted(area_scores_list,key=lambda d:d[1],reverse=True)
        for area_score in area_scores_list:
            key,value = area_score
            if result_name:
                fout.write("%s,%.5f \n" % (key,value))
            rank_1.append(["%s" % key ,"%.5f" % value ])
        result_dict["rank_1"] = rank_1
    else:
        print "caculated result not equal to area_list"
        fa_logger.error("caculated result not equal to area_list")
        sys.exit()
    if result_name:
        fout.close()
        print "save to",result_name
        fa_logger.info("save to"+result_name)
    return result_dict
    
if __name__ == "__main__":
    optparser = OptionParser()
    optparser.add_option('-f', '--inputFile',
                         dest='input',
                         help='filename containing csv convert from rec',
                         default=None)
    optparser.add_option('-c', '--components',
                         dest='components',
                         help='factor analysis factors',
                         default=2,
                         type='int')
    
    (options, args) = optparser.parse_args()
    
    inFile = None
    if options.input is None:
            inFile = sys.stdin
            #inFile = "INTEGRATED-DATASET.csv"
    elif options.input is not None:
            inFile = options.input
    else:
            print 'No dataset filename specified, system with exit\n'
            sys.exit('System will exit')
    components = options.components
    inFile = "FApython20151019145628.txt"
    #components = 1
    full_name = os.path.realpath(inFile)
    pos = full_name.find(".txt")
    result_name = full_name[:pos] + "_result.txt"
    result_dict = sd_fa(inFile,components,result_name=result_name)
    pass
    
    
    
    
    
    


