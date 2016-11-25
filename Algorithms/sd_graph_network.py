# -*- coding:utf-8 -*-
'''
Description : Check the data input 
require     : windows Anaconda-2.3.0
author      : qiangu_fang@163.com
usage  $ python sd_graph_network.py -f filename -m methon -s start_point -e end_point 
'''
import os
import sys
import logging
import networkx as nx
import pandas as pd
from pandas import Series
from optparse import OptionParser

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PAR_DIR = os.path.dirname(BASE_DIR)

graph_network_logger = logging.getLogger('SD_API.Method.sd_graph_network')
graph_network_logger.setLevel(logging.INFO)
fh = logging.FileHandler(PAR_DIR + os.path.sep + "LOG" + os.path.sep + "SD_Graph_Network.log")
fh.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
fh.setFormatter(formatter)
graph_network_logger.addHandler(fh)

def data_set(fname):
	df = pd.read_csv(fname,names=["Edge","Start_point","End_point","Capacity"],skiprows=[0],sep="\t")
	data = (pd.DataFrame(df)).set_index("Edge")
	return data.values

def max_flow(inFile,source,end):
    para_list = data_set(inFile)

    G = nx.DiGraph()

    for i in para_list:
        G.add_edge(str(i[0]),str(i[1]), capacity=i[2])

    flow_value, flow_dict = nx.maximum_flow(G, source, end)
    
    result=[]
    for x in flow_dict.keys():
    	if flow_dict[x]:
    	    for k,v in flow_dict[x].items():
    	        result.append([x,k,v])
    	else:continue

    Result = pd.DataFrame(result, columns = ["start","end","distance"])

    return flow_value,Result

def min_span_tree(inFile):
    para_list = data_set(inFile)

    G = nx.Graph()

    for i in para_list:
       G.add_edge(str(i[0]),str(i[1]),weight=i[2])
    
    T=nx.minimum_spanning_tree(G)
    result=[]
    flow_value=0

    for x in T.edges():
        result.append(list(x)+[(str(T.get_edge_data(x[0],x[1])["weight"]))])
        flow_value+=T.get_edge_data(x[0],x[1])["weight"]

    Result = pd.DataFrame(result, columns = ["start","end","distance"])

    return flow_value,Result

def shortest_path(inFile, source, end, direction):

    para_list = data_set(inFile)
    if direction:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    edges = []
    for i in para_list:
    	tmp = []
    	for j in i:
    		tmp += [str(j)]
    	tmp[-1] = int(tmp[-1])
    	edges.append(tmp)

    G.add_weighted_edges_from(edges)

    path = nx.dijkstra_path(G,source=source,target=end)

    path_length = nx.dijkstra_path_length(G,source,end)
    result = []

    for x in range(len(path)-1):
    	result.append([path[x],path[x+1]]+[str(nx.dijkstra_path_length(G,path[x],path[x+1]))])

    Result = pd.DataFrame(result, columns = ["start","end","distance"])
    
    return path_length,Result

def max_flow_min_cost(inFile,source,end):
    df = pd.read_csv(inFile,names=["Edge","Start_point","End_point","Capacity","Cost"],skiprows=[0],sep="\t")
    data = (pd.DataFrame(df)).set_index("Edge")

    edges = []
    for i in data.values:
        edges += [(str(i[0]),str(i[1]),{"capacity":i[2],"weight":i[3]})]

    G = nx.DiGraph()
    G.add_edges_from(edges)
    mincostFlow = nx.max_flow_min_cost(G, "1", "4")
    mincost = nx.cost_of_flow(G, mincostFlow)
    
    result = []
    for x in mincostFlow.keys():
    	if mincostFlow[x]:
    	    for k,v in mincostFlow[x].items():
    	        result.append([x,k,v,G.get_edge_data(x,k)["weight"]])
    	else:continue

    maxFlow = nx.maximum_flow(G, "1", "4")[0]

    Result = pd.DataFrame(result, columns = ["start","end","capacity","cost"])
    return (mincost,maxFlow),Result


    

def main(inFile,method,source,end,direction):
    if method == "max_flow":
        value, dicts = max_flow(inFile,source,end)

    elif method == "min_span_tree":
        value,dicts = min_span_tree(inFile)

    elif method == "shortest_path":
    	value,dicts = shortest_path(inFile,source,end,direction)

    elif method == "max_flow_min_cost":
    	# max_flow_min_cost(inFile,source,end)
    	value, dicts = max_flow_min_cost(inFile,source,end)

    else:graph_network_logger.info("Method not find")

    return value,dicts


if __name__ == "__main__":
    optparser = OptionParser()
    optparser.add_option('-f', '--inputFile',
                         dest='input',
                         help='filename containing csv convert from rec',
                         default=None)

    optparser.add_option('-m', '--method',
                         dest='method',
                         help='Graph Network analysis method',
                         default=None)


    optparser.add_option('-s', '--source',
                         dest='source',
                         help='Source Point',
                         default=None,
                         type="str")

    optparser.add_option('-e', '--end',
                         dest='end',
                         help='End Point',
                         default=None,
                         type="str")

    optparser.add_option('-d', '--direction',
                         dest='direction',
                         help='direction/no_direction graph',
                         default=None,
                         type="str")

    (options, args) = optparser.parse_args()
    
    if options.input is None:
            inFile = sys.stdin
    elif options.input is not None:
            inFile = options.input
    else:
            print 'No dataset filename specified, system with exit\n'
            sys.exit('System will exit')
    
    graph_network_logger.info("Start,getting parameters")
    m = options.method
    s = options.source
    e = options.end
    d = options.direction
    
    graph_network_logger.info("Computing...")
    value, Result = main(inFile,m,s,e,d)
    full_name = os.path.realpath(inFile)
    pos = full_name.find(".txt")
    result_name = full_name[:pos] + "_"+str(m)+"_result.txt"

    f = open(result_name, "w")
    if m == "max_flow_min_cost":
    	f.write(str(value[0])+"\t"+str(value[1])+"\n")
    else:
        f.write(str(value)+"\n")
    f.close()

    graph_network_logger.info("Saving data to file")
    
    Result.to_csv(result_name,index = False, header=None,mode = "a", sep="\t")