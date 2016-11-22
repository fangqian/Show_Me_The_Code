# -*- coding:utf-8 -*-
'''
Description : Check the data input 
require     : windows Anaconda-2.3.0
author      : qiangu_fang@163.com
usage  $ python sd_graph_network.py -f filename -m methon -s start_point -e end_point 
'''
import os
import networkx as nx
import pandas as pd
from pandas import Series
from optparse import OptionParser

def data_set(fname):
	df = pd.read_csv(fname,names=["Edge","Start_point","End_point","Capacity"],skiprows=[0],sep="\t")
	data = (pd.DataFrame(df)).set_index("Edge")
	return data.values

def max_flow(inFile,source,end):
    para_list = data_set(inFile)

    G = nx.DiGraph()

    for i in para_list:
    	# print(i)
        G.add_edge(str(i[0]),str(i[1]), capacity=i[2])

    flow_value, flow_dict = nx.maximum_flow(G, source, end)
    
    result=[]
    for x in flow_dict.keys():
    	if flow_dict[x]:
    	    for k,v in flow_dict[x].items():
    	        result.append([x,k,v])
    	else:continue
    return flow_value,result

def min_span_tree(inFile):
    para_list = data_set(inFile)

    G = nx.Graph()

    for i in para_list:
        G.add_edge(str(i[0]),str(i[1]),weight=i[2])
        T=nx.minimum_spanning_tree(G)
        result=[]
        flow_value=0

    for x in T.edges():
        result.append(list(x)+list(str(T.get_edge_data(x[0],x[1])["weight"])))
        flow_value+=T.get_edge_data(x[0],x[1])["weight"]

    return flow_value,result
    
def main(inFile,method,source,end):
    if method == "max_flow":
        value, dicts = max_flow(inFile,source,end)

    elif method == "min_span_tree":
        value,dicts = min_span_tree(inFile)

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

    (options, args) = optparser.parse_args()
    
    if options.input is None:
            inFile = sys.stdin
    elif options.input is not None:
            inFile = options.input
    else:
            print 'No dataset filename specified, system with exit\n'
            sys.exit('System will exit')

    m = options.method
    s = options.source
    e = options.end
    # n = options.node

    value, dicts = main(inFile,m,s,e)
    full_name = os.path.realpath(inFile)
    pos = full_name.find(".txt")
    result_name = full_name[:pos] + str(m)+"_result.txt"

    f = open(result_name, "w")
    f.write(str(value))
    f.write("\n")
    f.close()
    print(dicts)
    Result = pd.DataFrame(dicts, columns = ["start","end","distance"])
    Result.to_csv(result_name,index = False, header=None,mode = "a", sep="\t")


#     import networkx as nx

# G = nx.Graph()

# G.add_weighted_edges_from([(1,2,4),(2,3,2),(1,3,6),(3,4,3),(2,4,1)])

# p = nx.shortest_path(G,source=1,target=4)
# # print(dir(p))
# x = path=nx.all_pairs_shortest_path(G) 
# # print(dir(x))
# q = nx.shortest_path_length(G,source=1,target=4)
# y = nx.shortest_path_length(G,source=1,target=3)

# print(nx.dijkstra_path_length(G,1,4))

# print(x[1][2])
# print(p,x[1][4],q,y)
