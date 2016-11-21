import networkx as nx
import pandas as pd
from pandas import Series
from optparse import OptionParser

# para_list = [[1,2,16],[1,3,13],[3,2,4],[2,4,12],[3,5,14],
#              [4,3,9],[5,4,7],[5,6,4],[4,6,20]]
def data_set(fname):
	df = pd.read_csv(fname,names=["Edge","Start_point","End_point","Capacity"],skiprows=[0],sep="\t")
	data = (pd.DataFrame(df)).set_index("Edge")
	return data.values

def max_flow(inFile,source,end):
    para_list = data_set(inFile)
    # print(para_list)

    G = nx.DiGraph()

    for i in para_list:
    	print(i)
        G.add_edge(str(i[0]),str(i[1]), capacity=i[2])

    flow_value, flow_dict = nx.maximum_flow(G, source, end)
    print(flow_value,flow_dict)

def main(inFile,method,source,end):
	if method == "max_flow":
		max_flow(inFile,source,end)
	else:print("error")


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
    main(inFile,m,s,e)