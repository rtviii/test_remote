import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


adjacency = np.array([
                        [1,1,0,5],
                        [1,9,0,0],
                        [0,3,1,1],
                        [0,0,1,1],
                    ], dtype=np.float64)
                
print(adjacency)

def adj_2_graph(adjacency_matrix:np.ndarray):
    genes  = []
    edges   = []
    traits = [( 'T1',{"color":"blue"} ),( 'T2',{"color":"blue"} ),( 'T3',{"color":"blue"} ),( 'T4',{"color":"blue"} )]
    for col in range(len( adjacency_matrix )):
        genes.append(( 'gene{}'.format(col),{"color":"green"}))
        for index,value in enumerate(adjacency_matrix[:,col]):
            print(value,index)
            if value == 1.0:
                edges.append(('gene{}'.format(col), traits[index][0]))

    gr    = nx.Graph()
    gr.add_nodes_from([*genes, *traits])
    gr.add_edges_from(edges)
    # gr.add_edges_from(edges)
    nx.draw(gr, node_size=500, with_labels=True,)
    plt.show()

adj_2_graph(adjacency)