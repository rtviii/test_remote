import numpy as np
from pprint import pprint


x =np.array([
                        [1,1,0,0],
                        [1,1,0,0],
                        [0,0,1,1],
                        [0,0,1,1],
                    ], dtype=np.float64) * np.array([-1,1,-1,1])[:,None]

pprint(x)