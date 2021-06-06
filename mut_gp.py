import numpy as np

adj = np.array([
                        [2,2,2,2],
                        [2,2,2,2],
                        [2,2,2,2],
                        [2,2,2,2],
                    ], dtype=np.float64)

template = np.random.randint(-1,1,(4,4))
probs    = np.random.uniform(low=0, high=1, size=(4,4)).round(1)

def knockout(target:np.ndarray,template:np.ndarray,probs:np.ndarray)->np.ndarray:
    rows,cols =probs.shape
    for i in  range(rows):
        for j in range(cols):
            if probs[i,j] <0.5:
                target[i,j] = template[i,j]
    return target




