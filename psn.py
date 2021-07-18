from multiprocessing import pool
import random
import numpy as np
import matplotlib.pyplot as plt
import numpy as np


mutrate = 5/16000 # or 1 mutations every 10000th  iteration
# period  = 16000 # 1000 iterations
period = int(1e7)*16
poolsize   =   np    .random.poisson(         mutrate *period)
# indices    =   np    .random.randint(         low     =0, high=1000, size=poolsize)
# entries    =   random.sample(range  (1,16001),poolsize)
# mut_posns  = [*map   (lambda x: ((x%16)//4,(x%16)%4), entries)]



