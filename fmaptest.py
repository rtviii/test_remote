import math
import pprint
import numpy as np
amplitude = 1
std       = 1
phenotype = np.array( [1,2,3,1] )
mean      = np.array([1,1,1,1])

amplitude * math.exp(-( 
    np.sum(((phenotype - mean)**2)/(2*std**2)
    )))


presum   = ((phenotype - mean)**2)/(2*std**2)


inner = ((phenotype - mean)**2)/(2*std**2)
print("inner:", inner)
print("Pre sum:" , presum)


exponent = -(np.sum(
                    ((phenotype - mean)**2)
                    /(2*std**2)
                    )

                )
print("Exponent" , exponent)

res = amplitude * math.exp(exponent)

print(res)