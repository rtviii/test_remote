import pickle
import pprint


filename = 'here/PERSIST_expexp1_itn1000_sp1_LI0.1' + '.pickle'
with open(filename, 'rb') as filehandler: 
    reloaded_array = pickle.load(filehandler)

pprint (reloaded_array)