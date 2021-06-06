import pstats
import sys


file = sys.argv[1]
p    = pstats.Stats(file)
p.sort_stats('tottime').print_stats(20)