#### Let's optimize the fuck out of  this

- Integrate the BD-process into the population class. (Got only marginal speedups)

- moved poplen and fitness calculations to Population class. cut down from 535sec/10k to 13sec/10k. Embarassing.

- sample average fitness only every N iterations: 

    births and deaths of single individuals affect the measure only slightly while calculating it is an O(2n) operation -- plucking fitnesses and summing up

- rewrite the whole thing in cpp?

# Time profiling



---------------

## March 18:

n_traits was left thardcoded in mutation_duplicate. Scrap everything earlier and use the new Individ_T.py

---------------

## May 2nd

keeping population constant and just swapping out an individual out for a new(possibly mutated one?)


## May 14

-population = cosntant
-aiming for 10**7s iterations
-picking by frequency(buckets of mutated individuals)
-mutating architecture

1. profile time