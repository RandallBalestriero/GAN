In each folder (with name describing the experiment) one can find
- a python file with an explicit name. This file performs the
  experiment with possibly some input arguments and save the results.
- a bash file that is used to launch multiple runs simultaneously in
  parallel
- possibly a reader file that loads the generated fiels and generates the
  figures/plots/tables.

For the dropout on the circle experiment and the boxplot of the dimensions, the codes are as standalone and can be run directly. Note that for almost
all the codes, the library used is JAX.
