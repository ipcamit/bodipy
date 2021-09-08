Bayesian Search For Substituted BODIPY
======================================

`get_target_bodipy.py` is a simple Python program to explore the application of Bayesian optimization for BODIPY chemical space search.
It utilizes Kernel Ridge Regression based ML model to evaluate the S<sub>0</sub>&#8594;S<sub>1</sub> excitation gap.<sup>1</sup>
The Bayesian optimization is performed using Expected Improvement, with Gaussian Process based surrogate. The Gaussian Process model is built using `gaussian_process` module of `scikit-learn`. 

It can be run using following command line
```
$ python3 get_target_bodipy.py <target(eV)>
```

Additional parameters can be sought using `--help` argument. Given below are all possible flags.

---
| Flag/ arg | Description | Default [range] | Compulsory |
|:----:|:-----------:|:-------:|:----------:|
|[target]| Target S0->S1 value, in eV. Positional argument, non optional.| - | &#10003; |
| --group, -g | # of substitutions in target BODIPY. | 2 [2, 7]|&#x2717;|
|  --data -d | Location of datafiles to be used in KRR ML, contains descriptor and coefficients.| `./data`|&#x2717;|
|  --restart, -r | # of evaluations for single EI evaluation. More evaluations give more robust minima, with higher computation cost. | 5 [1, &#8734;] | &#x2717;|
| --exploration, -x | Exploitation vs Exploration parameter | 0.01 (0,100)| &#x2717;|
| --seed, -s | Number of initial evaluations to build Gaussian Process surrogate. More evaluations might help converging faster. | 5 [1, &#8734;] | &#x2717;|
| --iter, -i | Maximum number of iterations. | 200 [1, &#8734;] | &#x2717;|
---

Once run, it will run for `iter` times and print successive improvements towards obtaining target molecule. An example run is shown below:
```
$ python3 get_target_bodipy.py 2.7

Searching for 2D BODIPY near 2.700000 eV
Reading ML model from ./data
Iterations 200; Initial evaluations 5
Bayesian opt. parameters:
 Exploration/Exploitation param: 0.010000; Eval. per EI: 5
=================================================================
ITER    POS            GROUPS             S0S1(eV)        Target
=================================================================
0       1 6             28 27           3.337201        2.700000
1       3 5             29 30           3.184931        2.700000
2       5 6             27 22           3.183506        2.700000
13      4 5             30 25           2.999981        2.700000
18      5 7             23 19           2.952890        2.700000
38      2 5             15 25           2.866237        2.700000
83      5 4             34 6            2.709659        2.700000
=================================================================
```

Requirements:
1. Python3
2. Numpy
3. Scipy (scipy.optimize.minimize for iter minimization)
4. Scikit-learn (for Gaussian Process)
5. [MOPAC](http://openmopac.net/) for calculating minimum energy geometry at the PM7 level
6. [OBabel](http://openbabel.org/wiki/Main_Page) for file conversion
7. [QML](https://www.qmlcode.org/) for calculating the SLATM descriptor using the PM7 geometry

Please find complimentary web interface at [`https://moldis.tifrh.res.in/db/bodipy`](https://moldis.tifrh.res.in/db/bodipy).

![](https://moldis.tifrh.res.in/index.html)
<a href="https://moldis.tifrh.res.in/index.html">
<img src="https://moldis.tifrh.res.in/Images/MolDis.png"  height="100">
</a>





Reference:
[1] _Data-Driven Modeling of S0 -> S1 Transition in the Chemical Space of BODIPYs: High-Throughput Computation, Machine Learning Modeling and Inverse Design_,       
    Amit Gupta, Sabyasachi Chakraborty, Debashree Ghosh, Raghunathan Ramakrishnan                
    submitted (2021) arxiv


https://moldis-group.github.io/BODIPYs/
