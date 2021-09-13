Inverse Design of Substituted BODIPY
======================================
We provide python codes to inverse design BODIPY molecules, as discussed in Ref-1. 

DesignBodipy_Bayes.py can be used to design molecules using Bayesian optimization based on Gaussian process regression.

DesignBodipy_GA.py can be used for genetic algoritm (GA) optimization

Both programs use a trained kernel ridge regression machine learning (KRR-ML) model to evaluate the S<sub>0</sub>&#8594;S<sub>1</sub> excitation energy.

## Example run: Bayesian Optimization
```
$ python3 DesignBodipy_Bayes.py <target(eV)>
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
$ python3 DesignBodipy_Bayes.py 2.7
```
Screenshot of output
```
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


## Example run: GA Optimization
```
$ python3 DesignBodipy_GA.py 2.7
```
Screenshot of output
```
Searching for 7D BODIPY near 2.700000 eV
Reading ML model from ./data
Generations 200; Parent population 20
Starting Genetic optization
Starting population estimation
Calculating parent: 20   
Current Gen 0, Median: 2.985662  Best: 2.767993
Best Groups [19.0, 18.0, 20.0, 34.0, 45.0, 7.0, 30.0] ; Pos [0, 1, 2, 3, 4, 5, 6]
Evaluating: 10 of 10   
Current Gen 1, Median: 2.926014  Best: 2.767993
Best Groups [19.0, 18.0, 20.0, 34.0, 45.0, 7.0, 30.0] ; Pos [0, 1, 2, 3, 4, 5, 6]
Evaluating: 10 of 10   
Current Gen 2, Median: 2.885984  Best: 2.708469
Best Groups [19.0, 40.0, 14.0, 43.0, 23.0, 7.0, 11.0] ; Pos [0, 1, 2, 3, 4, 5, 6]
Evaluating: 10 of 10   
Current Gen 3, Median: 2.859153  Best: 2.708469
Best Groups [19.0, 40.0, 14.0, 43.0, 23.0, 7.0, 11.0] ; Pos [0, 1, 2, 3, 4, 5, 6]
...

```


## Requirements:
1. Python3.6 and above
2. Numpy
3. Scipy (scipy.optimize.minimize for iter minimization)
4. Scikit-learn (for Gaussian Process)
5. [MOPAC](http://openmopac.net/) for calculating minimum energy geometry at the PM7 level
6. [OBabel](http://openbabel.org/wiki/Main_Page) for file conversion
7. [QML](https://www.qmlcode.org/) for calculating the SLATM descriptor using the PM7 geometry

## Relevant resources
A publicly accessible web interface hosting a trained machine learning (ML) model to predict S<sub>0</sub>  → S<sub>1</sub> excitation energy of BODIPYs is available at [`https://moldis.tifrh.res.in/db/bodipy`](https://moldis.tifrh.res.in/db/bodipy).  


![](https://moldis.tifrh.res.in/index.html)
<a href="https://moldis.tifrh.res.in/index.html">
<img src="MolDis.png"  height="100">
</a>



## Reference:
[1] _Data-Driven Modeling of S0 -> S1 Transition in the Chemical Space of BODIPYs: High-Throughput Computation, Machine Learning Modeling and Inverse Design_,       
    Amit Gupta, Sabyasachi Chakraborty, Debashree Ghosh, Raghunathan Ramakrishnan                
    submitted (2021) arxiv                  
    Dataset: [https://moldis-group.github.io/BODIPYs/](https://doi.org/10.6084/m9.figshare.16529214.v1)      
    Dataset DOI: [10.6084/m9.figshare.16529214.v1](https://doi.org/10.6084/m9.figshare.16529214.v1)        


