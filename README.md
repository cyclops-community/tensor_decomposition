## Python Tensor Decomposition Algorithms

This repository implements efficient numerical algorithm for Alternating Least Squares (ALS) in CP and Tucker decompositions, as well as fast Nonlinear Least Squares (NLS) for CP decomposition.

This repository implements everything in Python, and is compatible with both Numpy backend, which allows fast seaquential running, and [Cyclops Tensor Framework](https://github.com/cyclops-community/ctf) backend, which allows fast distributed parallism.

### Prerequisite

Run
```
pip install -r requirements.txt
```
to install necessary packages. 

### Tests cases
Run all tests with
```bash
# sudo pip install nose
nosetests -v tests/*.py
```

### Running CP/Tucker decomposition with ALS

Run 

```
python run_als.py -h
```
to see the existing input arguments and their functions.

Run 

```
python tests.py
```
to test some simple CP runs.

### Running CP decomposition with NLS

Run

```
python run_nls.py -h

```
to see existing input arguments and their functions with default values


### ALS/GN performance comparision

The comparison in contraction ALS and GN CG iteration is done using Contraction.py file.
Run

```
python Contraction.py -h
```

to see existing input arguments


### Convergence probability

The file Convprob.py can be used to compare the convergence probability of different methods. Run

```
python Convprob.py -h
```
to see input arguments with their functions and default values

## Visualization with Visdom

For now visdom can fetch all the csv files following the particular format and plot them.

Go to the Visdom folder then execute the following commands:
```
visdom -port XXXXX

python visdom_pull_server.py -port XXXXX
```
