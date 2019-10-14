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


### ALS/NLS performance comparision

### Convergence probability


## Visualization with Visdom

For now visdom can fetch all the csv files following the particular format and plot them.

Go to the Visdom folder then execute the following commands:
```
visdom -port XXXXX

python visdom_pull_server.py -port XXXXX
```
