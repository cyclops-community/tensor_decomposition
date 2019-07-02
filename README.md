## Python Tensor Decomposition Algorithms

### TODOs:

- [ ] Documentation based on sphinx conventions

- [ ] Test cases development

- [ ] Randomized CP/Tucker implementation

- [ ] [Multigrid methods implementation](https://arxiv.org/pdf/1111.6091.pdf)

- [ ] Tensor train implementation

- [ ] Gradient Based algorithms 

- [ ] [Nesterov Acceleration for ALS](https://arxiv.org/pdf/1810.05846.pdf)

- [ ] Constrained decomposition (nonnegative) 

- [ ] Postprocessing tools (save/load parameters, eigenvalue visualization of perturbations)

- [ ] Implement sliced als version

- [ ] NLP based real tensors for experiments


### Note:

Run
```
pip install -r requirements.txt
```
to install necessary packages. 

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


Run 

```
python test_ALS3.py --s 64 --R 10 --r 10 --num-iter 10 --num-lowr-init-iter 2 --sp-fraction 1 --sp-updatelowrank 1 --sp-res 1 --run-naive 1 --run-lowrank 0 --num-slices 1
```
to execute a test case. 
