# Lowrank-ALS

### Note:
Currectly the code is only tested under Python3. 

Run
```
pip install -r requirements.txt
```
to install necessary packages. 

Run 

```
python cpals.py -h
```
to see the existing input arguments and their functions.

Run 

```
python cptests.py
```
to test some simple CP runs.


Run 

```
python test_ALS3.py --s 64 --R 10 --r 10 --num-iter 10 --num-lowr-init-iter 2 --sp-fraction 1 --sp-updatelowrank 1 --sp-res 1 --run-naive 1 --run-lowrank 0 --num-slices 1
```
to execute a test case. 
