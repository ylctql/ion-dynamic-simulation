# ion-dynamic-simulation

First git clone this repo
```
git clone https://github.com/ylctql/ion-dynamic-simulation.git
```
Then pip install it:
```
pip install .
```
And it could be used. 

**Remember to get your electric field file, it's not included in this repo!**

To simulate the motions of ions,
```
python3 src/monolithic.py <number of ions> [--CUDA]
```
--CUDA is for cuda acceleration on large number of ions, which is an optional parameter. If not used, the program runs on CPU for default, which is faster when the number of ions is small.
