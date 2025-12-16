# ion-dynamic-simulation

### 0. Start-up
Before using the program, you may run the following command at the root directory of this repo
```
pip install .
```
This will automatically install the required packages and do the compilation. We strongly recommend you to use virtual environment (such as Anaconda) in case of unexpected conflicts with your pre-installed versions. Please check https://zhuanlan.zhihu.com/p/32925500 for tutorials.

**After that, you should put your own data of electric fields in the directory `/data` as it is too large to be contained in this repo!** If you do not have one, you may download a sample copy of data from https://cloud.tsinghua.edu.cn/f/aef6277873a0464f9eab/?dl=1.

### 1. Create a configure
In this project, we defined two classes called `Data_loader` and `Configure`. First you may load the grid data by 

```
basis = Data_Loader(filename, basis_filename, flag_smoothing)
basis.loadData()
```
Here, the `filename` is what you have put in the `/data` directory as your electric field grid data, and the `basis_filename` is some custom basis settings (which has been provided in this repo). After a `Data_Loader` is defined and loaded, you may define your own configuration of electric fields by 

```
configure = Configure(basis=basis)
configure.load_from_file(file)
```
Here `file` is a json file that includes the potential data of all the electrodes, for instance

```
{
    "V_static": {
        "RF": -7.25156238001962,
        "U1": 0.17269838528176118,
        "U2": 0.43617245456405995,
        "U3": 0.611378067391062,
        "U4": -0.886486757350588,
        "U5": 0.9771554372897736,
        "U6": 0.5212677070289424,
        "U7": 0.22461453785257968
    },
    "V_dynamic": {
        "RF": 274.9013367225178
    }
}
```
You may also load a cofigure by `configure.load_from_param()` or directly specify when you create a configure. Check the codes for detailed usage. After defining your own `configure`, it is time to perform simulation and optimization.

### 2. Built-in pipelines

Two sample pipelines are provided in the `/src` directory, which are `main.py` and `optimize.py`. Here we introduce them individually.

#### `main.py`
After defining a valid configure, you may use the following command for simulation

```
python src/main.py --N <number of ions> [--CUDA] [--plot] [--time <total simulation time>]
```

`--CUDA` is an optional choice for using CUDA. If not chosen, the program runs on CPU as default.

`--plot` is an optional choice for showcasing the real-time motion of ions.

`--time` is an optional parameter that records the simulation time in miliseconds. If not specified, the program runs ceaselessly until being shut down from the terminal or by closing the plotting window.

After the process is done, an estimated thickness will be printed.

#### `optimize.py`
After defining a valid configure, you may use the following command for optimization

```
python src/optimize.py --N <number of ions> [--CUDA] [--time <total simulation time>] [--epochs <number of epochs>]
```

`--CUDA`, `--plot`, `--time` are similar arguments with those of `main.py`. Just note that here the default value for `--time` is 10.0 and everything will be fine. 

`--epochs` is the parameter that records the total number of optimzation epochs. If not specified, the optimization takes 10 epochs.

After the optimization is completed, the configure is saved in a json `file` by
```
configure.save(file)
```

### 3. Contact

If you have any questions or ideas, feel free to contact us through Email (alexyan0431@gmail.com).


