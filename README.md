# Table of contents

1. [Introduction](#introduction)
2. [Requirements](#requirements)
3. [Installation](#installation)
    1. [Optional](#installation-optional)
    2. [Mandatory](#installation-mandatory)
4. [Usage](#usage)
5. [Note](#note)



## 1. Introduction <a name="introduction"></a>

This tool shows performance of several machine learning models in a multi-class classification problem.
The dataset used to train the models is under `./res/dataset/` folder.

## 2. Requirements <a name="requirements"></a>

The following libraries are required:
1. [numpy](https://numpy.org/)
2. [pandas](https://pandas.pydata.org/)
3. [matplotlib](https://matplotlib.org/)
4. [sklearn](https://scikit-learn.org/stable/index.html)
5. [imblearn](https://pypi.org/project/imblearn/)
6. [scipy](https://www.scipy.org/)

## 3. Installation <a name="installation"></a>

### 3.1 Optional <a name="installation-optional"></a>

```bash
# Install virtualenv package for Python:
pip install virtualenv

# Create virtual environment named MulticlassClassifier:
virtualenv MulticlassClassifier
# or
virtualenv -p python3 MulticlassClassifier

# Activate virtual environment:
source MulticlassClassifier/bin/activate

# Deactivate virtual environment:
deactivate
```

### 3.2 Mandatory <a name="installation-mandatory"></a>

```bash
# Install required pip-tools for requirements installation:
python -m pip install pip-tools

# Move to root project directory:
cd /project_root/

# Read requirements to install:
pip-compile --output-file requirements.txt requirements.in

# Install required packages:
pip-sync
```

## 4. Usage <a name="usage"></a>

Launch tool:

```bash
# Start main:
python src/Client.py

# or you can specify a configuration file:
python src/Client.py /path/custom_config.ini
```

Configurations file example:
```ini
# [LEGEND]
#   [opt] := optional parameter
#   [dft] := default value
#   [mnd] := mandatory parameter


[GENERAL]
# [opt] - Directory for temporary files
# [dft] - /tmp
tmp = /Volumes/Ramdisk/

# [opt] - Set verbose level to debug
# [dft] - False
debug = True


[TRAINING]
# [mnd] - Dataset for training purpose (fully qualified path name)
dataset = /Volumes/Data/Projects/Python/MulticlassClassifier/res/dataset/training_set.csv
# dataset = /Volumes/Data/Projects/Python/MulticlassClassifier/res/dataset/auto-mpg.data
# dataset = /Volumes/Data/Projects/Python/MulticlassClassifier/res/dataset/diabetes.csv

# [opt] - Set test ratio from dataset
# [dft] - 0.2
# dataset.test_ratio = 0.3

# [opt] - Seed for RNG
# [dft] - 0
rng.seed = 43531
# rng.seed = 2873587

# [opt] - Compute pair-plot
# [dft] - False
# pair_plot.compute = True

# [opt] - Save pair-plot on file
# [dft] - False
# pair_plot.save = True

# [opt] - Thread to use during training
# [dft] - 1
threads = 4


[MOBD]
# [opt] - Best benchmark computed for F1-score metric
# [dft] - 0.0
benchmark.best_found = (0.8444, 'Multi-Layer Perceptron')

# [opt] - Current benchmark threshold evaluation and deadline (time format: dd/mm/yyyy)
# [dft] - (0.0 - 'datetime.today()')
benchmark.threshold = (0.8906, '04/09/2020')

# [opt] - Dataset for project evaluation (fully qualified path name)
#   If tool shutdown without message error, probably the format of test set file is wrong.
#   By default this tool manages test file without index column (so, saved from pandas.data_frame.to_csv('path', index=False))
#   If you want to input test set file with index column just go to Evaluator.__init__() and change line
#     self.__test = Set(pd.read_csv(self.conf.dataset_test)) to self.__test = Set(pd.read_csv(self.conf.dataset_test, index_col=0))
# [dft] -
# dataset.test = /Volumes/Data/Projects/Python/MulticlassClassifier/res/dataset/test_set_index.csv
# dataset.test = /Volumes/Data/Projects/Python/MulticlassClassifier/res/dataset/test_set_no_index.csv
# dataset.test = /Volumes/Data/Projects/Python/MulticlassClassifier/res/dataset/test_set_no_index_features.csv
```

## 5. Note <a name="note"></a>

In order to evaluate project using secret *test set* just use `dataset.test` option from `conf.ini` (see [usage](#usage)) and
insert fully qualified path name of test file, then launch this tool as shown in [usage](#usage).
