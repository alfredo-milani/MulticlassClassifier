# Table of contents
1. [Introduction](#introduction)
2. [Requirements](#requirements)
3. [Installation](#installation)
    1. [Optional](#installation-optional)
    2. [Mandatory](#installation-mandatory)
4. [Usage](#usage)
5. [Note](#note)



## 1. Introduction <a name="introduction"></a>
TODO

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


[MOBD]
# [opt] - Compute pairplot
# [dft] - False
# pair_plot.compute = True

# [opt] - Seed for RNG
# [dft] - 0
rng.seed = 43531

# [opt] - Best benchmark computed
# [dft] - 0.0
benchmark.value = 0.8555

# [opt] - Current benchmark threshold evaluation
# [dft] - 0.0
benchmark.threshold = 0.8868

# [opt] - Current deadline (time format: dd/mm/yyyy)
# [dft] - None
benchmark.deadline = 23/07/2020

# [opt] - Set test ratio from dataset
# [dft] - 0.2
# dataset.test_ratio = 0.3

# [mnd] - Dataset for training purpose (fully qualified path name)
dataset = /Volumes/Data/Projects/Python/MulticlassClassifier/res/dataset/training_set.csv

# [opt] - Dataset for project evaluation (fully qualified path name)
# [dft] -
# dataset.test =
```

## 5. Note <a name="note"></a>
