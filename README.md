# Table of contents

1. [Introduction](#introduction)
2. [Requirements](#requirements)
3. [Installation](#installation)
    1. [Optional](#installation-optional)
    2. [Mandatory](#installation-mandatory)
4. [Usage](#usage)
5. [Note](#note)



## 1. Introduction <a name="introduction"></a>

This tool evaluate performance of several machine learning models in a multi-class classification problem.
The dataset used to train the models is `./res/dataset/training_set.csv`.

Under `./doc` folder there are the requirements (`./doc/MOBD_2019-2020.pdf`) and the report (`./doc/[MOBD] Project 2019-2020.pdf`) for this project.

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

# Once finished, to deactivate virtual environment:
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
# Start client:
python src/Client.py

# or you can specify a configuration file:
python src/Client.py /abs_path/custom_conf_file.ini
```

Note that `src/Client.py` script provide all functionalities to evaluate/train selected classifiers.

If custom configuration file is not specified, `src/Client.py` reads `./res/conf/conf.ini` file to get application's configuration.
This file, should already contain all parameters set properly, except for `dataset.train` and `dataset.test` option in which
you must specify, respectively, absolute path for training dataset and testing set.

To modify behaviour of the software just modify option from configuration file.  

Configurations file example (`./res/conf/conf.ini`):
```ini
# [LEGEND]
#   [opt] := optional parameter
#   [dft] := default value
#   [mnd] := mandatory parameter

[GENERAL]
# [opt] - Directory for temporary files.
# [dft] - /tmp
# tmp = /Volumes/Ramdisk/

# [opt] - Set verbose level to debug.
# [dft] - False
# debug = True


[TRAINING]
# [mnd] - Dataset for training purpose (fully qualified path name).
dataset.train = /ABS_PATH_TRAINING_SET.csv

# [opt] - Set test ratio from dataset.
# [dft] - 0.2
# dataset.test_ratio = 0.3

# [opt] - Seed for RNG.
# [dft] - 0
rng.seed = 43531
# rng.seed = 2873587

# [opt] - Compute pair-plot.
# [dft] - False
# pair_plot.compute = True

# [opt] - Save pair-plot on file (will be used folder set with tmp option, see GENERAL section).
# [dft] - False
# pair_plot.save = True

# [opt] - Max number of jobs to use during training.
# [dft] - -1
# jobs = 24

# [opt] - Create dump file for best classifiers.
# [dft] - False
#   If this option is set to True:
#     - if it has been specified dataset.test (section MOBD) option with fully qualified path of test set (for MOBD project evaluation)
#       then, will be loaded classifiers dump from folder ./res/classifier/*.joblib and all classifiers will be evaluated
#       on test set file, after dataset preprocessing, using various metrics;
#     - if it has not been specified dataset.test option, then all classifiers will be trained with dataset specified in
#       dataset.train option (section TRAINING), after dataset preprocessing, a dump of bests classifiers will be saved
#       under ./res/classifier/*.joblib folder and bests classifiers will be evaluated on test set created from
#       training set, using various metrics.
#   Otherwise:
#     - if it has been specified dataset.test (section MOBD) option with fully qualified path of test set (for MOBD project evaluation)
#       then, all classifiers will be trained, after dataset preprocessing, using file specified with option
#       dataset.train (section TRAINING) and, bests classifiers will be evaluated on test set file, using various metrics;
#     - if it has not been specified dataset.test option, then all classifiers will be trained, after dataset preprocessing,
#       with dataset specified in dataset.train option and will be evaluated on test set created from training set, using various metrics.
classifier.dump = True


[MOBD]
# [opt] - Best benchmark computed for F1-score metric.
# [dft] - (0.0 - '')
benchmark.best_found = (0.8973, 'Multi-Layer Perceptron')

# [opt] - Current benchmark threshold evaluation and deadline (time format: dd/mm/yyyy).
# [dft] - (0.0 - datetime.today())
benchmark.threshold = (0.8906, '04/09/2020')

# [opt] - Dataset for project evaluation (fully qualified path name).
# [dft] - ''
#   If you specify option dataset.test, this tool will perform evaluation on specified test set file once launched,
#     so no further actions are needed.
#
#   If tool shutdown without message error, probably the format of test set file is wrong.
#   By default this tool manages test file without index column (so, saved with command
#     pandas.data_frame.to_csv('/abs_path', index=False); see ./res/dataset/test_set_no_index.csv for example file).
#   If you want to input test set file with index column (saved with command
#     pandas.data_frame.to_csv('/abs_path', index=True); see ./res/dataset/test_set_index.csv for example file)
#     just go to Evaluator.__init__() and change line self.__test = Set(pd.read_csv(self.conf.dataset_test)) to
#     self.__test = Set(pd.read_csv(self.conf.dataset_test, index_col=0)).
#   If you want to input test set file without index column and without header row (does not have F1-20 and CLASS row;
#     see ./res/dataset/test_set_no_index_features.csv for example file), just go to Evaluator.__init__() and change
#     line self.__test = Set(pd.read_csv(self.conf.dataset_test)) to
#     self.__test = Set(pd.read_csv(self.conf.dataset_test, header=None,
#                                   names=[f"F{i}" for i in range(1, 21)] + ["CLASS"]))
dataset.test = /ABS_PATH_TEST_SET.csv
```

If you want to run this tool in a python console (or in a custom python script):
```python
from model.Conf import Conf
from classifier.Evaluator import Evaluator

# create configuration instance
conf = Conf.get_instance()
# load parameter from a file using fully qualified path name
conf.load_from('/ABS_PATH_CONF_FILE.ini')
# once configuration has been loaded, if you want to modify some parameter, there are many getters/setters
# e.g. to enable classifiers evaluation using dump files, use:
# conf.classifier_dump = True
# however it is not recommended because default values in conf.ini (under ./res/conf/) should be fine 
#   in order to evaluate MOBD project using secret test set

# start evaluation
Evaluator(conf).process()
```

## 5. Note <a name="note"></a>

In order to evaluate project using *secret test set* just use `dataset.test` option from `conf.ini` (see [usage](#usage) for configuration file example) and
insert fully qualified path name of secret test file, then launch this tool normally as shown in [usage](#usage).

Note that, *secret test set* to be inserted, **must** have same format of `training_set.csv`, so must have row for features name (F1-F20) and class (CLASS)
and **must** not have index column, so a valid test file should be created from command `pandas.data_frame_object.to_csv('/absolute_path', index=False)`.
Evaluator class is responsible for test set evaluation. In the constructor of Evaluator class (Evaluator.py file), is it possible to modify the way used to load secret test set to be used.
In particular, as shown in Evaluator.py file:
```python
# load test set if it has same format as training_set.csv provided
# as example file see ./res/dataset/test_set_no_index.csv
self.__test = Set(pd.read_csv(self.conf.dataset_test))
# load test set if it has header (F1-20 and CLASS row) and index, so a test test saved using
#   command pd.to_csv('/path', index=True)
# as example file see ./res/dataset/test_set_index.csv
# self.__test = Set(pd.read_csv(self.conf.dataset_test, index_col=0))
# load test set if it does not have header row (does not have F1-20 and CLASS row) and
#   it is was not saved using command pd.to_csv('/path', index=True), so it has not index
# as example file see ./res/dataset/test_set_no_index_features.csv
# self.__test = Set(pd.read_csv(self.conf.dataset_test, header=None,
#                               names=[f"F{i}" for i in range(1, 21)] + ["CLASS"]))
```