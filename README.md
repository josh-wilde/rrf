# Two-Stage Classifier

This repo is associated with Chapter 4 of the thesis "Analytics-Enabled Quality and Safety Management Methods for High-Stakes Manufacturing Applications" by Joshua Wilde at the MIT Operations Research Center.

## Installation
### Install Python packages into a virtual environment
1. Install Python 3.9 and make sure this python version is active. 
2. Navigate to the `rrf` repo. 
3. Create a virtual environment named `venv` in this directory. It will be ignored by git. 
4. Activate this virtual environment: `source venv\bin\activate`. 
5. Install the required python packages from `requirements.txt`: `python -m pip install -r requirements.txt`.

### Install R packages into a virtual environment
1. Install R 4.1 and make sure that this version is active (when called by `R` from command line) if you have multiple R versions. 
2. Run R from the command line: `R`. 
3. Restore the virtual R environment by running the following command in R: `renv::restore()`.
**Warning**: this last step will take a long time on the cluster the first time you run it, but it is a one-off.

## Usage
The `rrf/rrf/scripts/run_cv/run_cv.py` file contains the main script for running cross validation experiments for a particular model and data configuration.
You will need to set up your own config file to run a new experiment. 
Templates for configs used for the thesis chapter associated with this repo are located in `rrf/rrf/config/templates`.
These templates also contain instructions for creating input data files.
