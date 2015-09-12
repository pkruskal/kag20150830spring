# Note: Kaggle only runs Python 3, not Python 2

# We'll use the pandas library to read CSV files into dataframes
import pandas as pd

# The competition datafiles are in the directory ../input
# List the files we have available to work with
print("> ls ../input")
from subprocess import check_output
print(check_output(["ls", "/input"]).decode("utf8"))

# Read train data file:
train = pd.read_csv("./input/train.csv")

# Write summaries of the train and test sets to the log
print('\nSummary of train dataset:\n')
print(train.describe())

print(train.loc[1])

const_cols = [col for col in train.columns if len(train[col].unique()) == 1]
print(const_cols) # ['VAR_0207', 'VAR_0213', 'VAR_0840', 'VAR_0847', 'VAR_1428']


bool_cols = [col for col in train.columns if len(train[col].unique()) == 2]
print(bool_cols) #

var_cols = [col for col in train.columns if len(train[col].unique()) > 5]
print(len(var_cols)) #