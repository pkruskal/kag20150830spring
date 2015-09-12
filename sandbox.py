import pandas as pd
import matplotlib.pylab as plt
import numpy as np
from scipy import stats


'''
COMENTS OF INTEREST
'''

'''
Guys, there's a big hint in the competition overview:
You are challenged to construct new meta-variables and employ feature-selection methods to approach this dauntingly wide dataset.
I'm thinking PCA as it is good at deriving orthogonal features from a large number of features....
'''


#LOAD THE DATA AS DATA FRAME
train = './input/train.csv'
test  =  './input/test.csv'#'vali_100.tsv'



def preview1(train):
    print('\nSummary of train dataset:\n')
    description = train.describe()
    description = description.T
    for key in description.keys():
        print description[key]

    a = description['mean'].values
    a = a[np.where(~np.isnan(a))]
    plt.hist(np.log(a-min(a)+1),1000)
    plt.xlabel('mean variable value')

    '''
    kde = stats.gaussian_kde(a)
    x = np.linspace(min(a),18000000,10000)
    plt.plot(x,kde(x))
    '''

    #print(train.loc[1])

    const_cols = [col for col in train.columns if len(train[col].unique()) == 1]
    print 'variables with constant values'
    print(const_cols) # ['VAR_0207', 'VAR_0213', 'VAR_0840', 'VAR_0847', 'VAR_1428']

    bool_cols = [col for col in train.columns if len(train[col].unique()) == 2]
    print('binary variables')
    print(bool_cols) #

    var_cols = [col for col in train.columns if len(train[col].unique()) > 5]
    print('number of variables with more then 5 values')
    print(len(var_cols)) #

trainDF = pd.DataFrame.from_csv(train)

#print off variable value pairs
for key in trainDF.keys():
    print key, trainDF[key].values[0]


preview1(trainDF)