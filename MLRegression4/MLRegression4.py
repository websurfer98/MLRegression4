import pandas as pd
import numpy as np
from sklearn import linear_model
import random
import matplotlib.pyplot as plt

# Load the King County house data
# Download the King County House Sales training data csv file: wk3_kc_house_train_data.csv
# Download the King County House Sales validation data csv file: wk3_kc_house_valid_data.csv
# Download the King County House Sales testing data csv file: wk3_kc_house_test_data.csv
# Download the King County House Sales subset 1 data csv file: wk3_kc_house_set_1_data.csv
# Download the King County House Sales subset 2 data csv file: wk3_kc_house_set_2_data.csv
# Download the King County House Sales subset 3 data csv file: wk3_kc_house_set_3_data.csv
# Download the King County House Sales subset 4 data csv file: wk3_kc_house_set_4_data.csv

# Helper functions
def polynomial_dataframe(feature, degree): # feature is pandas.Series type
    poly_dataframe = pd.DataFrame()
   
    # first check if degree >= 1
    if degree >= 1:
        # then loop over the remaining degrees:
        for power in range(1, degree+1):
            # first we'll give the column a name:
            name = 'power_' + str(power)
            if power == 1:
                poly_dataframe[name] = feature
            else:
                poly_dataframe[name] = poly_dataframe["power_1"] ** power
            
    return poly_dataframe

def test_polynomial_dataframe():
    feature = pd.Series(np.random.randn(5))
    poly_dataframe = polynomial_dataframe(feature, 5)
    print(poly_dataframe)

# Unit test invocations for the helper functions
# Test - Passes
# test_polynomial_dataframe()

def loadFile(dataFile, maxPol):
    print("\nLoading file: " + dataFile + " with maxPol: " + str(maxPol))
     # Use the type dictionary supplied
    dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

    sales = pd.read_csv(dataFile, dtype=dtype_dict, usecols=["sqft_living", "price"], header=0)
    sales = sales.sort_values(by=['sqft_living','price'])
    # print(sales.head(5))
    poly_data = polynomial_dataframe(sales['sqft_living'], maxPol)
    poly_data['price'] = sales['price']
    #print(poly1_data.head(10))

    return poly_data

def displayCoeffsRidge(trainFile, alpha, maxPol, displayPlot):
    poly_data = loadFile(trainFile, maxPol)
    print("Using an l2 penalty value of: ", alpha)

    # y is the output, i.e. price
    y = poly_data['price']
    # X is everything else but the output, so drop the output!
    X = poly_data.drop('price', axis=1)
    #print(X.head(5))

    lm = linear_model.Ridge(alpha=alpha, copy_X=True, fit_intercept=True, normalize=True)
    lm.fit(X, y)
    print('Intercept: ', lm.intercept_)
    coeffs = pd.DataFrame(list(zip(X.columns, lm.coef_)), columns = ['features', 'estimatedCoefficients'])
    print(coeffs)

    if displayPlot:
        # Plot using Matplotlib
        # Examples http://matplotlib.org/1.5.1/examples/index.html
        plt.plot(poly_data['power_1'], poly_data['price'], '.', poly_data['power_1'], lm.predict(X), '-')
        plt.show()

    return lm

#################################################################
# K-fold validation using wk3_kc_house_train_valid_shuffled.csv #
#################################################################
def getIndices(n, k, i):
    start = int((n*i)/k)
    end = int((n*(i+1))/k-1)
    return (start, end)

def kFoldRidge(poly_data, alpha, n, k, displayPlot):
    print("Using an l2 penalty value of: ", alpha)
    rss = 0
    for i in range(k):
        (start, end) = getIndices(n, k, i)

        # So we validate on this set and the rest is training data
        vald_data =  poly_data[start:end+1]
        train_data = poly_data[0:start].append(poly_data[end+1:n])

        # y is the output, i.e. price
        y = train_data['price']
        # X is everything else but the output, so drop the output!
        X = train_data.drop('price', axis=1)

        lm = linear_model.Ridge(alpha=alpha, copy_X=True, fit_intercept=True, normalize=True)
        lm.fit(X, y)
        #print('Intercept: ', lm.intercept_)
        coeffs = pd.DataFrame(list(zip(X.columns, lm.coef_)), columns = ['features', 'estimatedCoefficients'])
        #print(coeffs)

        vald_X = vald_data.drop('price', axis=1)
        r = np.sum((vald_data['price'] - lm.predict(vald_X)) ** 2)
        rss += r
        print("Rss: ", r)
       
    # Return the average validation error
    print("Avg. validation Rss: ", rss/k)
    return rss/k

######################################################
# Load house data and use the 15th degree polynomial #
######################################################

# When we have so many features and so few data points, the solution can become highly numerically unstable, 
# which can sometimes lead to strange unpredictable results. Thus, rather than using no regularization, we 
# will introduce a tiny amount of regularization (l2_penalty=1.5e-5) to make the solution numerically stable
l2_small_penalty = 1.5e-5
displayCoeffsRidge('kc_house_data.csv', l2_small_penalty, 15, False)

#####################
# Using the 4 files #
#####################
l2_small_penalty=1e-9
print("\n\nUsing a small penalty =", l2_small_penalty)
displayCoeffsRidge('wk3_kc_house_set_1_data.csv', l2_small_penalty, 15, False)
displayCoeffsRidge('wk3_kc_house_set_2_data.csv', l2_small_penalty, 15, False)
displayCoeffsRidge('wk3_kc_house_set_3_data.csv', l2_small_penalty, 15, False)
displayCoeffsRidge('wk3_kc_house_set_4_data.csv', l2_small_penalty, 15, False)

# Penalize variance by using large weights
l2_large_penalty=1.23e2
print("\n\nUsing a large penalty =", l2_large_penalty)
displayCoeffsRidge('wk3_kc_house_set_1_data.csv', l2_large_penalty, 15, False)
displayCoeffsRidge('wk3_kc_house_set_2_data.csv', l2_large_penalty, 15, False)
displayCoeffsRidge('wk3_kc_house_set_3_data.csv', l2_large_penalty, 15, False)
displayCoeffsRidge('wk3_kc_house_set_4_data.csv', l2_large_penalty, 15, False)

###########################################
# Using the shuffled file containing data #
###########################################
maxPol = 15
poly_data = loadFile('wk3_kc_house_train_valid_shuffled.csv', maxPol)
n = len(poly_data)
k = 10
l2_penalty = np.logspace(3, 9, num=13)
print("L2 penalty list", l2_penalty)

bestRss = None
bestPenalty = None
for i in range(0, len(l2_penalty)):
    rss = kFoldRidge(poly_data, l2_penalty[i], n, k, False)
    if bestRss == None:
            bestRss = rss
            bestPenalty = l2_penalty[i]
    else:
        if rss < bestRss:
            bestRss = rss   
            bestPenalty = l2_penalty[i]

print("Best Rss = ", bestRss)
print("Best penalty = ", bestPenalty)

# Train using all of available training data 
lm = displayCoeffsRidge('wk3_kc_house_train_valid_shuffled.csv', bestPenalty, 15, False)

###################
# Using test data #
###################
poly_data = loadFile('wk3_kc_house_test_data.csv', maxPol)
y = poly_data['price']
test_X = poly_data.drop('price', axis=1)

rss += np.sum((y - lm.predict(test_X)) ** 2)
print("Test Rss: ", rss)
