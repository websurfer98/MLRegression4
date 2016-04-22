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

# Global vars to store the best match
bestRss = None
bestPoly = None

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

def displayCoeffs(trainFile, maxPol, calcRss, valdFile, displayPlot):
    poly_data = loadFile(trainFile, maxPol)

    # y is the output, i.e. price
    y = poly_data['price']
    # X is everything else but the output, so drop the output!
    X = poly_data.drop('price', axis=1)
    #print(X.head(5))

    lm = linear_model.LinearRegression(copy_X=True, fit_intercept=True, normalize=False)
    lm.fit(X, y)
    print('Intercept: ', lm.intercept_)
    coeffs = pd.DataFrame(list(zip(X.columns, lm.coef_)), columns = ['features', 'estimatedCoefficients'])
    print(coeffs)

    if calcRss:
        global bestRss, bestPoly
        vald_data = loadFile(valdFile, maxPol)
        vald_X = vald_data.drop('price', axis=1)
        rss = np.sum((vald_data['price'] - lm.predict(vald_X)) ** 2)
        print("Validation Rss: ", rss)

        if bestRss == None:
            bestRss = rss
            bestPoly = maxPol
        else:
            if rss < bestRss:
                bestRss = rss   
                bestPoly = maxPol

    if displayPlot:
        # Plot using Matplotlib
        # Examples http://matplotlib.org/1.5.1/examples/index.html
        plt.plot(poly_data['power_1'], poly_data['price'], '.', poly_data['power_1'], lm.predict(X), '-')
        plt.show()

#####################
# Using the 4 files #
#####################
displayCoeffs('wk3_kc_house_set_1_data.csv', 15, False, '', False)
displayCoeffs('wk3_kc_house_set_2_data.csv', 15, False, '', False)
displayCoeffs('wk3_kc_house_set_3_data.csv', 15, False, '', False)
displayCoeffs('wk3_kc_house_set_4_data.csv', 15, False, '', False)

###########################
# Using the training file #
###########################
for x in range(1, 16):
    displayCoeffs('wk3_kc_house_train_data.csv', x, True, 'wk3_kc_house_valid_data.csv', False)

print("Best polynomial = " + str(bestPoly))

###################
# Using test data #
###################
displayCoeffs('wk3_kc_house_train_data.csv', bestPoly, True, 'wk3_kc_house_test_data.csv', False)

