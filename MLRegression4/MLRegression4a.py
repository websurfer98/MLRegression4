import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Helper functions
def get_numpy_data(df, features, output):
    df['constant'] = 1 # add a constant column 

    # prepend variable 'constant' to the features list
    features = ['constant'] + features

    # Filter by features
    fm = df[features]
    y = df[output]
   
    # convert to numpy matrix/vector whatever...
    # http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.as_matrix.html
    features_matrix = fm.as_matrix()
    output_array = y.as_matrix()

    return(features_matrix, output_array)

def predict_output(feature_matrix, weights):
    result = feature_matrix.dot(weights.T)
    return result

def feature_derivative_ridge(errors, feature, weight, l2_penalty, feature_is_constant):
    if feature_is_constant:
        return 2 * feature.T.dot(errors)
    else:
        return 2 * feature.T.dot(errors) + 2*l2_penalty*weight

def ridge_regression_gradient_descent(feature_matrix, output, initial_weights, step_size, l2_penalty, max_iterations=100):
    weights = np.array(initial_weights)
    for j in range(max_iterations):
        # compute the predictions based on feature_matrix and weights
        yhat = predict_output(feature_matrix, weights)

        # compute the errors as predictions - output
        errors = np.subtract(yhat, output)
        
        gradient_sum_squares = 0 # initialize the gradient
        # while not converged, update each weight individually:
        for i in range(len(weights)):
            # Recall that feature_matrix[:, i] is the feature column associated with weights[i]
            # compute the derivative for weight[i]
            delw = feature_derivative_ridge(errors, feature_matrix[:, i], weights[i], l2_penalty, i==0)
            
            # add the squared derivative to the gradient magnitude
            gradient_sum_squares = delw ** 2
            
            # update the weight based on step size and derivative
            weights[i] -= step_size * delw
        
    return(weights)

# Load the King County house data

# Use the type dictionary supplied
dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}
sales = pd.read_csv('kc_house_data.csv', dtype=dtype_dict)

print("Testing begin...")
(example_features, example_output) = get_numpy_data(sales, ['sqft_living'], 'price')
my_weights = np.array([1., 10.])
test_predictions = predict_output(example_features, my_weights)
errors = test_predictions - example_output # prediction errors

# next two lines should print the same values
print(feature_derivative_ridge(errors, example_features[:,1], my_weights[1], 1, False))
print(np.sum(errors*example_features[:,1])*2+20.)
print('The magnitues above should be the same')

# next two lines should print the same values
print(feature_derivative_ridge(errors, example_features[:,0], my_weights[0], 1, True))
print(np.sum(errors)*2.)
print('The magnitues above should be the same')

print("Testing end...")

# Load training data and test data
train_data = pd.read_csv('kc_house_train_data.csv', dtype=dtype_dict)
test_data = pd.read_csv('kc_house_test_data.csv', dtype=dtype_dict)


#######################
# Using training data #
#######################
print("\n\nUsing training data for model 1...")

simple_features = ['sqft_living']
my_output= 'price'
(simple_feature_matrix, output) = get_numpy_data(train_data, simple_features, my_output)
initial_weights = np.array([0.0, 0.0])
step_size = 1e-12
l2_penalty = 0.0
max_iterations = 1000

simple_weights_0_penalty = ridge_regression_gradient_descent(simple_feature_matrix, output, initial_weights, step_size, l2_penalty, max_iterations)
print("simple_weights 0 penalty: ", simple_weights_0_penalty[1])

l2_penalty = 1e11
simple_weights_high_penalty = ridge_regression_gradient_descent(simple_feature_matrix, output, initial_weights, step_size, l2_penalty, max_iterations)
print("simple_weights high penalty: ", simple_weights_high_penalty[1])

# Plot the results
plt.plot(simple_feature_matrix,output,'k.',
        simple_feature_matrix,predict_output(simple_feature_matrix, simple_weights_0_penalty),'b-',
        simple_feature_matrix,predict_output(simple_feature_matrix, simple_weights_high_penalty),'r-')
plt.show()

###################
# Using test data #
###################
print("\n\nUsing test data for model 1...")

test_features = ['sqft_living']
test_output= 'price'
(test_feature_matrix, output) = get_numpy_data(test_data, test_features, test_output)

# RSS computations
yhat = test_feature_matrix.dot(initial_weights)
diff = yhat - output
rss = diff.T.dot(diff)
print("Rss using initial weights: ", rss)

yhat = test_feature_matrix.dot(simple_weights_0_penalty)
diff = yhat - output
rss = diff.T.dot(diff)
print("Rss using simple_weights_0_penalty: ", rss)

yhat = test_feature_matrix.dot(simple_weights_high_penalty)
diff = yhat - output
rss = diff.T.dot(diff)
print("Rss using simple_weights high penalty: ", rss)

#######################
# Using training data #
#######################
print("\n\nUsing training data for model 2...")

model_features = ['sqft_living', 'sqft_living15']
my_output = 'price'

(feature_matrix, output) = get_numpy_data(train_data, model_features, my_output)
initial_weights = np.array([0.0, 0.0, 0.0])
step_size = 1e-12
l2_penalty = 0.0
max_iterations = 1000

multiple_weights_0_penalty = ridge_regression_gradient_descent(feature_matrix, output, initial_weights, step_size, l2_penalty, max_iterations)
print("multiple_weights 0 penalty: ", multiple_weights_0_penalty[1])

l2_penalty = 1e11
multiple_weights_high_penalty = ridge_regression_gradient_descent(feature_matrix, output, initial_weights, step_size, l2_penalty, max_iterations)
print("multiple_weights high penalty: ", multiple_weights_high_penalty[1])


###################
# Using test data #
###################
print("\n\nUsing test data for model 2...")

model_features = ['sqft_living', 'sqft_living15']
my_output= 'price'
(test_feature_matrix, output) = get_numpy_data(test_data, model_features, my_output)

# RSS computations
yhat = test_feature_matrix.dot(initial_weights)
diff = yhat - output
rss = diff.T.dot(diff)
print("Rss using initial weights: ", rss)

yhat = test_feature_matrix.dot(multiple_weights_0_penalty)
diff = yhat - output
rss = diff.T.dot(diff)
print("Rss using simple_weights_0_penalty: ", rss)
print("First house prediction using model 2 with no regularization: ", yhat[0])
print("First house prediction error using model 2 with no regularization: ", diff[0])

yhat = test_feature_matrix.dot(multiple_weights_high_penalty)
diff = yhat - output
rss = diff.T.dot(diff)
print("Rss using simple_weights high penalty: ", rss)
print("First house prediction using model 2 with high regularization: ", yhat[0])
print("First house prediction error using model 2 with high regularization: ", diff[0])

# Predict house price for first house
print("Actual price of house in test set: ", output[0])