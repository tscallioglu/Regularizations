import matplotlib.pyplot as plt
from utils import plot_decision_boundary, load_2D_dataset, predict_dec, predict

from reg_utils import model, compute_cost_with_regularization
from testCases import *

if __name__ == '__main__':

    plt.rcParams['figure.figsize'] = (7.0, 4.0) # Sets default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'
    
    train_X, train_Y, test_X, test_Y = load_2D_dataset()
    
    print("NON-REGULARIZED MODEL")
    parameters = model(train_X, train_Y)
    print ("Non-Regularized Model - On the training set:")
    predictions_train = predict(train_X, train_Y, parameters)
    print ("Non-Regularized Model - On the test set:")
    predictions_test = predict(test_X, test_Y, parameters)
    
    plt.title("Model without regularization")
    axes = plt.gca()
    axes.set_xlim([-0.75,0.40])
    axes.set_ylim([-0.75,0.65])
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
    
    A3, Y_assess, parameters = compute_cost_with_regularization_test_case()
    
    
    print("\nMODEL with L2 REGULARIZATION - Cost = " + str(compute_cost_with_regularization(A3, Y_assess, parameters, lambd = 0.1)))
    
    parameters = model(train_X, train_Y, lambd = 0.7)
    print ("L2 Regularized Model - On the train set:")
    predictions_train = predict(train_X, train_Y, parameters)
    print ("L2 Regularized Model - On the test set:")
    predictions_test = predict(test_X, test_Y, parameters)
    
    plt.title("Model with L2-regularization")
    axes = plt.gca()
    axes.set_xlim([-0.75,0.40])
    axes.set_ylim([-0.75,0.65])
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
    
    
    print("\nMODEL with DROPOUT")
    parameters = model(train_X, train_Y, keep_prob = 0.86, learning_rate = 0.3)
    
    print ("Dropout Regularized Model - On the train set:")
    predictions_train = predict(train_X, train_Y, parameters)
    print ("Dropout Regularized Model - On the test set:")
    predictions_test = predict(test_X, test_Y, parameters)
    
    
    plt.title("Model with dropout")
    axes = plt.gca()
    axes.set_xlim([-0.75,0.40])
    axes.set_ylim([-0.75,0.65])
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
    
    
