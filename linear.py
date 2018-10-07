# Linear regression algorithm
# optimization method: gradient descent

import numpy as np
import matplotlib.pyplot as plt

def cost_function(X, y, theta):
    m = len(X)
    M = np.hstack(((np.ones(len(X))).reshape(-1,1),X.reshape(-1,1)))
    return (1/(2*m))*np.dot((np.dot(M,theta)-y).T,(np.dot(M,theta)-y))

def gradient(X,y,theta):
    m = len(X)
    M = np.hstack(((np.ones(len(X))).reshape(-1,1),X.reshape(-1,1)))
    return (1/m)*(np.dot(np.dot(M.T,M),theta)-np.dot(M.T,y))

class linear_regression:

    def __init__(self):
        self.theta = np.array([0,0])
        self.cost = 0
        self.theta_history = []

    def train(self,X,y,num_steps,learn_rate):
        self.theta = 2*np.random.rand(2)-1
        self.theta_history.append(self.theta)
        for i in range(0,num_steps):
            self.theta = self.theta - learn_rate*gradient(X,y,self.theta)
            self.theta_history.append(self.theta)
        self.cost =  cost_function(X,y,self.theta)

    def get_parameters(self):
        return self.theta

    def get_score(self,X,y):
        return cost_function(X,y,self.theta)

    def get_cost_history(self,X,y):
        return [cost_function(X,y,theta) for theta in self.theta_history]

    def predict(self,z):
        return self.theta[0]+self.theta[1]*z
