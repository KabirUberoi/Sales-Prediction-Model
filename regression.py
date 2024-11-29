import numpy as np
import pandas as pd
from queue import Queue as Q
import matplotlib.pyplot as plt


class base_regressor:
    def __init__(self, batch_size, learning_rate, num_epochs, k=6, tolerance=1e-5, theta=None):
        self.batch_size=batch_size
        self.learning_rate=learning_rate
        self.theta=theta
        self.num_epochs=num_epochs
        self.k=k
        self.tolerance=tolerance

    def predict(self, X):
        return np.dot(X, self.theta)

    def make_batches(self):
        batch_size=self.batch_size
        m=self.X.shape[0]
        data=np.hstack((self.X, self.y))
        # np.random.shuffle(data)
        self.batches=[]
        for i in range(m//batch_size):
            batch=data[batch_size*i: batch_size*(i+1), :]
            self.batches.append(np.hsplit(batch, [batch.shape[1]-1]))

        if(m%batch_size!=0):
            batch=data[(batch_size*(m//batch_size)):, :]
            self.batches.append(np.hsplit(batch, [batch.shape[1]-1]))

    def run_epoch(self):
        losses=[]
        for batch in self.batches:
            [data_X, data_y]=batch[0], batch[1]
            yhat=np.dot(data_X, self.theta)
            J_theta, dtheta=self.calculate_loss_and_gradient(data_X, data_y, yhat)
            losses.append(J_theta)
            # dtheta=np.dot(data_X.T, (yhat-data_y))/m
            self.theta=self.theta-self.learning_rate*dtheta
        return np.average(losses)

    def calculate_loss_and_gradient(self, data_X, data_y, yhat):
        pass

    def evaluate_loss(self):
        pass

    def evaluate_val_loss(self):
        pass

    def preprocess(self):
        self.theta=np.zeros((self.X.shape[1], 1))


    def fit(self, X, y, X_test=None, y_test=None):
        print(X_test.shape)
        self.X=X
        self.y=y
        self.X_test=X_test
        self.y_test=y_test
        train_losses=[]
        val_losses=[]
        self.preprocess()
        q=Q()
        sum_of_diff=0
        epoch_loss=self.evaluate_loss()
        train_losses=[np.log(epoch_loss)]
        if self.X_test.any(): 
            val_loss=self.evaluate_val_loss()
            val_losses.append(np.log(val_loss))
        self.make_batches()
        epoch_count=0
        for _ in range(self.k):
            J_old=epoch_loss
            epoch_loss=self.run_epoch()
            train_losses.append(epoch_loss)
            if self.X_test.any():
                val_loss=self.evaluate_val_loss()
                val_losses.append(val_loss)
            diff=abs(J_old-epoch_loss)
            q.put(diff)
            sum_of_diff+=diff

        epoch_count=self.k


        while (sum_of_diff/self.k >= self.tolerance) and (epoch_count<self.num_epochs):

            J_old=epoch_loss
            epoch_loss=self.run_epoch()
            train_losses.append(np.log(epoch_loss))
            if self.X_test.any():
                val_loss=self.evaluate_val_loss()
                val_losses.append(np.log(val_loss))
            diff=abs(J_old-epoch_loss)
            q.put(diff)
            sum_of_diff+=diff
            sum_of_diff-=q.get()
            epoch_count+=1
            # if(epoch_count%100==0): print(epoch_count, sum_of_diff/k, theta)
        print("===========", epoch_count, "================")

        plt.plot(range(epoch_count+1), train_losses, label='Training Loss')
        if self.X_test.any(): plt.plot(range(epoch_count+1), val_losses, label='Validation Loss')
        plt.legend()


class linear_regressor(base_regressor):
    def __init__(self, batch_size=32, learning_rate=0.1, num_epochs=20, k=6, tolerance=1e-5):
        super().__init__(batch_size, learning_rate, num_epochs, k, tolerance)

    def evaluate_loss(self):
        m=self.X.shape[0]
        yhat=np.dot(self.X, self.theta)
        J_theta=np.sum((yhat-self.y)**2)/(2*m)

        # print(1-np.sum((y-yhat)**2)/np.sum((y-np.mean(y))**2))

        return J_theta
    
    def evaluate_val_loss(self):
        y_hat=np.dot(self.X_test, self.theta)
        J_theta=np.sum((y_hat-self.y_test)**2)/(2*self.X_test.shape[0])

        return J_theta

    def calculate_loss_and_gradient(self, data_X, data_y, yhat):
        m=data_X.shape[0]
        yhat=np.dot(data_X, self.theta)
        difference=yhat-data_y
        J_theta=np.sum((difference)**2)/(2*m)
        dtheta=np.dot(data_X.T, (difference))/m

        return J_theta, dtheta

class ridge_regressor(base_regressor):
    def __init__(self, batch_size=32, learning_rate=0.1, num_epochs=20, k=6, tolerance=1e-5, lamda=0.1):
        super().__init__(batch_size, learning_rate, num_epochs, k, tolerance)
        self.lamda=lamda

    def evaluate_loss(self):
        m=self.X.shape[0]
        yhat=np.dot(self.X, self.theta)
        J_theta=np.sum((yhat-self.y)**2)/(2*m) + self.lamda*np.sum(self.theta**2)/2

        # print(1-np.sum((y-yhat)**2)/np.sum((y-np.mean(y))**2))

        return J_theta
    
    def evaluate_val_loss(self):
        y_hat=np.dot(self.X_test, self.theta)
        J_theta=np.sum((y_hat-self.y_test)**2)/(2*self.X_test.shape[0]) + self.lamda*np.sum(self.theta**2)/2

        return J_theta

    def calculate_loss_and_gradient(self, data_X, data_y, yhat):
        m=data_X.shape[0]
        yhat=np.dot(data_X, self.theta)
        difference=yhat-data_y
        J_theta=np.sum((difference)**2)/(2*m) + self.lamda*np.sum(self.theta**2)/2
        dtheta=np.dot(data_X.T, (difference))/m + self.lamda*self.theta

        return J_theta, dtheta

class quadratic_regressor(base_regressor):
    def __init__(self, batch_size=32, learning_rate=0.1, num_epochs=32, k=6, tolerance=1e-5):
        super().__init__(batch_size, learning_rate, num_epochs, k, tolerance)

    def preprocess(self):
        self.X=np.hstack((self.X, self.X**2))
        if self.X_test!=None: self.X_test=np.hstack((self.X_test, self.X_test**2))
        self.theta=np.zeros((self.X.shape[1], 1))

    def evaluate_loss(self):
        m=self.X.shape[0]
        yhat=np.dot(self.X, self.theta)
        J_theta=np.sum((yhat-self.y)**2)/(2*m)

        # print(1-np.sum((y-yhat)**2)/np.sum((y-np.mean(y))**2))

        return J_theta
    
    def evaluate_val_loss(self):
        y_hat=np.dot(self.X_test, self.theta)
        J_theta=np.sum((y_hat-self.y_test)**2)/(2*self.X_test.shape[0])

        return J_theta

    def calculate_loss_and_gradient(self, data_X, data_y, yhat):
        m=data_X.shape[0]
        yhat=np.dot(data_X, self.theta)
        difference=yhat-data_y
        J_theta=np.sum((difference)**2)/(2*m)
        dtheta=np.dot(data_X.T, (difference))/m

        return J_theta, dtheta

    def predict(self, X):
        X=np.hstack((X, X**2))
        return np.dot(X, self.theta)
