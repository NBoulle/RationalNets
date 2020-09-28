"""

@author: Nicolas Boulle
Code modified from https://github.com/maziarraissi/DeepHPMs written by Maziar Raissi

Approximate the solution of the KdV equation by a neural network
Run the following command to use rational activation functions of type (3,2)
python3 KdV_rat.py --rP 3 --rQ 2

"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os
from scipy.interpolate import griddata
from plotting import newfig, savefig
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import random
import argparse

# Set seeds 10/10/10
random.seed(10)
np.random.seed(10)
tf.random.set_random_seed(10)

###############################################################################
############################## Helper Functions ###############################
###############################################################################

def initialize_NN(layers):
    weights = []
    biases = []
    num_layers = len(layers) 
    for l in range(0,num_layers-1):
        W = xavier_init(size=[layers[l], layers[l+1]])
        b = tf.Variable(tf.zeros([1,layers[l+1]]), dtype=tf.float32)
        weights.append(W)
        biases.append(b)        
    return weights, biases
    
def xavier_init(size):
    in_dim = size[0]
    out_dim = size[1]        
    xavier_stddev = np.sqrt(2/(in_dim + out_dim))
    return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

def rat_init(layers, initP, initQ):
    num_layers = len(layers)
    ratweightsP = []
    ratweightsQ = []    
    for l in range(0,num_layers-2):
        RP = [tf.Variable([coeffs], dtype=tf.float32) for coeffs in initP]
        ratweightsP = ratweightsP + RP
        RQ = [tf.Variable([coeffs], dtype=tf.float32) for coeffs in initQ]
        ratweightsQ = ratweightsQ + RQ    
    
    return ratweightsP, ratweightsQ

def neural_net(X, weights, biases, ratweightsP, ratweightsQ):
    num_layers = len(weights) + 1
    H = X
    degP = int(len(ratweightsP)/(num_layers-2)-1)
    degQ = int(len(ratweightsQ)/(num_layers-2)-1)
    for l in range(0,num_layers-2):
        W = weights[l]
        b = biases[l]
        H = tf.add(tf.matmul(H, W), b)
        H = tf.math.divide(tf.math.polyval(ratweightsP[(degP+1)*l:(degP+1)*l+(degP+1)],H),tf.math.polyval(ratweightsQ[(degQ+1)*l:(degQ+1)*l+(degQ+1)],H))
    W = weights[-1]
    b = biases[-1]
    Y = tf.add(tf.matmul(H, W), b)
    return Y

###############################################################################
################################ DeepHPM Class ################################
###############################################################################

class DeepHPM:
    def __init__(self, t_train, x_train, u_train, t_val, x_val, u_val, u_layers, lb_idn, ub_idn, rP, rQ):
        
        # Domain Boundary
        self.lb_idn = lb_idn
        self.ub_idn = ub_idn
        
        # Init for Identification
        self.idn_init(t_train, x_train, u_train, t_val, x_val, u_val, u_layers, rP, rQ)
            
        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        init = tf.global_variables_initializer()
        self.sess.run(init)
        
    ###########################################################################
    ############################# Identifier ##################################
    ###########################################################################
        
    def idn_init(self, t_train, x_train, u_train, t_val, x_val, u_val, u_layers, rP, rQ):
        # Training Data for Identification
        self.t = t_train
        self.x = x_train
        self.u = u_train
        
        # Validation data
        self.t_val = t_val
        self.x_val = x_val
        self.u_val = u_val
        
        # Layers for Identification
        self.u_layers = u_layers
        
        # Initialize NNs for Identification
        self.u_weights, self.u_biases = initialize_NN(u_layers)
        
        # Initialize rational layers with approximation to ReLU
        self.rP = rP
        self.rQ = rQ
        # Type 2/2
        if rP == 2 and rQ == 2:
            initP = [0.5, 0.4282, 0.0359]
            initQ = [1.0, -1.0, 1.0]
        # Type 3/2
        elif rP == 3 and rQ == 2:
            initP = [1.1915, 1.5957, 0.5, 0.0218]
            initQ = [2.383, 0.0, 1.0]
        # Type 4/3
        elif rP == 4 and rQ == 3:
            initP = [5.2361, 8.4721, 3.8906, 0.5]
            initQ = [0.0, 16.9443, 0.0, 1.0]
        # Type 5/4
        elif rP == 5 and rQ == 4:
            initP = [37.5756, 69.9321, 40.9101, 8.3779, 0.5, 0.0043]
            initQ = [75.1512, -0.0, 81.8202, 0.0, 1.0]
        # Type not implemented 
        else:
            raise ValueError("Initialization coefficients for type (%d, %d) rationals not implemented. Run the MATLAB code initial_rational_coeffs.m to find the coefficients.")
        
        self.ratweightsP, self.ratweightsQ = rat_init(u_layers, initP, initQ)

        # tf placeholders for Identification
        self.t_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        self.x_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        self.u_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        
        # tf placeholders for Validation
        self.t_val_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        self.x_val_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        self.u_val_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        
        # tf graphs for Identification
        self.idn_u_pred = self.idn_net_u(self.t_tf, self.x_tf)
        self.idn_u_pred_val = self.idn_net_u(self.t_val_tf, self.x_val_tf)
        
        # Training and validation losses for Identification
        self.LossArray = []
        self.ValArray = []
        self.idn_u_loss = tf.reduce_mean(tf.square(self.idn_u_pred - self.u_tf))
        self.idn_u_val = tf.reduce_mean(tf.square(self.idn_u_pred_val - self.u_val_tf))
        
        # Optimizer for Identification
        self.idn_u_optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.idn_u_loss,
                                var_list = self.u_weights + self.u_biases + self.ratweightsP + self.ratweightsQ,
                                method = 'L-BFGS-B',
                                options = {'maxiter': 10**4,
                                          'iprint':10,
                                          'maxcor': 50,
                                          'maxls': 50,
                                          'ftol': 1e-20,
                                          'gtol': 1.0*np.finfo(float).eps})
        
    
    def idn_net_u(self, t, x):
        X = tf.concat([t,x],1)
        H = 2.0*(X - self.lb_idn)/(self.ub_idn - self.lb_idn) - 1.0
        u = neural_net(H, self.u_weights, self.u_biases, self.ratweightsP, self.ratweightsQ)
        return u
    
    def callback(self, loss, validation):
        # Save training loss
        self.LossArray = self.LossArray + [loss]
        
        # Save validation loss
        self.ValArray = self.ValArray + [validation]
        
    def save_loss(self):
        its = [i for i in range(len(self.LossArray))]
        L = np.vstack((its, self.LossArray)).transpose()
        np.savetxt("Results/Rational_%d_%d/loss_rat.csv" % (self.rP,self.rQ), L, delimiter=',')
        
        # Validation loss
        L = np.vstack((its, self.ValArray)).transpose()
        np.savetxt("Results/Rational_%d_%d/val_rat.csv" % (self.rP,self.rQ), L, delimiter=',')
    
    def idn_u_train(self):
        tf_dict = {self.t_tf: self.t, self.x_tf: self.x, self.u_tf: self.u, 
                   self.t_val_tf: self.t_val, self.x_val_tf: self.x_val, self.u_val_tf: self.u_val}
        self.idn_u_optimizer.minimize(self.sess,
                                      feed_dict = tf_dict,
                                      fetches = [self.idn_u_loss, self.idn_u_val],
                                      loss_callback = self.callback)
   
    def idn_predict(self, t_star, x_star):
        tf_dict = {self.t_tf: t_star, self.x_tf: x_star}
        u_star = self.sess.run(self.idn_u_pred, tf_dict)
        return u_star

###############################################################################
################################ Main Function ################################
###############################################################################

if __name__ == "__main__": 
    
    # Get the type of the rational
    parser = argparse.ArgumentParser()
    parser.add_argument("--rP", type=int, default = 3)
    parser.add_argument("--rQ", type=int, default = 2)
    args, _ = parser.parse_known_args()
    rP = args.rP
    rQ = args.rQ
    
    # Create results folder
    if not os.path.exists("Results/Rational_%d_%d" % (rP,rQ)):
        os.makedirs("Results/Rational_%d_%d" % (rP,rQ))

    # Doman bounds
    lb_idn = np.array([0.0, -20.0])
    ub_idn = np.array([40.0, 20.0])
    
    ### Load Data ###
    
    data_idn = scipy.io.loadmat('Data/KdV_sine.mat')
    
    t_idn = data_idn['t'].flatten()[:,None]
    x_idn = data_idn['x'].flatten()[:,None]
    Exact_idn = np.real(data_idn['usol'])
    
    T_idn, X_idn = np.meshgrid(t_idn,x_idn)
    
    keep = 1
    index = int(keep*t_idn.shape[0])
    T_idn = T_idn[:,0:index]
    X_idn = X_idn[:,0:index]
    Exact_idn = Exact_idn[:,0:index]
    
    t_idn_star = T_idn.flatten()[:,None]
    x_idn_star = X_idn.flatten()[:,None]
    X_idn_star = np.hstack((t_idn_star, x_idn_star))
    u_idn_star = Exact_idn.flatten()[:,None]
    
    
    # Save exact as CSV file
    np.savetxt("Results/Rational_%d_%d/domain.csv" % (rP,rQ), X_idn_star, delimiter=',')
    np.savetxt("Results/Rational_%d_%d/exact_sol.csv" % (rP,rQ), Exact_idn, delimiter=',')
     
    ### Training Data ###
    
    # For identification and validation: 10^4 each
    N_train = 10**4
    N_val = 10**4
    idx = np.random.choice(t_idn_star.shape[0], N_train+N_val, replace=False)    
    idx_train = idx[0:N_train]
    idx_val = idx[N_train:]
    t_train = t_idn_star[idx_train,:]
    x_train = x_idn_star[idx_train,:]
    u_train = u_idn_star[idx_train,:]
    t_val = t_idn_star[idx_val,:]
    x_val = x_idn_star[idx_val,:]
    u_val = u_idn_star[idx_val,:]
    
    noise = 0.00
    u_train = u_train + noise*np.std(u_train)*np.random.randn(u_train.shape[0], u_train.shape[1])
        
    # Layers
    u_layers = [2, 50, 50, 50, 50, 1]
    
    # Model
    model = DeepHPM(t_train, x_train, u_train, t_val, x_val, u_val, u_layers, lb_idn, ub_idn, rP, rQ)
        
    # Train the identifier
    model.idn_u_train()
    model.save_loss()
    
    u_pred_identifier = model.idn_predict(t_idn_star, x_idn_star)
    
    error_u_identifier = np.mean((u_idn_star-u_pred_identifier)**2)
    print('Mean Squared Error: %e' % (error_u_identifier))
    
    U_pred = griddata(X_idn_star, u_pred_identifier.flatten(), (T_idn, X_idn), method='cubic')    
    
    # Save identifier as CSV file
    np.savetxt("Results/Rational_%d_%d/idn_rat.csv" % (rP,rQ), U_pred, delimiter=',')
    
    ######################################################################
    ############################# Plotting ###############################
    ######################################################################    
    
    fig, ax = newfig(1.0, 0.6)
    ax.axis('off')
    
    ######## Exact solution #######################
    ########      Predicted p(t,x,y)     ########### 
    gs = gridspec.GridSpec(1, 2)
    gs.update(top=0.8, bottom=0.2, left=0.1, right=0.9, wspace=0.5)
    ax = plt.subplot(gs[:, 0])
    h = ax.imshow(Exact_idn, interpolation='nearest', cmap='jet', 
                  extent=[lb_idn[0], ub_idn[0]*keep, lb_idn[1], ub_idn[1]],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(h, cax=cax)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.set_title('Exact Dynamics', fontsize = 10)
    
    ######## Approximation Error ########### 
    ax = plt.subplot(gs[:, 1])
    h = ax.imshow(abs(Exact_idn-U_pred), interpolation='nearest', cmap='jet', 
                  extent=[lb_idn[0], ub_idn[0]*keep, lb_idn[1], ub_idn[1]], 
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(h, cax=cax)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.set_title('Identifier Error', fontsize = 10)
    
    savefig('Results/Rational_%d_%d/KdV_idn_rat' % (rP,rQ))
