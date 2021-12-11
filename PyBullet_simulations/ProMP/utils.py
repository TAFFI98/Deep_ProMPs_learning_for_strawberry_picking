import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt
from pathlib import Path
import os
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_probability.python as tfp
tf.keras.backend.set_floatx('float64')

def plot_traj_distribution_1_joint(save_path,config, mean_traj_1,traj_std_1,mean_traj_2=np.zeros(shape=(150,)),traj_std_2=np.zeros(shape=(150,)),show=False,save=False):
    '''
    PLOT OF THE TRAJECTORY DISTRIBUTION FOR 1 JOINT
    '''
    #TRAJ 1
    traj_true_mean = mean_traj_1
    right_traj_1_true = mean_traj_1+traj_std_1
    left_traj_1_true = mean_traj_1-traj_std_1
    right_traj_3_true=mean_traj_1+3*traj_std_1
    left_traj_3_true = mean_traj_1-3*traj_std_1

    #TRAJ 2
    if  mean_traj_2.all() != 0:
        traj_pred_mean=mean_traj_2
        right_traj_1_pred = mean_traj_2+traj_std_2
        left_traj_1_pred = mean_traj_2-traj_std_2
        right_traj_3_pred =mean_traj_2+3*traj_std_2
        left_traj_3_pred = mean_traj_2-3*traj_std_2

    q1true= traj_true_mean
    q1righttrue = right_traj_3_true
    q1lefttrue = left_traj_3_true
    q1righttrue_1 = right_traj_1_true
    q1lefttrue_1= left_traj_1_true
    if  mean_traj_2.all() != 0:
        q1pred = traj_pred_mean
        q1rightpred = right_traj_3_pred
        q1leftpred = left_traj_3_pred
        q1rightpred_1= right_traj_1_pred
        q1leftpred_1 = left_traj_1_pred

    fig= plt.figure(figsize=(8, 3))
    fig.suptitle('Trajectories distributions configuration  ' + str(config), fontweight="bold")
    # Q1
    x = np.linspace(0, 150, 150)
    plt.plot(q1true, 'c', label='', linewidth=0.5)
    plt.plot(q1righttrue, 'b', linewidth=0.5)
    plt.plot(q1lefttrue, 'b', linewidth=0.5)
    plt.plot(q1righttrue_1, 'b', linewidth=0.5)
    plt.plot(q1lefttrue_1, 'b', linewidth=0.5)
    plt.fill_between(x, q1righttrue, q1lefttrue, alpha=0.25,facecolor='blue')
    plt.fill_between(x, q1righttrue_1, q1lefttrue_1, alpha=0.25,facecolor='blue')

    if mean_traj_2.all() != 0:
        plt.plot(q1pred, 'r', label='', linewidth=0.5)
        plt.plot(q1rightpred, 'm', linewidth=0.5)
        plt.plot(q1leftpred, 'm', linewidth=0.5)
        plt.plot(q1rightpred_1, 'm', linewidth=0.5)
        plt.plot(q1leftpred_1, 'm', linewidth=0.5)
        plt.fill_between(x, q1rightpred.reshape(150, ), q1leftpred.reshape(150, ), alpha=0.25,facecolor=(1, 0, 0, .4))
        plt.fill_between(x, q1rightpred_1.reshape(150, ), q1leftpred_1.reshape(150, ), alpha=0.25,facecolor=(1, 0, 0, .4))

    fig.set_dpi(200)
    if show == True:
        plt.show()  # Show the image.
    if save == True:
        # Create the path for saving plots.
        Path(save_path).mkdir(exist_ok=True, parents=True)
        plt.savefig(save_path+'/joint_traj_distrib_'+str(config) +'.png')

def LD_from_elements_pred(lower_tri_elements: tf.Tensor):
    """
    Given the tensor with the elements of the fake triangular matrix, obtain L and D.
    L and D are the matrices of LDL decomposition.
    """
    # Build the triangular matrix from the elements.
    lower_tri = tfp.math.fill_triangular(lower_tri_elements)
    # Extract the diagonal values.
    d = tf.linalg.band_part(lower_tri, 0, 0)
    # Exploit the diagonal to compute an identity matrix with the correct size.
    identity = tf.linalg.diag(tf.linalg.diag_part(d) / tf.linalg.diag_part(d))
    # Compute the L matrix (lower uni-traingular).
    L = lower_tri - d + identity
    # Compute the D matrix (diagonal).
    # Note. Since d is diagonal exp(d) causes all 0 values to become 1. We address that by removing 1 from
    # all elements, then adding 1 back to the diagonal elements with the identity.
    D = tf.math.exp(d) - tf.ones_like(d) + identity
    return L, D

def LD_from_elements_true(lower_tri_elements: tf.Tensor):
    """
    Given the tensor with the elements of the fake triangular matrix, obtain L and D.
    L and D are the matrices of LDL decomposition.
    """

    # Build the triangular matrix from the elements.
    lower_tri = tfp.math.fill_triangular(lower_tri_elements)
    # Extract the diagonal values.
    d = tf.linalg.band_part(lower_tri, 0, 0)
    # Exploit the diagonal to compute an identity matrix with the correct size.
    identity = tf.linalg.diag(tf.linalg.diag_part(d) / tf.linalg.diag_part(d))
    # Compute the L matrix (lower uni-traingular).
    L = lower_tri - d + identity
    # Compute the D matrix (diagonal).
    # Note. Since d is diagonal exp(d) causes all 0 values to become 1. We address that by removing 1 from
    # all elements, then adding 1 back to the diagonal elements with the identity.
    D = d
    return L, D

def LD_reconstruct(L: tf.Tensor, D: tf.Tensor):
    """Re-compose the matrix from L and D decomposition.
    L is lower uni-triangular.
    D is diagonal.
    """
    cov = tf.matmul(L, tf.matmul(D, tf.transpose(L, (0, 2, 1))))
    return cov



def RMSE(promp):
    '''
    LOSS FUNCTION USED TO TRAIN THE DEEP MODELS
    '''
    all_phi = tf.cast(promp.all_phi(), 'float64')
    def RMSE_loss(ytrue, ypred):

        mean_true = ytrue[..., 0:-36]               # (n_batch, 56)
        lower_tri_elements_true = ytrue[..., -36:]  # (n_batch, 1596)
        mean_pred = ypred[..., 0:-36]              # (n_batch, 56)
        lower_tri_elements_pred= ypred[..., -36:]  # (n_batch, 1596)

        mean_traj_true=tf.transpose(tf.matmul(all_phi, tf.transpose(mean_true)))
        mean_traj_pred=tf.transpose(tf.matmul(all_phi, tf.transpose(mean_pred)))

        L_true, D_true= LD_from_elements_true(lower_tri_elements_true)
        L_pred, D_pred= LD_from_elements_pred(lower_tri_elements_pred)

        cov_true = LD_reconstruct(L_true, D_true)
        cov_pred = LD_reconstruct(L_pred, D_pred)

        cov_traj_true =tf.linalg.matmul(cov_true, tf.transpose(all_phi))
        cov_traj_true=tf.linalg.matmul(tf.transpose(cov_traj_true, perm=[0, 2, 1]), tf.transpose(all_phi))
        cov_traj_pred =tf.linalg.matmul(cov_pred, tf.transpose(all_phi))
        cov_traj_pred=tf.linalg.matmul(tf.transpose(cov_traj_pred, perm=[0, 2, 1]), tf.transpose(all_phi))

        loss3 = K.sqrt(K.mean(K.square(mean_traj_true - mean_traj_pred)))
        loss4 = K.sqrt(K.mean(K.square(cov_traj_true - cov_traj_pred)))
        return K.mean(loss3)+K.mean(loss4)

    return RMSE_loss

def is_pos_def(x,tol=0):
    '''
    RETURNS A BOOLEAN THAT DEFINES IF A MATRIX IS OR NOT POSITIVE DEFINITE
    '''
    return np.all(np.linalg.eigvals(x) > tol)

def plot_sampled_trajectories(save_path, sampled_traj,show=False,save=False):
    traj_true_mean = sampled_traj
    q1true, q2true, q3true, q4true, q5true, q6true, q7true = traj_true_mean[0,:], traj_true_mean[1,:], traj_true_mean[2,:], traj_true_mean[3,:],traj_true_mean[4,:], traj_true_mean[5,:], traj_true_mean[6,:]
    fig, axarr = plt.subplots(2,4)
    axarr[0, 0].tick_params(axis='both', labelsize=5)
    axarr[0, 1].tick_params(axis='both', labelsize=5)
    axarr[0, 2].tick_params(axis='both', labelsize=5)
    axarr[0, 3].tick_params(axis='both', labelsize=5)
    axarr[1, 0].tick_params(axis='both', labelsize=5)
    axarr[1, 1].tick_params(axis='both', labelsize=5)
    axarr[1, 2].tick_params(axis='both', labelsize=5)
    axarr[1, 3].tick_params(axis='both', labelsize=5)
    fig.suptitle('Samples from the predicted distributions', fontweight="bold")

    # Q1
    plt.sca(axarr[0, 0])
    plt.plot(q1true, 'c', label='q1', linewidth=0.5)
    plt.legend(loc=1, fontsize='x-small')

    # Q2
    plt.sca(axarr[0, 1])
    plt.plot(q2true, 'c', label='q2', linewidth=0.5)
    plt.legend(loc=1, fontsize='x-small')

    # Q3
    plt.sca(axarr[0, 2])
    plt.plot(q3true, 'c', label='q3', linewidth=0.5)
    plt.legend(loc=1, fontsize='x-small')

    # Q4
    plt.sca(axarr[0, 3])
    plt.plot(q4true, 'c', label='q4', linewidth=0.5)
    plt.legend(loc=1, fontsize='x-small')

    # Q5
    plt.sca(axarr[1, 0])
    plt.plot(q5true, 'c', label='q5', linewidth=0.5)
    plt.legend(loc=1, fontsize='x-small')

    # Q6
    plt.sca(axarr[1, 1])
    plt.plot(q5true, 'c', label='q6', linewidth=0.5)
    plt.legend(loc=1, fontsize='x-small')

    # Q7
    plt.sca(axarr[1, 2])
    plt.plot(q7true, 'c', label='q7', linewidth=0.5)
    plt.legend(loc=1, fontsize='x-small')

    if show == True:
        plt.show()  # Show the image.
    if save == True:
        # Create the path for saving plots.
        Path(save_path).mkdir(exist_ok=True, parents=True)
        plt.savefig(save_path + 'joint_traj_samples' + '.png')
