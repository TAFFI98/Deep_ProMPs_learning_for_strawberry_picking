import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_probability as tfp
tfd = tfp.distributions
from LDL_decomposition import LD_reconstruct,LD_from_elements_pred,LD_from_elements_true
import scipy
from scipy.spatial import distance
from statsmodels.stats.correlation_tools import cov_nearest

#np.set_printoptions(threshold=np.inf)
tf.config.run_functions_eagerly(True)
tf.keras.backend.set_floatx('float64')
def KL():

    def KL_loss(ytrue, ypred):

        mean_true = ytrue[..., 0:-36]               # (n_batch, 56)
        lower_tri_elements_true = ytrue[..., -36:]  # (n_batch, 1596)

        mean_pred = ypred[..., 0:-36]              # (n_batch, 56)
        lower_tri_elements_pred= ypred[..., -36:]  # (n_batch, 1596)

        L_true, D_true = LD_from_elements_true(lower_tri_elements_true)
        L_pred, D_pred = LD_from_elements_pred(lower_tri_elements_pred)

        cov_true = LD_reconstruct(L_true, D_true)
        cov_pred = LD_reconstruct(L_pred, D_pred)

        dist_pred = tfd.MultivariateNormalTriL(mean_pred,tf.linalg.cholesky(cov_pred), allow_nan_stats=False)
        dist_true = tfd.MultivariateNormalTriL(mean_true,tf.linalg.cholesky(cov_true), allow_nan_stats=False)#.sample(1000)
        # log_likelihood = dist_pred.log_prob(dist_true)
        # return - tf.math.reduce_mean(log_likelihood,axis=-1)
        return tfd.kl_divergence(dist_true, dist_pred)


    return KL_loss


def KL_traj(promp):

    def KL_loss_traj(ytrue, ypred):
        all_phi = tf.cast(promp.all_phi(), 'float64')
        mean_true = ytrue[..., 0:-36]               # (n_batch, 56)
        lower_tri_elements_true = ytrue[..., -36:]  # (n_batch, 1596)

        mean_pred = ypred[..., 0:-36]              # (n_batch, 56)
        lower_tri_elements_pred= ypred[..., -36:]  # (n_batch, 1596)

        L_true, D_true= LD_from_elements_true(lower_tri_elements_true)
        L_pred, D_pred= LD_from_elements_pred(lower_tri_elements_pred)

        cov_true = LD_reconstruct(L_true, D_true)
        cov_pred = LD_reconstruct(L_pred, D_pred)

        mean_traj_true=tf.transpose(tf.matmul(all_phi, tf.transpose(mean_true)))
        mean_traj_pred=tf.transpose(tf.matmul(all_phi, tf.transpose(mean_pred)))

        cov_traj_true =tf.linalg.matmul(cov_true, tf.transpose(all_phi))
        cov_traj_true=tf.linalg.matmul(tf.transpose(cov_traj_true, perm=[0, 2, 1]), tf.transpose(all_phi))
        cov_traj_pred =tf.linalg.matmul(cov_pred, tf.transpose(all_phi))
        cov_traj_pred=tf.linalg.matmul(tf.transpose(cov_traj_pred, perm=[0, 2, 1]), tf.transpose(all_phi))

        dist_pred = tfd.MultivariateNormalTriL(mean_traj_pred,tf.linalg.cholesky(cov_traj_pred), allow_nan_stats=False)
        dist_true = tfd.MultivariateNormalTriL(mean_traj_true,tf.linalg.cholesky(cov_traj_true), allow_nan_stats=False)#.sample(100)
        # log_likelihood = dist_pred.log_prob(dist_true)
        # return - tf.math.reduce_mean(log_likelihood,axis=-1)
        return tfd.kl_divergence(dist_true, dist_pred)

    return KL_loss_traj

def KL_custom():

    def KL_loss_custom(ytrue, ypred):

        mean_true = ytrue[..., 0:-36]               # (n_batch, 56)
        lower_tri_elements_true = ytrue[..., -36:]  # (n_batch, 1596)

        mean_pred = ypred[..., 0:-36]              # (n_batch, 56)
        lower_tri_elements_pred= ypred[..., -36:]  # (n_batch, 1596)

        L_true, D_true = LD_from_elements_true(lower_tri_elements_true)
        L_pred, D_pred = LD_from_elements_pred(lower_tri_elements_pred)

        cov_true = LD_reconstruct(L_true, D_true)
        cov_pred = LD_reconstruct(L_pred, D_pred)

        a = tf.math.log(tf.linalg.det(cov_pred)/tf.linalg.det(cov_true))            #(n_batch, 1)
        b = tf.linalg.trace(tf.linalg.matmul(tf.linalg.inv(cov_pred), cov_true))    #(n_batch, 1)
        e = tf.expand_dims(mean_pred-mean_true,axis=-1)                                               # (n_batch, 56,1)
        c = tf.transpose(e,perm=[0, 2, 1])                                          # (n_batch, 1,56)
        d = tf.linalg.inv(cov_pred)                                                 # (n_batch, 56, 56)

        loss=0.5 *(a-56+b+tf.linalg.matmul(c,tf.linalg.matmul(d,e)))
        return loss

    return KL_loss_custom

def RMSE(promp):
    all_phi = tf.cast(promp.all_phi(), 'float64')
    def RMSE_loss(ytrue, ypred):

        mean_true = ytrue              # (n_batch, 56)
        mean_pred = ypred           # (n_batch, 56)


        mean_traj_true=tf.transpose(tf.matmul(all_phi, tf.transpose(mean_true)))
        mean_traj_pred=tf.transpose(tf.matmul(all_phi, tf.transpose(mean_pred)))

        loss = K.sqrt(K.mean(K.square(mean_traj_true - mean_traj_pred)))

        return K.mean(loss)

    return RMSE_loss