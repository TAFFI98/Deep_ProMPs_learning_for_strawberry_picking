import gym
import time
from ProMP.utils import plot_sampled_trajectories
import numpy as np
from ProMP.ProMP_framework import ProMP
import panda_gym
import cv2
np.random.seed(6)
def main():
    env = gym.make('panda-gym-v0',GUI_on=True)
    for i in range(0,1):

        ''' Reset the environment '''
        initial_observation, RGB_home, DEPTH_home= env.reset()
        print( "----- INITIAL OBSERVATION -----")
        print(" Strawberry position: ", initial_observation[-3:])
        print(" EE position: ", initial_observation[0:3])
        print(" EE orientation: ", initial_observation[3:6])

        ''' Show RGB image'''
        #cv2.imshow('RGB image', cv2.cvtColor(RGB_home, cv2.COLOR_BGR2RGB))
        #cv2.waitKey()

        ''' ProMP intialization'''
        N_BASIS = 8
        N_DOF = 1
        N_JOINTS = 9
        N_T = 150
        promp = ProMP(N_BASIS, N_DOF, N_T)

        ''' Predict mean and covariance of ProMP weights '''
        weights_mean_pred, weights_cov_pred = env.predict_weights_mean_and_covariance(RGB_home,DEPTH_home)

        ''' Sample weights'''
        theta_sampled = weights_mean_pred
        # theta_sampled = np.empty(shape=weights_mean_pred.shape)
        #
        # for j in range(N_JOINTS):
        #      theta_sampled[j, :] = np.random.multivariate_normal(weights_mean_pred[j, :], weights_cov_pred[j, :,:], 1)


        theta_sampled[0, :] = np.asarray([   0.96272227   ,-3.07597423   , 5.2272431  ,  -5.17788843  ,  3.14103524,
    -0.49190963 ,  -0.69864367  ,  0.58515826])
        theta_sampled[1, :] = np.asarray( [  -1.31811883  , -1.60885413   , 8.94047759  ,-17.57065217 ,  23.36652378,
   -21.20109123 ,  13.86443629 ,  -4.8841831 ])
        theta_sampled[2, :] = np.asarray([  -5.47390793 ,  21.91985018 , -44.42364004  , 60.32668178 , -57.95701512,
    40.14997575  ,-17.67112393 ,   3.72785836])
        theta_sampled[3, :] = np.asarray( [  -5.44394375  ,  2.42676067  , -1.54869403 ,  -5.90515226  ,  5.602792,
    -9.19868656   , 4.2731447  ,  -3.9038881 ])
        theta_sampled[4, :] = np.asarray([   8.68523978,  -31.96847354   ,68.57178003 ,-106.07284214 , 127.50466189,
  -116.78696606  , 76.54433566 , -27.36699449])
        theta_sampled[5, :] = np.asarray(  [   3.90045633   , 0.63051991  ,  8.92024704  , -7.74309628  , 17.39602992,
   -10.85795189  , 12.79939897  , -0.92650112])
        theta_sampled[6, :] = np.asarray(  [  18.99935339 , -62.77380476  ,126.0453975  ,-175.08534848 , 178.87169136,
  -135.76926138  , 75.78147156 , -23.88649916])


        ''' Reconstruct trajectory '''
        sampled_trajectories = np.zeros(shape=(N_JOINTS,N_T))
        for j in range(N_JOINTS):
            sampled_trajectories[j,:] = np.squeeze(promp.trajectory_from_weights(theta_sampled[j,:], vector_output=False))

        ''' Plot the reconstructed trajectory'''
        plot_sampled_trajectories(save_path='/Users/alessandratafuro/Desktop/', sampled_traj=sampled_trajectories, show=False, save=False)

        ''' Compute the reward '''
        policy_final_reward = 0
        for i in range(N_T):
            target_pos = sampled_trajectories[:,i]
            observation, reward, done, info = env.step(target_pos)
            print('reward: ', reward)
            print(info)
            policy_final_reward += reward
            time.sleep(env.timeStep)
        print('Reward of mean policy:   ', policy_final_reward)

    ''' Close the environment'''
    env.close()

if __name__ == '__main__':
    main()