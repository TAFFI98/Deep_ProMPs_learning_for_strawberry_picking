import numpy as np
from ProMP import ProMP_framework
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
import time
import gym

class PoWER_class():
    def __init__(self):
        self.N_BASIS = 8
        self.N_T = 150
        self.N_DOF = 1
        self.N_JOINTS = 7
        self.N_ITERATIONS = 40
        self.promp = ProMP_framework.ProMP(self.N_BASIS, self.N_DOF, self.N_T)
        self.phi = self.promp.all_phi().T # (8,150)



    def PoWER_algorithm(self,predicted_mean_w,predicted_cov_w,env):
        '''
        predicted_mean_w comes from deep_proMPs and has shape (N_JOINTS,N_BASIS)
        predicted_cov_w comes from deep_proMPs and has shape (N_JOINTS,N_BASIS,N_BASIS)
        '''

        ''' INITIALIZE WEIGHTS OF TARGET POLICY '''
        theta_target = np.zeros(shape = (self.N_ITERATIONS+1,self.N_JOINTS,self.N_BASIS))
        # Set first initialization value
        theta_current = np.zeros(shape = (self.N_JOINTS,self.N_BASIS))
        for j in range(self.N_JOINTS):
            theta_current[j,:] = predicted_mean_w[j,:]
            theta_target[0,j,:] = theta_current[j,:]


        ''' INITIALIZE EXPLORATION COVARIANCE '''
        covariance_target = np.zeros(shape=(self.N_ITERATIONS + 1, self.N_JOINTS, self.N_BASIS, self.N_BASIS))
        # Set first initialization value
        covariance_current = np.zeros(shape = (self.N_JOINTS,self.N_BASIS, self.N_BASIS))
        for j in range(self.N_JOINTS):
            covariance_current[j, :, :] = predicted_cov_w[j, :,:]
            covariance_target[0, j, :, :] = covariance_current[j, :,:]


        ''' INITIALIZE W '''
        W = np.zeros(shape=(self.N_ITERATIONS+1,self.N_JOINTS, self.N_T, self.N_BASIS, self.N_BASIS))
        # Set first initialization value
        for j in range(self.N_JOINTS):
            for t in range(self.N_T):
                # Vector of basis functions activation at time t
                PHI = np.reshape(self.phi[:, t], newshape=(self.N_BASIS, 1))
                d = np.matmul(PHI.T, np.matmul(covariance_current[j, :, :], PHI))
                W[0, j, t, :, :] = np.matmul(PHI, PHI.T) * np.linalg.inv(d)


        ''' Initialize a matrix with the rewards at all timesteps of all rollouts '''
        reward_t_all_episodes = np.zeros(shape = (self.N_ITERATIONS,self.N_T))

        ''' Initialize a matrix with the value function at all timesteps of all rollouts '''
        Q_t_all_episodes = np.zeros(shape = (self.N_ITERATIONS,self.N_T))

        ''' Initialize a matrix with the epsilon values '''
        epsilon = np.zeros(shape = (self.N_ITERATIONS,self.N_JOINTS,self.N_BASIS))

        ''' Initialize a matrix with the finale rewards of all rollouts, used aa a look-up table for importance sampling '''
        final_reward_all_episodes = np.zeros(shape = (self.N_ITERATIONS,2) )

        ''' Reward of the current target policy initialization '''
        policy_reward = []
        ''' COMPUTE THE REWARD OF THE CURRENT TARGET POLICY '''
        init_obs = env.reset_panda()

        ''' PRINT THE INITIAL OBSERVATION  '''
        # print("----- INITIAL OBSERVATION TARGET POLICY-----")
        # print(" Strawberry position: ", init_obs[-3:])
        # print(" EE position: ", init_obs[0:3])
        # print(" EE orientation: ", init_obs[3:6])

        # Compute the rollout from the current value of the target weights
        policy_rollout = np.zeros(shape=(self.N_JOINTS, self.N_T))
        for j in range(self.N_JOINTS):
            policy_rollout[j, :] = np.squeeze(
                self.promp.trajectory_from_weights(theta_current[j, :], vector_output=False))
        # Compute the relative reward
        policy_final_reward = 0
        for t in range(self.N_T):
            _, reward_policy, _, info_t = env.step(policy_rollout[:, t])
            time.sleep(env.timeStep)
            policy_final_reward += reward_policy
        policy_reward.append(policy_final_reward)
        print('Reward of current target policy:  ', policy_final_reward)
        print(info_t)

        ''' START ALGORITHM ITERATIONS '''
        n_iter = 0
        MEAN = predicted_mean_w
        for iter in range(self.N_ITERATIONS):

            ''' SAMPLE NEW WEIGHTS USING THE CURRENT EXPLORATION COVARIANCE MATRIX '''
            theta_sampled = np.zeros(shape = (self.N_JOINTS,self.N_BASIS))
            for j in range(self.N_JOINTS):
                theta_sampled[j,:] = np.random.multivariate_normal(MEAN[j,:], covariance_current[j, :, :], 1)
                epsilon[iter,j,:] = theta_sampled[j,:] - theta_current[j,:]

            ''' COMPUTE THE REWARD OF THE CURRENT SAMPLED POLICY '''
            initial_obs = env.reset_panda()

            ''' PRINT THE INITIAL OBSERVATION  '''
            # print("----- INITIAL OBSERVATION SAMPLED POLICY-----")
            # print(" Strawberry position: ", initial_obs[-3:])
            # print(" EE position: ", initial_obs[0:3])
            # print(" EE orientation: ", initial_obs[3:6])

            # Compute the rollout from the sampled weights
            current_rollout = np.zeros(shape = (self.N_JOINTS,self.N_T))
            for j in range(self.N_JOINTS):
                current_rollout[j,:] = np.squeeze(self.promp.trajectory_from_weights(theta_sampled[j,:],vector_output = False))
            # Compute the relative reward
            episode_reward = 0
            for t in range(self.N_T):
                _, reward, _, _ = env.step(current_rollout[:,t])
                time.sleep(env.timeStep)
                episode_reward += reward
                reward_t_all_episodes[iter,t] = reward
            # print('Reward of sampled policy after ', str(iter + 1), ' iterations:  ', episode_reward)


            '''
            IMPORTANCE SAMPLING: final_reward_all_episodes is ordered following the increasing 
            value of the rollouts final rewards. For the update of the parameters only the last
            best ten rollouts are considered
            '''
            for i in range(self.N_ITERATIONS):
                if final_reward_all_episodes[i,0] == 0:
                    final_reward_all_episodes[i,:] = [episode_reward,int(iter)]
                    break
            ind = np.argsort(final_reward_all_episodes[:,0])
            final_reward_all_episodes = final_reward_all_episodes[ind]
            print(final_reward_all_episodes[-40:, 0])
            ''' Compute Q function'''
            for t in range(self.N_T):
                # Compute the Value function as the um of rewards from a certain time step onward
                for tt in range(t, self.N_T):
                    Q_t_all_episodes[iter, t] += reward_t_all_episodes[iter, tt]
            n_iter += 1

            if n_iter >= 40:
                ''' Compute term 1 and term 2 terms used for updating the parameters'''
                term_1_best_rollout = np.zeros(shape = (self.N_JOINTS,self.N_BASIS,self.N_BASIS))
                term_1_covariance = np.zeros(shape = (self.N_JOINTS,self.N_BASIS,self.N_BASIS))
                term_2_best_rollout = np.zeros(shape = (self.N_JOINTS,self.N_BASIS))
                term_2_covariance = np.zeros(shape = (self.N_JOINTS,self.N_BASIS))
                for j in range(self.N_JOINTS):
                    for i in range(1,6):
                        # Get the rollout number for the 10 best rollouts
                        best_index = final_reward_all_episodes[-i, 1].astype(int)
                        for t in range(self.N_T):
                            term_1_best_rollout[j,:,:] += W[0,j, t, :, :] * Q_t_all_episodes[best_index, t]
                            term_2_best_rollout[j,:] += np.reshape(np.matmul(W[0,j, t, :, :],np.reshape(epsilon[best_index, j, :], newshape=(self.N_BASIS, 1)) * Q_t_all_episodes[best_index, t]) , newshape=(8,))
                            term_1_covariance[j,:,:] += np.matmul(np.reshape(epsilon[best_index, j, :], newshape=(self.N_BASIS, 1)),np.reshape(epsilon[best_index, j, :], newshape=(self.N_BASIS, 1)).T) * Q_t_all_episodes[best_index, t]
                            term_2_covariance[j,:] += Q_t_all_episodes[best_index, t]

                ''' Update parameters '''
                for j in range(self.N_JOINTS):
                    theta_target[iter+1,j,:] = theta_current[j,:] + np.matmul(np.linalg.inv(term_1_best_rollout[j,:,:]),term_2_best_rollout[j,:])

                ''' Update exploration covariance '''
                for j in range(self.N_JOINTS):
                    mean_1 = np.mean(np.diag(np.divide(term_1_covariance[j,:,:],term_2_covariance[j,:])))
                    mean_2 = np.mean(np.diag(predicted_cov_w[j,:,:]))
                    idx = np.argmin([mean_1,mean_2])
                    covariance_target[iter + 1, j, :, :] = np.divide(term_1_covariance[j,:,:],term_2_covariance[j,:])


                for j in range(self.N_JOINTS):
                         theta_current[j,:] = theta_target[iter + 1, j, :]
                         covariance_current[j, :,:] = covariance_target[iter + 1, j, :, :]

                ''' Update W '''
                for j in range(self.N_JOINTS):
                    for t in range(self.N_T):
                        # vector of basis functions activation at time t
                        PHI = np.reshape(self.phi[:, t], newshape=(self.N_BASIS, 1))
                        d = np.matmul(PHI.T, np.matmul(covariance_current[j, :, :], PHI))
                        W[iter+1, j, t, :, :] = np.matmul(PHI, PHI.T) / d


            else:
                ''' Update W '''
                for j in range(self.N_JOINTS):
                    for t in range(self.N_T):
                        # vector of basis functions activation at time t
                        PHI = np.reshape(self.phi[:, t], newshape=(self.N_BASIS, 1))
                        d = np.matmul(PHI.T, np.matmul(covariance_current[j, :, :], PHI))
                        W[iter+1, j, t, :, :] = np.matmul(PHI, PHI.T) / d


        return theta_current, covariance_current, np.asarray(policy_reward), info_t['Success']

















