import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import linalg
from scipy.linalg import block_diag
from numpy.linalg import multi_dot
from scipy.linalg import sqrtm
from scipy.sparse import csgraph
import os
import itertools

# Adjusted parameters based on problem sets
N_TRIAL = 30000 #N_TRIALS -> number of rounds
N_ARMS = 100 #N_ARMS -> number of clients
N_FEATURE = 10 #N_FEATURE -> number of features
M = 29 #M -> Number of selected clients
K = 1 #K -> Number of underlying groups

# Read files (X, Y, Beta) -> Can be adjusted based on problem sets
Y_1 = np.genfromtxt(r"/N100S2_Y_x10.csv",delimiter=',')
Beta = np.genfromtxt(r"/N100S2_Beta_x10.csv",delimiter=',')
X_1 = {}
for i in range(N_ARMS):
    name = '/DATA/SET2/X10/N100/'+ 'N100S2_'+'X10_' + str(i+1) + '.csv'
    readX = np.genfromtxt(name,delimiter=',')
    X_1[i] = readX.T


def make_regret(payoff, oracle):
    return np.cumsum(oracle - payoff)

def plot_regrets(results, oracle):
    [plt.plot(make_regret(payoff=x['r_payoff'], oracle=oracle), label="alpha: "+str(alpha)) for (alpha, x) in results.items()]

# X transformation to a sparse matrix
def X_reshape(X, X_tr, t, K, n_arms, n_feature):  #
  for arm in range(1, n_arms):
    X_tr = np.concatenate((X_tr,np.kron(np.identity(n = K),X[arm].reshape(-1,1))), axis = 1)
  return X_tr

# convert to a sparse matrix -> convert to a long sparse vector with flatten() -> Np x 1
def X_to_X_m(X, t, arm_choice, n_arms, n_feature): 
  X_m = np.copy(X[t])
  for arm in np.arange(n_arms): # N x p
    if arm not in arm_choice:
      X_m[arm] = np.zeros(shape=n_feature)
  return X_m

def upload(gammaU, IDclient, A_loc, A_up_buff): #, eta, n_clients, n_feature
    numerator = linalg.det(A_loc[IDclient])
    denominator = linalg.det(A_loc[IDclient] - A_up_buff[IDclient])
    if denominator == 0:
        return True
    else:
        check = numerator/denominator
        return check > gammaU

def Fed_LinUCB(N_TRIAL, N_ARMS, N_FEATURE,alpha_g, alpha_l, eta, X_g, X_l, Y, m, oracle, gammaU, gammaD, inner_iters):
    print(alpha_g, alpha_l, gammaU)
    # X = X_g # X = X_g = X_l
    # n_trial, n_clients, n_feature = X_l.shape
    # n_trial, n_clients, n_g_feature = X_g.shape
    n_trial = N_TRIAL
    n_clients = N_ARMS
    n_feature = N_FEATURE
    n_g_feature = N_FEATURE
    # 1.1.output object
    r_payoff = np.empty(n_trial)
    c_payoff = np.empty(n_trial)
    cum_regret = np.empty(n_trial)
    client_choice = np.empty(shape=(n_trial, m), dtype=int)
    p = np.empty(shape=(n_trial, n_clients))
    totalCommCost = 0
    cum_totalCommCost = np.empty(n_trial)
    
    # 1.2. local statistics est. (l) local_l
    A_loc = np.array([eta * np.identity(n=n_feature) for _ in np.arange(n_clients)]) #np.zeros((n_feature * n_clients, n_feature * n_clients))
    b_loc = np.array([np.zeros(shape=n_feature)  for _ in np.arange(n_clients)])
    theta_loc = np.empty(shape=(n_trial, n_clients, n_feature))
    
    # 1.3 local statistics (l-g) local
    A_loc_g = np.array([eta * np.identity(n=n_g_feature) for _ in np.arange(n_clients)]) #np.zeros((n_feature * n_clients, n_feature * n_clients))
    A_up_buff = np.array([np.zeros((n_g_feature, n_g_feature)) for _ in np.arange(n_clients)]) #np.zeros((n_feature * n_clients, n_feature * n_clients))
    b_loc_g = np.array([np.zeros(shape=n_g_feature)  for _ in np.arange(n_clients)])
    b_up_buff = np.array([np.zeros(shape=n_g_feature)  for _ in np.arange(n_clients)])
    theta_loc_g = np.empty(shape=(n_trial, n_clients, n_g_feature))
    
    # 1.4 Global statistics (g) Aggregated
    A_gob = eta * np.identity(n=n_g_feature) #np.zeros((n_feature * n_clients, n_feature * n_clients))
    A_down_buff = np.array([np.zeros((n_g_feature, n_g_feature)) for _ in np.arange(n_clients)])  #np.zeros((n_feature * n_clients, n_feature * n_clients))
    b_gob = np.zeros(shape=n_g_feature)
    b_down_buff = np.array([np.zeros(shape=n_g_feature)  for _ in np.arange(n_clients)])
    theta_gob = np.empty(shape=(n_trial, n_g_feature))
    
    # 2. Algorithm
    for t in np.arange(n_trial):
        # Compute estimates prediction (p) for all clients
        for a in np.arange(n_clients):
            inv_A_l = np.linalg.inv(A_loc[a])
            inv_A_l_g = np.linalg.inv(A_loc_g[a])
            theta_loc[t, a] = inv_A_l.dot(b_loc[a])
            theta_loc_g[t, a] = inv_A_l_g.dot(b_loc_g[a])
            # X_1_tr = (X_to_X_m(X, t, [a], n_clients, n_feature)).flatten()
            X_l_tr = X_l[a][t]
            X_g_tr = X_g[a][t]
            p[t, a] = theta_loc[t, a].dot(X_l_tr) + alpha_l * np.sqrt(np.dot(np.dot(X_l_tr, inv_A_l), X_l_tr)) + theta_loc_g[t, a].dot(X_g_tr) + alpha_g * np.sqrt(np.dot(np.dot(X_g_tr, inv_A_l_g), X_g_tr))
            
        # The central server chooses m best clients
        idx = np.argpartition(p[t], -m)[-m:]
        chosen_clients = idx[np.argsort(-(p[t])[idx])]
        for i in np.arange(m):
          client_choice[t][i] = chosen_clients[i]
        
        # Update local statistics based on following conditions
        for chosen_client in client_choice[t]:
            # client local update
            # X_1_tr_chosen = (X_to_X_m(X, t, [chosen_client], n_clients, n_feature)).flatten()
            X_l_tr_chosen = X_l[chosen_client][t]#X_l[t, chosen_client]
            X_g_tr_chosen = X_g[chosen_client][t]#X_g[t, chosen_client]
            y_l = Y[t, chosen_client] - np.dot(theta_loc[t, chosen_client], X_l_tr_chosen)
            y_g = Y[t, chosen_client] - np.dot(theta_loc_g[t, chosen_client], X_g_tr_chosen)
            for iter in range(inner_iters):
                theta_loc_g[t, chosen_client] = np.dot(np.linalg.pinv(A_loc_g[chosen_client] + np.outer(X_g_tr_chosen, X_g_tr_chosen)), b_loc_g[chosen_client] + y_g * X_g_tr_chosen)
                l2_norm = np.linalg.norm(theta_loc_g[t, chosen_client], ord=2)
                theta_loc_g[t, chosen_client] = theta_loc_g[t, chosen_client] / max(l2_norm, 1)
                y_l = Y[t, chosen_client] - np.dot(theta_loc_g[t, chosen_client], X_g_tr_chosen)
                
                theta_loc[t, chosen_client] = np.dot(np.linalg.pinv(A_loc[chosen_client] + np.outer(X_l_tr_chosen, X_l_tr_chosen)), b_loc[chosen_client] + y_l * X_l_tr_chosen)
                l2_norm = np.linalg.norm(theta_loc[t, chosen_client], ord=2)
                theta_loc[t, chosen_client] = theta_loc[t, chosen_client] / max(l2_norm, 1)
                y_g = Y[t, chosen_client] - np.dot(theta_loc[t, chosen_client], X_l_tr_chosen)
            
            A_loc_g[chosen_client] += np.outer(X_g_tr_chosen, X_g_tr_chosen)
            b_loc_g[chosen_client] += y_g * X_g_tr_chosen

            A_up_buff[chosen_client] += np.outer(X_g_tr_chosen, X_g_tr_chosen)
            b_up_buff[chosen_client] += y_g * X_g_tr_chosen

            A_loc[chosen_client] += np.outer(X_l_tr_chosen, X_l_tr_chosen)
            b_loc[chosen_client] += y_l * X_l_tr_chosen
            
            # check upload triggering event
            if upload(gammaU, chosen_client, A_loc_g, A_up_buff): #, eta, n_clients, n_feature
                totalCommCost += 1
                
                # update server's statistics
                A_gob += A_up_buff[chosen_client]
                b_gob += b_up_buff[chosen_client]
                
                # update server's download buffer for other clients
                for clientID in np.arange(n_clients):
                    if clientID != chosen_client:
                        A_down_buff[clientID] += A_up_buff[chosen_client]
                        b_down_buff[clientID] += b_up_buff[chosen_client]
                        
                # clear client's upload buffer
                A_up_buff[chosen_client] = np.zeros((n_g_feature, n_g_feature))
                b_up_buff[chosen_client] = np.zeros(shape=n_g_feature)

                # check download triggering event for all clients
                for cli in np.arange(n_clients):
                    # if download(gammaD, cli, A_gob, A_down_buff): #, eta, n_clients, n_feature
                    totalCommCost += 1
                        
                    # update client's local statistics, and clear server's download buffer
                    A_loc_g[cli] += A_down_buff[cli]
                    b_loc_g[cli] += b_down_buff[cli]
                        
                    # clear cserver's download buffer
                    A_down_buff[cli] = np.zeros((n_g_feature, n_g_feature))
                    b_down_buff[cli] = np.zeros(shape=n_g_feature)
            
            #else: if do not pass the upload, then the statistics are still the same in local
               
        #else: for other clients not selected at round t, the statistics are still the same in local      
        
        # Cumulative regret
        r_payoff[t] = np.sum([Y[t, choice] for choice in client_choice[t]])      
        cum_regret[t] = np.sum(oracle[0:t+1] - r_payoff[0:t+1])
        cum_totalCommCost[t] = totalCommCost
        print(t, cum_regret[t], cum_totalCommCost[t])
        # if (t+1) % 3000 == 0:
        #     print('TRIAL:',t,'DONE', '| cum_regret:', cum_regret[t])
        #     print('Total Communication cost:', totalCommCost)
        # # print(cum_regret[t], totalCommCost)
        # if cum_regret[t] > break_point:
        #     print('break at:', t, 'cum. regret:', cum_regret[t])
        #     break
    
    return dict(A_gob=A_gob, b_gob=b_gob, theta_loc=theta_loc, p=p, client_choice = client_choice, r_payoff=r_payoff, totalCommCost=totalCommCost, cum_totalCommCost=cum_totalCommCost)

# Generate oracle results (optimal)
oracle_lst = []
true_choice = []
new_y =  -1 * Y_1 + 30#-1 * Y_1 + 30
for t in np.arange(N_TRIAL):
  # Find indices of M highest arms
  all_reward_t = [new_y.T[t, arm] for arm in np.arange(N_ARMS)]
  chosen_arms = np.array(all_reward_t).argsort()[-M:][::-1]
  # Sum of M highest rewards
  oracle_payoff_t = np.sum([new_y.T[t, choice] for choice in chosen_arms])
  # Append to the list
  oracle_lst.append(oracle_payoff_t)
  true_choice.append(chosen_arms)
oracle_case1 = np.array(oracle_lst)

# Run experiments
alpha_to_test = [1.25] 
print('M:', M)
results_dict = {alpha: Fed_LinUCB(N_TRIAL=N_TRIAL, N_ARMS=N_ARMS, N_FEATURE=N_FEATURE, alpha_g = 1, alpha_l = alpha, eta = 0.3, X_g=X_1, X_l=X_1, Y=(-1 * Y_1 + 30).T, m=M, oracle=oracle_case1, gammaU=1.01, gammaD=1, inner_iters = 50)\
                for alpha in alpha_to_test}