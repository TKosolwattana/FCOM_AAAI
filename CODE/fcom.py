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
N_FEATURE = 6 #N_FEATURE -> number of features
M = 33 #M -> Number of selected clients
K = 3 #K -> Number of underlying groups

# Read files (X, Y, Beta) -> Can be adjusted based on problem sets
Y_2 = np.genfromtxt(r"/DATA/Y2_set1.csv",delimiter=',', skip_header=1)
# Create X_i = [1, t, t^2]
X_2_lst = []
for T in np.arange(N_TRIAL):
  X_2t_lst = []
  for arm in np.arange(N_ARMS):
    temp = []
    temp.append(1)
    temp.append(0.0001*(T+1))
    temp.append((0.0001*(T+1))**2)
    temp.append((0.0001*(T+1))**3)
    temp.append((0.0001*(T+1))**4)
    temp.append((0.0001*(T+1))**5)
    X_2t_lst.append(np.array(temp))
  X_2_lst.append(np.array(X_2t_lst))
X_2 = np.array(X_2_lst)


# Generate oracle results (optimal)
oracle_lst = []
true_choice = []
new_y = Y_2 #
for t in np.arange(N_TRIAL):
  # Find indices of M highest arms
  all_reward_t = [new_y.T[t, arm] for arm in np.arange(N_ARMS)]
  chosen_arms = np.array(all_reward_t).argsort()[-M:][::-1]
#   idx = np.argpartition(np.array(all_reward_t), -M)[-M:]
#   chosen_arms = idx[np.argsort(-(np.array(all_reward_t))[idx])]
  # Sum of M highest rewards
  oracle_payoff_t = np.sum([new_y.T[t, choice] for choice in chosen_arms])
  # Append to the list
  oracle_lst.append(oracle_payoff_t)
  true_choice.append(chosen_arms)
  # if (t+1) % 1000 == 0:
  #   print('TRIAL:',t,'DONE', '| arm selected:', chosen_arms)
oracle_case1 = np.array(oracle_lst)

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

def upload(gammaU, IDclient, A_loc, A_up_buff):    
    return (linalg.det(A_loc[IDclient])) > gammaU*linalg.det(A_loc[IDclient] - A_up_buff[IDclient])

def Fed_CLUCB(N_TRIAL, N_ARMS, N_FEATURE, break_point, eta_1, eta_2, alpha_q, alpha_c, X, Y, init_q, init_c, m, K, X_to_X_m, X_reshape, oracle, gammaU, iterations):
    print('---------------------------------------------------------')
    print('gammaU:', gammaU, 'alpha_q:', alpha_q, 'alpha_c:', alpha_c)
    # n_trial, n_clients, n_feature = X.shape
    n_trial = N_TRIAL
    n_clients = N_ARMS
    n_feature = N_FEATURE
    # 1.1. Output objects
    t_break = 0
    final_c_regret = 100000
    totalCommCost = 0
    client_choice = np.empty(shape=(n_trial, m), dtype=int)
    r_payoff = np.empty(n_trial)   
    c_payoff = np.empty(n_trial) 
    cum_regret = np.empty(n_trial)
    p = np.empty(shape=(n_trial, n_clients))
    cum_totalCommCost = np.empty(n_trial)
    
    # 1.2. Intialize local statistics
    A_loc = np.array([eta_1 * np.identity(n=K * n_feature) for _ in np.arange(n_clients)])
    A_up_buff = np.array([np.zeros((K * n_feature, K * n_feature)) for _ in np.arange(n_clients)])
    b_loc = np.array([np.zeros(shape=K * n_feature)  for _ in np.arange(n_clients)])
    b_up_buff = np.array([np.zeros(shape=K * n_feature)  for _ in np.arange(n_clients)])
    q_loc = np.empty(shape = (n_trial, n_clients, K * n_feature)) #Kp x 1
    A_down_buff = np.array([np.zeros((K * n_feature, K * n_feature)) for _ in np.arange(n_clients)])
    b_down_buff = np.array([np.zeros(shape=K * n_feature)  for _ in np.arange(n_clients)])
    
    D_loc = np.array([eta_2 * np.identity(n= K) for _ in np.arange(n_clients)])
    d_loc = np.array([np.zeros(shape= K)  for _ in np.arange(n_clients)])
    c_loc = np.empty(shape = (n_trial, n_clients, K)) #K x 1 (n clients)
    
    # temp parameters
    te_q_loc = np.empty(shape = (n_trial, n_clients, K * n_feature)) #Kp x 1
    te_c_loc = np.empty(shape = (n_trial, n_clients, K)) #K x 1 (n clients)
    
    #add initialization for each client
    for b in np.arange(n_clients): 
        q_loc[0, b] = init_q
        c_loc[0, b] = init_c[b]
        te_q_loc[0, b] = init_q
        te_c_loc[0, b] = init_c[b]
        
    # 1.3 Global statistics
    A_gob = eta_1 * np.identity(n=K * n_feature)  
    b_gob = np.zeros(shape=K * n_feature)     
    q_gob = np.zeros(shape=K * n_feature)
    
    # 2. Algorithm
    for t in np.arange(n_trial):
        for a in np.arange(n_clients):
            #Calculate inv(A_loc[a]), inv(D_loc[a]), q_loc[t,a], c_loc[t,a]
            inv_A = np.linalg.inv(A_loc[a])
            inv_D = np.linalg.inv(D_loc[a])
            if t != 0:
                q_loc[t, a] = inv_A.dot(b_loc[a])
                c_loc[t, a] = inv_D.dot(d_loc[a])
                te_q_loc[t, a] = q_loc[t, a]
                te_c_loc[t, a] = c_loc[t, a]
        
            #X Transformation 
            X_tr = np.kron(np.eye(K), X[t, a].T)
            
            #Calculate cb_q and cb_c
            #cb_q  
            X_q_a = c_loc[t, a].dot(X_tr)
            cb_q = alpha_q * np.sqrt(X_q_a.dot(inv_A).dot(X_q_a.T))
            
            #cb_c
            X_c = X_tr.dot(q_loc[t, a])
            cb_c = alpha_c * np.sqrt((X_c).T.dot(inv_D).dot(X_c))
            
            #Predictions
            p[t, a] = c_loc[t, a].dot(X_tr).dot(q_loc[t, a]) + cb_q + cb_c #FInv.dot
            
        chosen_clients = p[t].argsort()[-m:][::-1]
        for i in np.arange(m):
            client_choice[t][i] = chosen_clients[i]
        
        
        # each client solve for q and c iteratively and locally using ALS
        
        for chosen_client in client_choice[t]:
            for j in np.arange(iterations):
                X_tr_chosen = np.kron(np.eye(K), X[t, chosen_client].T)
                X_q = (te_c_loc[t, chosen_client].dot(X_tr_chosen)).T
                X_C_Tilde = X_tr.dot(te_q_loc[t, chosen_client])
                
                # client local buffers update
                A_up_buff[chosen_client] = A_up_buff[chosen_client] + np.outer(X_q, X_q) 
                b_up_buff[chosen_client] = b_up_buff[chosen_client] + Y[t, chosen_client] * X_q
                
                A_loc[chosen_client] = A_loc[chosen_client] + np.outer(X_q, X_q)
                b_loc[chosen_client] = b_loc[chosen_client] + Y[t, chosen_client] * X_q
                D_loc[chosen_client] = D_loc[chosen_client] + np.outer(X_C_Tilde, X_C_Tilde)           
                d_loc[chosen_client] = d_loc[chosen_client] + Y[t, chosen_client] * X_C_Tilde
                
                te_q_loc[t, chosen_client] = np.linalg.inv(A_loc[chosen_client]).dot(b_loc[chosen_client])
                te_c_loc[t, chosen_client] = np.linalg.inv(D_loc[chosen_client]).dot(d_loc[chosen_client])
                
            
        #each client check upload conditions whether to upload to the server
        # for chosen_client in client_choice[t]:
            c_loc[t, chosen_client] = te_c_loc[t, chosen_client]
            
            if upload(gammaU, chosen_client, A_loc, A_up_buff):
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
                A_up_buff[chosen_client] = np.zeros((K * n_feature, K * n_feature))
                b_up_buff[chosen_client] = np.zeros(shape=K * n_feature)
                
                q_gob = np.linalg.inv(A_gob).dot(b_gob)          
        
                # Send all statistics back to all clients
                for cli in np.arange(n_clients):
                    totalCommCost += 1 
                    # A_loc[cli] = A_gob
                    # b_loc[cli] = b_gob
                    A_loc[cli] += A_down_buff[cli]
                    b_loc[cli] += b_down_buff[cli]
                    
                    # clear cserver's download buffer
                    A_down_buff[cli] = np.zeros((K * n_feature, K * n_feature))
                    b_down_buff[cli] = np.zeros(shape=K * n_feature)
                    
            #q_loc[clientID] = q_gob
                
        # Cumulative regret
        r_payoff[t] = np.sum([Y[t, choice] for choice in client_choice[t]])      
        cum_regret[t] = np.sum(oracle[0:t+1] - r_payoff[0:t+1])
        cum_totalCommCost[t] = totalCommCost
        if (t+1) % 5000 == 0:
            print('TRIAL:',t,'DONE', '| cum_regret:', cum_regret[t])
            print('Total Communication cost:', totalCommCost)
        # print(cum_regret[t], totalCommCost)
        if cum_regret[t] > break_point:
            print('break at:', t, 'cum. regret:', cum_regret[t])
            break
        if t+1 == 30000:
            t_break = n_trial
            final_c_regret = cum_regret[t]
        
    return dict(A_gob=A_gob, b_gob=b_gob, q_loc=q_loc, c_loc = c_loc, p = p, client_choice = client_choice, r_payoff = r_payoff, totalCommCost=totalCommCost, cum_totalCommCost=cum_totalCommCost,t_break=t_break, final_c_regret=final_c_regret)

# Initialization
np.random.seed(4569) #3 #59
vec_q = np.array([np.random.rand() for _ in range(K * N_FEATURE)])
# vec_q = q[~np.isnan(q)]
# vec_C: C (NK x 1)
np.random.seed(4569)
longvec_C = np.array([np.random.rand() for _ in range(N_ARMS * K)])
matrix_c = longvec_C.reshape(N_ARMS, K)
vec_C = matrix_c

# Run experiments
alpha_q = [0.25 + i*0.25 for i in range(0,8)]
alpha_c = [0.25 + i*0.25 for i in range(0,8)]

results_dict = {}
rec_a_q = 4569
rec_a_c = 4569
min_target_cregret = 1000
for (al_q, al_c) in itertools.product(alpha_q, alpha_c):
    results_dict[(al_q, al_c)] = Fed_CLUCB(N_TRIAL=N_TRIAL, N_ARMS=N_ARMS, N_FEATURE=N_FEATURE, break_point = min_target_cregret, eta_1 = 0.3, eta_2 = 0.3, alpha_q =al_q, alpha_c = al_c, X=X_2, Y=Y_2.T, init_q=vec_q, init_c=vec_C,m=M, K = K, X_to_X_m=X_to_X_m, X_reshape=X_reshape, oracle=oracle_case1, gammaU=1.05, iterations=5)

