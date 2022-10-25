import numpy as np

class MAP: 
    pass

def viterbi(obs_tape,inital_hs_probs, trans_probs,emission_probs):
    
    # Two Hidden States 
    tape_length = len(tape_path)
    obs = obs_tape

    # Emission Probs must be KXN where K is num of hidden states and N is num of emitted states
    num_hidden_states = emission_probs.shape[0]
    
    T1 = np.ndarray(num_hidden_states,tape_length,dtype=np.float16)
    T2 = np.ndarray(num_hidden_states,tape_length,dtype=np.float16)

    # Fill them
    for i in range(num_hidden_states):
        T1[i,0] = inital_hs_probs[i]*emission_probs[i,obs[i]]
        T2[i,0] = 0

    # Now onto the sequence
    for j in range(1,tape_length):
        for i in range(num_hidden_states):
            T1[i,j] = max(T1[:,j-1]*trans_probs[:,i]*emission_probs[i,obs[j]])
            T2[i,j] = np.argmax(T1[:,j-1]*trans_probs[:,i]*emission_probs[i,obs[j]])
    
    best_last_hs = []
    Zt = np.argmax(T1[:,-1])
    for o in range(len(statess)-1,-1,-1):
        best_path.insert(0,best_last_hs)
        best_last_hs = T2[best_last_hs,o]

    return best_path








        





    
