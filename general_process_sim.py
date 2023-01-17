import numpy as np
import sys
import matplotlib.pyplot as plt
import argparse
import scipy
from scipy import *
from scipy.linalg import expm
from sp_sims.simulators.stochasticprocesses import *
from sp_sims.statistics.statistics import *
from sp_sims.estimators.algos import *
from sp_sims.sanitycheck.truebirthdeath import *
from sp_sims.utils.utils import *


def get_true_trans_probs(Q):
    #P = power_series_exp(delta_t*Q)
    P = expm(Q)
    # Get the Norm 
    print("Norm for given Q Matrix ", their_l1_norm(Q))
    return P

def their_l1_norm(M):
    return np.sum(np.sum(np.abs(M),1),0)


def print_mat_text(mat, axs):
    axs.imshow(mat)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            axs.text(j,i,"%.2f " % mat[i,j],ha="center",va="center",color="w",fontsize=8)

def test_for_convergence(q_matrix, sampling_rates):
    """
    Plots the difference between theoretical matrix and sampled 
    ones as a function of sampling rate
    Arguments:
        q_matrix: original generator matrix
        sampling_rates: array of rates from which we want to sample from
    Returns:
        Nothing, Just Plots The Norm of their difference
    """
    gemc = GeneralEmbeddedMarkC(args.length, preload_gmat)
    emb_prob_matrix = gemc.get_prob_mat()
    holding_times, state_tape = gemc.generate_history(0)
    localrates_only = q_matrix - np.multiply(np.diag(q_matrix,0),np.eye(q_matrix.shape[0]))
    hold_rates=np.sum(localrates_only, axis=1)

    print("Local rates are ", hold_rates)

    # Lets generate ourselves an episode
    norms = []
    for sr in sampling_rates:
        theo_P = scipy.linalg.expm((1/sr)*preload_gmat)
        samp_states =  simple_sample(sr, state_tape, holding_times)
        emp_P = trans_matrix(samp_states)
        norms.append(np.linalg.norm(theo_P-emp_P, ord='fro'))

    fig,axs = plt.subplots(1,1)
    fig.tight_layout()
    fig.set_size_inches(10,10)
    axs.plot(sampling_rates, norms)
    for rate in hold_rates:
        axs.axvline(x=rate, color='g', label='rate state='+str(rate))
    axs.axvline(x=np.mean(hold_rates),color='r', label='Hold Rates Mean')
    axs.set_title("Sampling Rate vs $||P-\hat{P}||_{fro}$")
    plt.legend()
    plt.show()
    return 

def get_ks(p_mat):
    eigvals = np.linalg.eigvals(p_mat)
    detp = np.linalg.det(p_mat)
    ks = []
    print('ln(det(P))=',np.abs(np.log(detp)))
    for e,val in enumerate(eigvals):
        print("Computing k for {}-st eigval: {}".format(e+1,val))
        curk = 0
        lks = []
        print("Cur lineq:",str(np.abs(np.angle(val)+2*np.pi*curk)))
        while np.abs(np.angle(val)+2*np.pi*curk) < np.abs(np.log(detp)):
            lks.append(curk)
            curk += 1
        ks.append(lks)
    assert len(ks)==len(eigvals)
    f =  lambda eival,k: np.log(np.abs(eival)) + 1j*np.angle(eival) + 2*np.pi*float(k)
    print("Ks are :",ks)
    
    # TODO: make a better function than this ugly thing
    def g(x):
        sumo = 0
        for j,valj in enumerate(eigvals):
            multo = 1
            for valk in eigvals: 
                if valk!=valj:
                    multo *= ((x-valk)/(valj-valk))
                    #  multo *= ((x-valk)/(valj-valk))*f(valj,ks[j])
            sumo+=multo*f(valj,0)
        return sumo
    # Then use G to compute the new matrix
    vec_g = np.vectorize(g)
    # Find Valid Generators 
    G = vec_g(p_mat)
    if np.sum(np.imag(G)) > 0.1:
        print("G has complex values!")
    G = np.real(G)
    return G
     


if __name__ == '__main__':
    # Go through arguments
    args = argparser()
    # Load the Pregenerated Generator Matrix
    preload_gmat = np.load(args.preload_gmat_loc)
    
    # Simulate the Process
    gemc = GeneralEmbeddedMarkC(args.length, preload_gmat)
    emb_prob_matrix = gemc.get_prob_mat()
    holding_times, state_tape = gemc.generate_history(0)
    
    # Lets set a useful sampling rate(twice the average holding rate should be good
    sampling_rate = np.mean(gemc.holding_rates)*2
    print("Mean Holding Rate:",np.mean(gemc.holding_rates))
    print("Sampling Rate:", sampling_rate) 
    
    # Create the Theoretical Probability Transition Matrices
    theo_P = scipy.linalg.expm((1/sampling_rate)*preload_gmat)
    print('Eigenvalues of theoretical transition matrix: ', [ "{:.2f}".format(val) for val in np.linalg.eigvals(theo_P)])

    # Creative Zone. 
    #   You may place functions below for testing ideas before going into the main function of the file
    #  test_for_convergence(preload_gmat, np.linspace(.1,sampling_rate*2.5,1000))

    # Some Sanity Check
    if args.show_sanity_check:
        fig,axs = plt.subplots(2,2)
        fig.tight_layout()
        fig.set_size_inches(10,10)
        print_mat_text(preload_gmat, axs[0,0]); axs[0,0].set_title('Preloaded Q-mat')
        print_mat_text(theo_P,axs[0,1]); axs[0,1].set_title('Corresponding T-mat')
        print_mat_text(emb_prob_matrix, axs[1,0]); axs[1,0].set_title('Embedded Chain')
        axs[1,1].scatter(range(len(holding_times)), state_tape,s=40*(holding_times/np.max(holding_times)))
        axs[1,1].plot(range(len(holding_times)), state_tape)
        axs[1,1].set_title('Path')
        plt.show()
        
        # Create Empirical Path
        sampstate_path = simple_sample(sampling_rate, state_tape, holding_times)
        emp_trans_mtx = trans_matrix(sampstate_path)
        fig,axs = plt.subplots(1,2)
        print_mat_text(theo_P, axs[0])
        axs[0].set_title('Theoretical Transition Matrix')
        print_mat_text(emp_trans_mtx, axs[1])
        axs[1].set_title('Sampled Transition Matrix')
        plt.suptitle('Transition Matrices at samprate '+str(sampling_rate))
        plt.show()

    
    ####################
    ## Recover Possible Empirical Generator Matrices
    ####################
    # Compute Our Transition Matrices
    this_samp_rate = 2
    print("Samp_rate for reconstruction:",this_samp_rate)
    sampled_tapes = simple_sample(this_samp_rate,state_tape,holding_times)
    emp_tmat = trans_matrix(sampled_tapes)
    # Compute Eigenvalues of Matrix
    eigvals = np.linalg.eigvals(emp_tmat)
    # Verify they are different
    unique_amnt = np.unique(eigvals)
    if len(unique_amnt) != len(eigvals):
        print('Eigenvalues of transition matrix are not distinct. Exiting...')
        exit(-1)
    # Get the Different Ks that satisfy this
    G = get_ks(emp_tmat)
    print(G)
    fig,axs = plt.subplots(1,2)
    fig.tight_layout()
    fig.set_size_inches(10,10)
    print_mat_text(G, axs[0]); axs[0].set_title('Generator G')
    print_mat_text(expm(G), axs[1]); axs[1].set_title('Corresponding P-Mat')
    plt.show()
    print("Done")
    



