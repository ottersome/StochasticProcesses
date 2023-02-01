import numpy as np
import sys
import matplotlib.pyplot as plt
import argparse
from math import factorial
from scipy.linalg import expm
from sp_sims.simulators.stochasticprocesses import *
from sp_sims.statistics.statistics import *
from sp_sims.estimators.algos import *
from sp_sims.sanitycheck.truebirthdeath import *
from sp_sims.utils.utils import *
import random
from tqdm import tqdm
from time import sleep


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


def get_true_trans_probs(Q):
    #  P = power_series_exp(Q)
    P = expm(Q)
    # Get the Norm 
    #  print(np.linalg.norm(Q,ord='fro'))
    return P


def print_mat_text(mat, axs):
    axs.imshow(mat)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            axs.text(j,i,"%.2f " % mat[i,j],ha="center",va="center",color="w",fontsize=8)

# This function will take a guess at which process generated the entire thing.
def take_a_guess(tape, p0, p1):
    # num = 0
    # denum = 0
    num = 1
    denum = 1
    for i in range(len(tape)-1):
        from_state = tape[i]
        to_state = tape[i+1]
        num  *= p0[from_state,to_state]
        denum *= p1[from_state,to_state]
        # num += np.log(p0[from_state,to_state])
        # denum += np.log(p1[from_state,to_state])

    return 0 if num > denum else 1

def rgt():
    return (random.random(),random.random(),random.random())

def return_lls(tape, p0, p1):
    num = 0
    denum = 0
    for i in range(len(tape)-1):
        from_state = tape[i]
        to_state = tape[i+1]
        #  num  *= p0[from_state,to_state]
        #  denum *= p1[from_state,to_state]
        num += np.log(p0[from_state,to_state])
        denum += np.log(p1[from_state,to_state])

    return -num,-denum

def test_estimator(rates,args):
    # Get the Probabilities
    roe = RaceOfExponentials(args.length,rates,state_limit=args.state_limit)
    holdTimes_tape, state_tape = roe.generate_history(args.init_state)
    sampled_tape = simple_sample(args.samprate, state_tape, holdTimes_tape)
    p_hat = state_transitions(np.full_like(sampled_tape, 1/args.samprate), sampled_tape)
    
    # A Figure
    fig,axs = plt.subplots(1,2)
    fig.tight_layout()
    fig.set_size_inches(10,10)

    # Building True Statistics
    tgm = generate_true_gmatrix(rates, args.state_limit)
    true_p = get_true_trans_probs(Q=tgm*(1/args.samprate))


    # print_mat_text(p_hat, axs[0])
    axs[0].set_title('Estimated Probability Matrix')
    # print_mat_text(true_p, axs[1])
    axs[1].set_title('True Probability Matrix')

    plt.show()


if __name__ == '__main__':
    # Go through arguments
    args = argparser()

    # Created Tapes
    rates0 = {"lam": 4/10,"mu":12/10} 
    rates1 = {"lam": 8/10,"mu":14/10} 
    print("Null Rates ", rates0)
    print("Alternative Rates ", rates1)

    print("Holding rates are : {} {}".format(rates0['lam']+rates0['mu'],rates1['lam']+rates1['mu']))
    print("Holding time intervals are : {} {}".format(1/(rates0['lam']+rates0['mu']),1/(rates1['lam']+rates1['mu'])))

    
    # Sanity Check
    # test_estimator(rates1,args)
    
    # We dont even need that. We just neeed a Q matrix ||Q|| < ln 2

    # Create(by hand) Q matrix
    #rates = [{"lam":args.lam, "mu":args.mu},{"lam":1.4, "mu":3.9}]
    rates = [rates0,rates1]
    # We now have the analytical results we need

    # We will create multiple different samples here
    # samp_rates = [args.samprate *2 ** j for j in range(10)]
    samp_rates = np.logspace(-3,4,1000, base=2)
    # samp_rates = np.log(samp_rates)
    # samp_rates = 2.758 + (0.597)*np.log(samp_rates)

    tgm0 = np.array([[-rates0['lam'],rates0['lam']],[rates0['mu'],-rates0['mu']]])
    tgm1 = np.array([[-rates1['lam'],rates1['lam']],[rates1['mu'],-rates1['mu']]])

    # Chose a random rate to test:
    ran_index = np.random.randint(780,1000,1)
    print("Random Index is ", ran_index)

    curves = []
    hit_rates = []
    l0s,l1s = ([],[])

    j = 0

    true_values = np.random.choice(2,args.detection_guesses)
    hts, sts = ([],[])
    for i in range(args.detection_guesses):
        roe = RaceOfExponentials(args.length,rates[true_values[i]],state_limit=args.state_limit)
        holdTimes_tape, state_tape = roe.generate_history(args.init_state)
        hts.append(holdTimes_tape); sts.append(state_tape)
    # Cont Probabilities
    cont_dist = []
    fst = np.array(sts[0])
    fht = np.array(hts[0])
    for i in range(2):
        tot_times = np.sum(fht[fst==i])
        cont_dist.append(tot_times)
    print(fht[fst==0])
    cont_dist = np.array(cont_dist)/np.sum(np.array(cont_dist))
    cont_dist = {i:cd for i,cd in enumerate(cont_dist)}
    print("Cont Dist : ",cont_dist)


    for cur_samp_rate in tqdm(samp_rates):
        
        # Loop through multiple sampling rate
        # print('Trying Sampling Rate: ',cur_samp_rate)
        guess = []

        true_p0 = get_true_trans_probs(Q=tgm0*(1/cur_samp_rate))
        true_p1 = get_true_trans_probs(Q=tgm1*(1/cur_samp_rate))

        true_ps = [true_p0,true_p1]
        
        
        l0c,l1c = ([],[])
        # For every sample rate we will generate sample path and guess from it
        for i in range(args.detection_guesses):
            # Generate a path from either q0 or 1
            # TODO: Use decimation to make it faster
            sampled_tape = simple_sample(cur_samp_rate, sts[i],hts[i])
            guess.append(take_a_guess(sampled_tape, true_p0, true_p1))

            l0, l1 = return_lls(sampled_tape, true_p0, true_p1)

            l0c.append(l0)
            l1c.append(l1)

            if i == 4 and j==836:
                # Get Empirical Transition Matrix:
                empp = trans_matrix(sampled_tape)
                fig, axs = plt.subplots(1,2)
                axs[0].set_title('TheoreticalTape')
                print_mat_text(true_ps[true_values[i]],axs[0])
                print_mat_text(empp,axs[1])
                axs[1].set_title('Sampled Tape')
                print('cur_samp rate : ', cur_samp_rate)
                fig.suptitle("P for samprate {} and parameters m:{}, l:{}".format(cur_samp_rate,rates[true_values[i]]['mu'],rates[true_values[i]]['lam']))
                # Get the single probabilities
                u,c = np.unique(sampled_tape, return_counts=True)
                c = c/np.sum(c)
                print("Chain Distribution is : ",dict(zip(u,c)))
                # Continuous Probabilities
                print("Cont Distribution is : ",cont_dist)

                plt.show()

        j += 1
        l0s.append(np.mean(l0c))
        l1s.append(np.mean(l1c))

        num_hits = (true_values == guess).sum()
        hit_rates.append(num_hits/args.detection_guesses)
        # print("For Sampling Rate {} we have ratio of right guesses: {}/{}".format(cur_samp_rate,num_hits,args.detection_guesses))
        # print("Going through Sampling Rate {} ".format(cur_samp_rate))

    # curves_np = np.array(curves)
    # avgd = np.mean(curves_np,axis=0)
    smoothed_hits = savitzky_golay(hit_rates, 21, 3)

    fig, axs = plt.subplots(1,2)
    plt.rcParams['text.usetex'] = True
    #  plt.plot(samp_rates,hit_rates)
    # plt.plot(l0s,label='Null Likelihood')
    # plt.plot(l1s,label='Alternative Likelihood')
    # axs[0].plot(samp_rates,(-1)*np.log(hit_rates),label='Sampling Rates(log scale)')
    axs[0].plot(samp_rates,hit_rates,label='Sampling Rates(log scale)',color='gray',alpha=0.2,linewidth=1)
    axs[0].plot(samp_rates,smoothed_hits,label='Sampling Rates(log scale)',color='green')
    for i,rate in enumerate(rates):
        axs[0].axvline(rate['lam'],label='$\lambda_'+str(i)+'$',c=rgt())
        axs[0].axvline(rate['mu'],label='$\mu_'+str(i)+'$',c=rgt())
    axs[0].set_title('Number of right guesses vs sampling rate')
    axs[0].set_xscale("log",base=2)
    axs[0].legend()

    # Likelihoods
    axs[1].plot(samp_rates, l0s,label='L0', c='blue')
    axs[1].plot(samp_rates, l1s,label='L1',c='green')
    axs[1].set_title('Likelihoods with respect to sampling rate')
    plt.legend()
    plt.show()
