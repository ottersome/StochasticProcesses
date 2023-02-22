import numpy as np
import sys
import matplotlib.pyplot as plt
import argparse
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
    print(np.linalg.norm(Q,ord='fro'))
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


    print_mat_text(p_hat, axs[0])
    axs[0].set_title('Estimated Probability Matrix')
    print_mat_text(true_p, axs[1])
    axs[1].set_title('True Probability Matrix')

    plt.show()


if __name__ == '__main__':
    # Go through arguments
    args = argparser()

    # Created Tapes
    rates1 = {"lam": 4/10,"mu":12/10} 
    rates2 = {"lam": 8/10,"mu":14/10} 
    print("Null Rates ", rates1)
    print("Alternative Rates ", rates2)

    print("Holding rates are : {} {}".format(rates1['lam']+rates1['mu'],rates2['lam']+rates2['mu']))
    print("Holding time intervals are : {} {}".format(1/(rates1['lam']+rates1['mu']),1/(rates2['lam']+rates2['mu'])))

    
    # Sanity Check
    # test_estimator(rates1,args)
    
    # We dont even need that. We just neeed a Q matrix ||Q|| < ln 2

    # Create(by hand) Q matrix
    #rates = [{"lam":args.lam, "mu":args.mu},{"lam":1.4, "mu":3.9}]
    rates = [rates1,rates2]
    # We now have the analytical results we need

    # We will create multiple different samples here
    # samp_rates = [args.samprate *2 ** j for j in range(10)]
    samp_rates = np.logspace(-3,4,1000, base=2)
    # samp_rates = np.log(samp_rates)
    # samp_rates = 2.758 + (0.597)*np.log(samp_rates)

    tgm0 = generate_true_gmatrix(rates[0], args.state_limit)
    tgm1 = generate_true_gmatrix(rates[1], args.state_limit)

    curves = []
    for avs in range(20):
        hit_rates = []
        l0s,l1s = ([],[])

        for cur_samp_rate in samp_rates:
            
            # Loop through multiple sampling rate
            true_values = np.random.choice(2,args.detection_guesses)
            print('Trying Sampling Rate: ',cur_samp_rate)
            guess = []

            true_p0 = get_true_trans_probs(Q=tgm0*(1/cur_samp_rate))
            true_p1 = get_true_trans_probs(Q=tgm1*(1/cur_samp_rate))
            
            # For every sample rate we will generate sample path and guess from it
            roe = RaceOfExponentials(args.length,rates[0],state_limit=args.state_limit)
            holdTimes_tape, state_tape = roe.generate_history(args.init_state)
            sampled_tape = simple_sample(cur_samp_rate, state_tape, holdTimes_tape)
            # Get Stationary Distribution
            unique,counts = np.unique(sampled_tape,return_counts=True)
            counts = counts / np.sum(counts)
            stat_dist = dict(zip(unique,counts))

            # Get Power Spectral Density
            psd(stat_dist)
             




            num_hits = (true_values == guess).sum()
            hit_rates.append(num_hits/args.detection_guesses)
            print("For Sampling Rate {} we have ratio of right guesses: {}/{}".format(cur_samp_rate,num_hits,args.detection_guesses))
            # print("Going through Sampling Rate {} ".format(cur_samp_rate))
        curves.append(hit_rates)
    np.save('curves.npy', curves)

    curves_np = np.array(curves)
    avgd = np.mean(curves_np,axis=0)

    #  plt.plot(samp_rates,hit_rates)
    # plt.plot(l0s,label='Null Likelihood')
    # plt.plot(l1s,label='Alternative Likelihood')
    plt.plot(samp_rates,(-1)*np.log(avgd),label='Sampling Rates(log scale)')
    for i,rate in enumerate(rates):
        plt.axvline(rate['lam'],label='$\lambda_'+str(i)+'$')
        plt.axvline(rate['mu'],label='$\mu_'+str(i)+'$')
    plt.title('Number of right guesses vs sampling rate')
    plt.xscale("log",base=2)
    plt.legend()
    plt.show()
