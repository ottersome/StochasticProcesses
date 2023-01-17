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
    print("Norm for given Q Matrix ", their_l1_norm(Q))
    return P

def their_l1_norm(M):
    return np.sum(np.sum(np.abs(M),1),0)


def print_mat_text(mat, axs):
    axs.imshow(mat)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            axs.text(j,i,"%.2f " % mat[i,j],ha="center",va="center",color="w",fontsize=8)

# This function will take a guess at which process generated the entire thing.
def take_a_guess(tape, p0, p1):
    num = 1
    denum = 1
    for i in range(len(tape)-1):
        from_state = tape[i]
        to_state = tape[i+1]
        num  *= p0[from_state,to_state]
        denum *= p1[from_state,to_state] 

    return 0 if num > denum else 1

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
    rates1 = {"lam": 1/16,"mu":1/12} 
    rates2 = {"lam": 1/12,"mu":1/10} 
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
    hit_rates = []
    #samp_rates = [args.samprate *2 ** j for j in range(10)]
    samp_rates = np.linspace(args.samprate*2**(-3),args.samprate*2**(3),100)
    tgm0 = generate_true_gmatrix(rates[0], args.state_limit)
    tgm1 = generate_true_gmatrix(rates[1], args.state_limit)

    # Let us first generate a single processes. We will probability repeat this experiment over and over

    # With this rates 
    roe0 = RaceOfExponentials(args.length,rates[0],state_limit=args.state_limit)
    roe1 = RaceOfExponentials(args.length,rates[1],state_limit=args.state_limit)
    holdTimes_tape0, state_tape0 = roe0.generate_history(args.init_state)
    holdTimes_tape1, state_tape1 = roe1.generate_history(args.init_state)
    true_values = np.random.choice(2,len(samp_rates))

    for i,cur_samp_rate in enumerate(samp_rates):
        
        # Loop through multiple sampling rate
        #  print('Trying Sampling Rate: ',cur_samp_rate)
        guess = []

        print("With rate ",cur_samp_rate)
        true_p0 = get_true_trans_probs(Q=tgm0*(1/cur_samp_rate))
        true_p1 = get_true_trans_probs(Q=tgm1*(1/cur_samp_rate))

        sampled_tape0 = simple_sample(args.samprate, state_tape0, holdTimes_tape0)
        sampled_tape1 = simple_sample(args.samprate, state_tape1, holdTimes_tape1)

        guessed_tape = sampled_tape0
        if true_values[i] == 1 : guessed_tape = sampled_tape1

        guess.append(take_a_guess(guessed_tape, true_p0,true_p1))

        num_hits = (true_values == guess).sum()
        hit_rates.append(num_hits)
        #  print("For Sampling Rate {} we have ratio of right guesses: {}/{}".format(cur_samp_rate,num_hits,args.detection_guesses))

    plt.plot(samp_rates,hit_rates)
    plt.title('Number of right guesses vs sampling rate')
    plt.show()
