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
    num = 0
    denum = 0
    for i in range(len(tape)-1):
        from_state = tape[i]
        to_state = tape[i+1]
        #  num  *= p0[from_state,to_state]
        #  denum *= p1[from_state,to_state]
        num += np.log(p0[from_state,to_state])
        denum += np.log(p1[from_state,to_state])

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
    rates1 = {"lam": 0.4,"mu":1.2} 
    rates2 = {"lam": 0.4,"mu":1.4} 
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
    #  samp_rates = [args.samprate *2 ** j for j in range(10)]
    samp_rates = np.linspace(0.01,3,400)

    tgm0 = generate_true_gmatrix(rates[0], args.state_limit)
    tgm1 = generate_true_gmatrix(rates[1], args.state_limit)


    for cur_samp_rate in samp_rates:
        
        # Loop through multiple sampling rate
        true_values = np.random.choice(2,args.detection_guesses)
        print('Trying Sampling Rate: ',cur_samp_rate)
        guess = []

        true_p0 = get_true_trans_probs(Q=tgm0*(1/cur_samp_rate))
        true_p1 = get_true_trans_probs(Q=tgm1*(1/cur_samp_rate))
        
        
        # For every sample rate we will generate sample path and guess from it
        for i in range(args.detection_guesses):
            # Generate a path from either q0 or 1
            roe = RaceOfExponentials(args.length,rates[true_values[i]],state_limit=args.state_limit)
            holdTimes_tape, state_tape = roe.generate_history(args.init_state)
            sampled_tape = simple_sample(cur_samp_rate, state_tape, holdTimes_tape)
            guess.append(take_a_guess(sampled_tape, true_p0, true_p1))

        num_hits = (true_values == guess).sum()
        hit_rates.append(num_hits/args.detection_guesses)
        print("For Sampling Rate {} we have ratio of right guesses: {}/{}".format(cur_samp_rate,num_hits,args.detection_guesses))

    plt.plot(samp_rates,hit_rates)
    for i,rate in enumerate(rates):
        plt.axvline(rate['lam'],label='$\lambda_'+str(i)+'$',color='g')
        plt.axvline(rate['mu'],label='$\mu_'+str(i)+'$',color='g')
    plt.title('Number of right guesses vs sampling rate')
    plt.show()
