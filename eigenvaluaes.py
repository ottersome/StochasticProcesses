import numpy as np
import sys
import matplotlib.pyplot as plt
import argparse
from sp_sims.simulators.stochasticprocesses import *
from sp_sims.statistics.statistics import *
from sp_sims.estimators.algos import *
from sp_sims.sanitycheck.truebirthdeath import *
from sp_sims.utils.utils import *


def eigenvalues_and_sampling(state_tape, holdTimes_tape,args, doublings=6, initial_rate=1):

    fig,axs = plt.subplots(1,3)
    fig.tight_layout()
    fig.set_size_inches(10,10)

    dets = []
    samp_time = 1/args.samprate
    eigenvals = []
    for i in range(doublings+1):
        rate = initial_rate*2**i
        print("Working with rates ",rate)
        sampled_tape = simple_sample(rate, state_tape, holdTimes_tape)
        p_hat = state_transitions(np.full_like(sampled_tape, samp_time), sampled_tape)
        if rate == 4:
            axs[0].imshow(p_hat)
            for i in range(p_hat.shape[0]):
                for j in range(p_hat.shape[1]):
                    axs[0].text(j,i,"%.2f " % p_hat[i,j],ha="center",va="center",color="w",fontsize=8)
        dets.append(np.linalg.det(p_hat))
        eigenvals.append(np.linalg.eig(p_hat))
        if rate == 4:
            print("Eigenvalues for rate 2:")
            print(eigenvals[-1][0])

    axs[1].plot(range(len(dets)),dets)
    for i,eigval in enumerate(eigenvals):
        axs[2].scatter(np.repeat(i,len(eigval[0])),eigval[0])
    plt.show()


def probabilities(args, runs=500):

    fig,axs = plt.subplots(1,2)
    fig.tight_layout()
    fig.set_size_inches(10,10)

    samp_time = 1/args.samprate
    p_hat = np.zeros((args.state_limit+1,args.state_limit+1))
    prob_of_interest = []
    for i in range(1,runs):
        # Generate a Process
        roe = RaceOfExponentials(args.length,rates,state_limit=args.state_limit)
        holdTimes_tape, state_tape = roe.generate_history(args.init_state)

        # Sample the Process
        rate = args.samprate
        print("Iteration ",i)
        sampled_tape = simple_sample(rate, state_tape, holdTimes_tape)
        # Get the Probabilities
        new_p_hat= state_transitions(np.full_like(sampled_tape, samp_time), sampled_tape)
        prob_of_interest.append(new_p_hat[1,2])

        p_hat = (1/i)*(new_p_hat + p_hat*(i-1))
        # Store the one we are most interested in 

    axs[0].imshow(p_hat)
    for i in range(p_hat.shape[0]):
        for j in range(p_hat.shape[1]):
            axs[0].text(j,i,"%.2f " % p_hat[i,j],ha="center",va="center",color="w",fontsize=8)
    axs[1].hist(prob_of_interest,bins=20)
    plt.show()

if __name__ == '__main__':
    # Go through arguments
    args = argparser()

    # Created Tapes
    rates = {"lambda": args.lam,"mu":args.mu} 
    print(f"Working with parameters mu:{args.mu} lambda:{args.lam}")

    roe = RaceOfExponentials(args.length,rates,state_limit=args.state_limit)
    holdTimes_tape, state_tape = roe.generate_history(args.init_state)
    
    #  tbd = TrueBirthDeath(args.length,rates)
    #  holdTimes_tape, state_tape = tbd.generate_history(args.init_state)

    # roe = EmbeddedMarkC_BD(args.length,rates, state_limit=args.state_limit)
    # holdTimes_tape, state_tape = roe.generate_history(args.init_state)

    # For Sanity Check Purposes
    #  if args.show_cont_tmatx:  show_trans_matrx(holdTimes_tape, state_tape)


    print("Method being used : ",args.method)

    # After Sanity Check we want to Check on the EigenValues of the matrix
    # probabilities(args)
    eigenvalues_and_sampling(state_tape,holdTimes_tape,args)
