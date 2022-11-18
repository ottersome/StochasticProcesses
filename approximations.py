import numpy as np
import sys
import matplotlib.pyplot as plt
import argparse
from sp_sims.simulators.stochasticprocesses import *
from sp_sims.statistics.statistics import *
from sp_sims.estimators.algos import *


def argparser(parser: argparse.ArgumentParser):
    parser.add_argument('--length',
                        dest='length',
                        default=10000,
                        type=int,
                        help='Length of episode in discrete realizations.')
    parser.add_argument('--mu',
                        dest='mu',
                        default = .15,
                        type=float,
                        help='Service Rate')
    parser.add_argument('--lambda',
                        dest='lam',
                        default=.10,
                        type=float,
                        help='Birth Rate')
    parser.add_argument('--samprate',
                        dest='samprate',
                        default=1.0,
                        type=float,
                        help='Rate at which we sample real line.')
    parser.add_argument('--init_state',
                        dest='init_state',
                        type=int,
                        default = 0,
                        help='Initial State in the real line.(Amnt of current events)')



def sampled_approximation(state_tape, holdTimes_tape,args):
    fig,axs = plt.subplots(1,2)
    font_size = 8
    fig.tight_layout()
    samp_time = 1/args.samprate
    fig.set_size_inches(10,10)

    print(r"Calculating the $\Delta t = ${}-Sampeld Transition Matrix MLE...".format(args.samprate))
    sampled_tape = simple_sample(args.samprate, state_tape, holdTimes_tape)
    phat = state_transitions(np.full_like(sampled_tape, samp_time), sampled_tape)

    axs[0].imshow(phat)
    for i in range(phat.shape[0]):
        for j in range(phat.shape[1]):
            axs[0].text(j,i,"%2.2f " % phat[i,j],ha="center",va="center",color="w",fontsize=font_size)

    Qdt = power_series_log(phat, 10)
    Q = Qdt / samp_time

    axs[1].imshow(Q)
    for i in range(Q.shape[0]):
        for j in range(Q.shape[1]):
            axs[1].text(j,i,"%2.2f " % Q[i,j],ha="center",va="center",color="w",fontsize=font_size)
    axs[1].set_title(
            r"Estimated Generator Matrix through samples at $\Delta t = ${}. $\mu$:{} $\lambda$:{} ".format(args.samprate,args.mu,args.lam))
    plt.savefig('Images/MLE_sample_driven'.format(
        args.samprate,args.lam,args.mu, format='eps',dpi=200))
    plt.show()
    print("Calculated. Displaying...")


def show_est_gen_matrix(state_tape,holdTimes_tape,args):

    fig, axs = plt.subplots(1,1)
    font_size = 8
    fig.tight_layout()
    fig.set_size_inches(10,10)

    print("Calculating the Event Driven Matrix MLE...")
    gen_matrix = event_driven_mle(state_tape,holdTimes_tape)
    print("Calculated. Displaying...")

    axs.imshow(gen_matrix)
    for i in range(gen_matrix.shape[0]):
        for j in range(gen_matrix.shape[1]):
            axs.text(j,i,"%2.2f " % gen_matrix[i,j],ha="center",va="center",color="w",fontsize=font_size)
    axs.set_title(r"Estimated Generator Matrix through event driven transitions. $\mu$:{} $\lambda$:{} ".format(args.mu,args.lam))
    plt.savefig('Images/MLE_event_driven'.format(
        args.samprate,args.lam,args.mu, format='eps',dpi=200))
    plt.show()


if __name__ == '__main__':
    # Create Markov Embedded Simulator
    parser  = argparse.ArgumentParser()
    argparser(parser)
    args = parser.parse_args()

    # Created Tapes
    rates = {"lambda": args.lam,"mu":args.mu} #This should keep us within the corner
    print(f"Working with parameters mu:{args.mu} lambda:{args.lam}")
    roe = RaceOfExponentials(args.length,rates)
    holdTimes_tape, state_tape = roe.generate_history(args.init_state)

    # Sample it
    # Calculate Stationary Distribution
    #  sampled_tape = simple_sample(args.samprate, state_tape, holdTimes_tapes)
    #  trans_matx_samp=state_transitions(
    #      np.full_like(sampled_tape,samp_rate,dtype=np.float16), sampled_tape)
     
    #  sample
    show_est_gen_matrix(state_tape,holdTimes_tape,args)
    #  sampled_approximation(state_tape, holdTimes_tape, args)

