import numpy as np
import sys
import matplotlib.pyplot as plt
import argparse
from sp_sims.simulators.stochasticprocesses import *
from sp_sims.statistics.statistics import *

def argparser(parser: argparse.ArgumentParser):
    parser.add_argument('--length',
                        dest='length',
                        default=1000,
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

def get_stationary(state_tape,holdTimes_tape,samp_rate ):
    #  trans_matrx_samp = state_transitions(np.full_like(state_tape,samprate,dtype=np.float16), state_tape)
    #
    #  initial = np.zeros((1,trans_matx_samp.shape[1]))
    #  initial[0,0] = 1
    #  res = np.copy(trans_matx_samp)
    #  frob_hor_ax = []
    #  frob_norms = []
    #  axs[0].plot(res,label="Power Method n={}".format(1))
    #
    #  for i in range(2,power_val):
    #      res = res@trans_matx_samp
    #      if np.log2(i) % 1 == 0:
    #          frob_norms.append(np.linalg.norm(trans_matx_samp-res,'fro'))
    #          fin = initial@res
    #          axs[0].plot(fin.flatten(),label="Power Method n={}".format(i))
    #
    #  axs[0].hist(state_tape,bins=trans_matx_samp.shape[1],density=True,label="Sampled Histogram")
    #
    #  # Plot the true distributoin
    #  maxx = trans_matx_samp.shape[1]+1
    #  x = np.linspace(1,maxx,100)
    #  # Theres a different expression for n = 0. Will add later
    #  meep = lambda expo: (args.lam/args.mu)**expo
    #  y = meep(x)
    #  y /= 1 + np.sum([meep(i-1) for i in range(1,1000)])
    #  y0 = 1/(1+ np.sum([meep(i-1) for i in range(1,1000)]))
    #  x = np.insert(x,0,0)
    #  y = np.insert(y,0,y0)
    #  axs[0].plot(x,y,label="Closed Form Solution")
    pass

def tri_digmatalgo(N,a,b,c,d,x):
    pass


def frob_comparison(state_tape,holdTimes_tape,samp_rate=1,power_val=64):

    fig,axs = plt.subplots(1,1)

    # 
    event_driven = state_transitions(holdTimes_tape, state_tape)
    event_diff_norms = []

    # Sweet jesus have mercy on the memory trans_matx_samp = []
    squared_nth_deg = []
    frob_norms = []
    for i in range(0,int(np.log2(power_val)+1)):
        pow2  = np.power(2,i)

        sampled_tape = simple_sample(samp_rate*pow2, state_tape, holdTimes_tape)

        trans_matx_samp.append(state_transitions(np.full_like(sampled_tape,1/(samp_rate*pow2),dtype=np.float16), sampled_tape))

        nth_mat = np.linalg.matrix_power(trans_matx_samp[-1],i+1)

        frob_norms.append(np.linalg.norm(trans_matx_samp[0]-nth_mat,ord='fro'))
        event_diff_norms.append(np.linalg.norm(event_driven-nth_mat,ord='fro'))

    # Frobeneius Norm
    xticks = [2**i for i in range(int(np.log2(power_val)+1))]
    axs.plot(xticks,frob_norms,label="Norms of diff from rate 1")
    axs.plot(xticks,event_diff_norms,label="Norms of diff from event driven",c='r')
    axs.set_xticks(xticks)
    
    axs.set_xscale('log')


    fig.tight_layout()
    fig.set_size_inches(10,10)
    plt.savefig('from_norms'.format(
        args.samprate,args.lam,args.mu,
        format='eps',dpi=200
        ))
    plt.legend()
    plt.show()

def power_matrix(holdTimes_tape, state_tape, powers=64):
    fig, ax = plt.subplots(1,1)
    fig.tight_layout()
    fig.set_size_inches(50,50)

    # We will be using event driven distributions for the moemnt being
    
    # Event Driven Empirical Probabilities
    trans_matx_event = state_transitions(holdTimes_tape, state_tape)
    for z in range(powers):
        ax.imshow(trans_matx_event)
        for i in range(trans_matx_event.shape[0]):
            for j in range(trans_matx_event.shape[1]):
                ax.text(j,i,"%2.2f " % trans_matx_event[i,j],ha="center",va="center",color="w")
        ax.set_title("Event Driven Transitions")
        plt.savefig('./Images/transition_matrices/trasmat_pow_'+str(z+1),dpi=300)
        trans_matx_event = trans_matx_event @ trans_matx_event
    print('Done with saving images')

def show_trans_matrix(holdTimes_tape, state_tape,samp_rate):

    fig, ax = plt.subplots(1,3)
    
    # Sample the Tapes
    sampled_tape_ori = simple_sample(samp_rate, state_tape, holdTimes_tape)
    sampled_tape_half = simple_sample(samp_rate/2, state_tape, holdTimes_tape)

    
    # Event Driven Empirical Probabilities
    trans_matx_event = state_transitions(holdTimes_tape, state_tape)
    im = ax[0].imshow(trans_matx_event)
    for i in range(trans_matx_event.shape[0]):
        for j in range(trans_matx_event.shape[1]):
            ax[0].text(j,i,"%2.2f " % trans_matx_event[i,j],ha="center",va="center",color="w")
    ax[0].set_title("Event Driven Transitions")

    # Sampled Transition Probabilities
    trans_matx_samp = state_transitions(np.full_like(sampled_tape_ori,args.samprate), sampled_tape_ori)
    im = ax[1].imshow(trans_matx_samp)
    for i in range(trans_matx_samp.shape[0]):
        for j in range(trans_matx_samp.shape[1]):
            ax[1].text(j,i,"%.2f " % trans_matx_samp[i,j],ha="center",va="center",color="w")
    ax[1].set_title("Transitions from sampled states at sr:{}".format(args.samprate))

    # Half the sampling rate
    trans_matx_samp_half = state_transitions(np.full_like(sampled_tape_half,args.samprate/2,dtype=np.float16), sampled_tape_half)
    im = ax[2].imshow(trans_matx_samp_half)
    for i in range(trans_matx_samp_half.shape[0]):
        for j in range(trans_matx_samp_half.shape[1]):
            ax[2].text(j,i,"%.2f " % trans_matx_samp_half[i,j],ha="center",va="center",color="w")
    ax[2].set_title("Transitions from sampled states at sr:{}".format(args.samprate/2))

    fig.tight_layout()
    fig.set_size_inches(50,50)
    plt.savefig('samprate_{}_lam_{}_mu_{}.jpg'.format(
        args.samprate,args.lam,args.mu,
        format='eps',dpi=600
        ))
    plt.show()


if __name__ == '__main__':
    # Create Markov Embedded Simulator
    parser  = argparse.ArgumentParser()
    argparser(parser)
    args = parser.parse_args()


    # Created Tapes
    rates = {"lambda": args.lam,"mu":args.mu} #This should keep us within the corner
    #  embedded_sp = EmbeddedMarkC_BD(args.length,rates)
    #  emb_hold_tape, emb_state_tape = embedded_sp.generate_history(args.init_state)
    roe = RaceOfExponentials(args.length,rates)
    holdTimes_tape, state_tape = roe.generate_history(args.init_state)

    # Sample it
    # Calculate Stationary Distribution
    
    #  frob_comparison(state_tape,holdTimes_tape,power_val=1024)
    
    #  show_trans_matrix(holdTimes_tape, state_tape,args.samprate)
    #frob_comparison(state_tape, holdTimes_tape)
    power_matrix(holdTimes_tape,state_tape,64)
    #  get_stationary()
   






