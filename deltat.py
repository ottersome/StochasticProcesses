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
    sampled_tape_ori = simple_sample(args.samprate, state_tape, holdTimes_tape)
    sampled_tape_half = simple_sample(args.samprate/2, state_tape, holdTimes_tape)

    fig, ax = plt.subplots(1,3)
    
    #- Analyze the tapes

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
    fig.set_size_inches(10,10)
    plt.savefig('samprate_{}_lam_{}_mu_{}.jpg'.format(
        args.samprate,args.lam,args.mu
        ))
    plt.show()



