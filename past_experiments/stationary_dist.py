import sys
import matplotlib.pyplot as plt
import argparse
from sp_sims.simulators.stochasticprocesses import *
from sp_sims.statistics.statistics import *
from sp_sims.estimators.estimators import viterbi

# NOT FINISHED
# Merely somethign I whipped up in a few minutes

def argparser(parser: argparse.ArgumentParser):
    parser.add_argument('--length',
                        dest='length',
                        default=1000,
                        type=int,
                        help='Length of episode in discrete realizations.')
    parser.add_argument('--mu',
                        dest='mu',
                        default = 1.5,
                        type=float,
                        help='Service Rate')
    parser.add_argument('--lambda',
                        dest='lam',
                        default=1.0,
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

    # Estimating Birth/Death  
    parser = argparse.ArgumentParser(description='Estimating Birth Intensity')
    argparser(parser)

    args = parser.parse_args()
    print(args)

    # Get Our Tape of Birth Death
    rates = {"lambda": args.lam,"mu":args.mu} #This should keep us within the corner
    embedded_sp = EmbeddedMarkC_BD(args.length,rates)
    emb_hold_tape, emb_state_tape = embedded_sp.generate_history(args.init_state)

    
    # Sample the tapes
    samp_state_tape = simple_sample(args.samprate, emb_state_tape, emb_hold_tape)

    # Plot the Empirical Distribution
    plt.hist(samp_state_tape,density=True,bins=20)

    #Plot the true distributoin
    maxx = np.max(samp_state_tape)
    x = np.linspace(1,maxx,100)
    # Theres a different expression for n = 0. Will add later
    meep = lambda expo: (args.lam/args.mu)**expo
    y = meep(x)
    y /= 1 + np.sum([meep(i-1) for i in range(1,1000)])
    y0 = 1/(1+ np.sum([meep(i-1) for i in range(1,1000)]))
    x = np.insert(x,0,0)
    y = np.insert(y,0,y0)
    plt.plot(x,y)
    plt.show()
        

    #  best_path = viterbi(emb_state_tape, (1,0),

    
