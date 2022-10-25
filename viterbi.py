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

    # We can estimate birth in a specific way:
    # We can think that whenever we are at state 0 we switch birht poisson
    
    indices = np.where(samp_state_tape==0)
    birth_inter_arrivals = []
    for i in indices[0]:
        j = i+1
        time = args.samprate
        while (j < len(samp_state_tape)) and (samp_state_tape[j] == 0):
            time += args.samprate
            j+=1
        # We get interarrival length of birth
        birth_inter_arrivals.append(time)
    
    # Distribution of birth arrival times
    birth_inter_arrivals = np.asarray(birth_inter_arrivals)
    birth_rates = 1.0/birth_inter_arrivals
    print('Amount of visits to 0: ',len(indices[0]))
    print('Average of Interval Lambda:',np.mean(birth_inter_arrivals))
    print('Average of Lambda:',np.mean(1.0/birth_inter_arrivals))
    fig, axs = plt.subplots(1,2)
    axs[0].plot(samp_state_tape)
    axs[1].hist(birth_rates)
    x = np.linspace(0,np.max(birth_rates),20)
    axs[1].plot(x,args.lam*np.exp(-args.lam*x))
    plt.show()
        

    #  best_path = viterbi(emb_state_tape, (1,0),

    
