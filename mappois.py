import sys
import matplotlib.pyplot as plt
import argparse
from sp_sims.simulators.stochasticprocesses import *
from sp_sims.statistics.statistics import *

def argparser(parser: argparse.ArgumentParser):
    parser.add_argument('length',
                        metavar='len',
                        type=int,
                        help='Length of episode in discrete realizations.')
    parser.add_argument('poi_rate',
                        metavar='prate',
                        type=float,
                        help='Rate at which we sample real line.')
    parser.add_argument('sampling_rate',
                        metavar='srate',
                        type=float,
                        help='Rate at which we sample real line.')
    parser.add_argument('no_samples',
                        metavar='nsamples',
                        type=int,
                        help='Rate at which we sample real line.')
    parser.add_argument('init_state',
                        metavar='init_state',
                        type=int,
                        help='Initial State in the real line.(Amnt of current events)')


def sampled_poisson(args,cur_srate):
    #Create an episod e
    #  print('Generating episode')
    poisson_sp =  PoissonPath(args.length, args.poi_rate)
    time_tape, state_tape = poisson_sp.generate_history()
    # We have our tapes. Time to iterate through them
    
    #Create our sampled states
    #  print("Sampling...")
    #  srate = args.sampling_rate
    srate = cur_srate
    stimes = np.arange(0,srate*args.no_samples, srate)
    sampled_states = np.zeros_like(stimes)# Empty array for filling now
    for i in range(len(time_tape)-1):
        boolean_array = np.logical_and(stimes>= time_tape[i],stimes< time_tape[i+1] )
        indices = np.where(boolean_array)[0]
        sampled_states[indices] = i
    # We have states and times, we can take intervals of times as the ones we estiamte for poisson
    # Get the differences in states
    #  print("Estimating...")
    differences = sampled_states[1:] - sampled_states[0:-1]# These are the independent incremetents of poisson
    # Estimator
    lambda_dat = (np.sum(differences))/len(differences)

    return lambda_dat


    #  emb_x_axis, emb_hist = eventd_stationary_state(emb_state_tape, emb_hold_tape)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Compare MAP to delta')
    argparser(parser)

    args = parser.parse_args()

    # This does for now. Ill add argparse later if needed 
    levels_of_n = []

    # Three levels of rates
    rates = [0.1, 1.0, 10] # TODO: add it as a cmdline param later
    
    final_samples  = []
    length = 1000

    for rate in rates:# Each distribution

        sampled_rates = [] #Hopefully this is asymptotically normal
        print("Working with rate: ",rate)

        for i in range(length):# Create a stochastic sequence of 1k realizations
            if i % 10==0: 
                print("Generating episode ",i)
            sampled_rates.append(sampled_poisson(args,rate))

        final_samples.append(sampled_rates)

    assert(len(final_samples) == len(rates))
    fig,axs = plt.subplots(1,3)
    for i,rate in enumerate(rates):
        axs[i].hist(final_samples[i],bins=int(length*0.1))
        axs[i].set_title('Samping rate {}'.format(rate))

    #  plt.legend(True)
    plt.show()





