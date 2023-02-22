# Should be Run by default so it will be a script from the get go
import sys
import time
import matplotlib.pyplot as plt
from sp_sims.simulators.stochasticprocesses import *
from sp_sims.statistics.statistics import *

def compare_stationary_event(length,initial_state):
    
    # Craete Tapes for Embedded
    rates = {"lambda": 1.0,"mu":1.5} #This should keep us within the corner
    embedded_sp = EmbeddedMarkC_BD(length,rates)
    emb_hold_tape, emb_state_tape = embedded_sp.generate_history(initial_state)

    # Create Tapes for Race
    race_sp = RaceOfExponentials(length,rates)
    race_hold_tape, race_state_tape = race_sp.generate_history(initial_state)

    # Poisson Ev
    # TODO change the sampling interval when needed
    poisson_sp = PoissonFight(length,1,rates)
    poi_hold_tape, poi_state_tape = poisson_sp.generate_history(initial_state)
    
    # Compare them with event drive
    emb_x_axis, emb_hist = eventd_stationary_state(emb_state_tape, emb_hold_tape)
    race_x_axis, race_hist = eventd_stationary_state(race_state_tape, race_hold_tape)
    poi_x_axis, poi_hist = eventd_stationary_state(poi_state_tape, poi_hold_tape)
    fig,subs = plt.subplots(1,3)
    
    emb_probs = emb_hist/np.sum(emb_hist)
    race_probs = race_hist/np.sum(race_hist)
    poi_probs = poi_hist/np.sum(poi_hist)

    max_density = np.max([emb_probs.max(),race_probs.max(),poi_probs.max()])

    subs[0].bar(emb_x_axis, emb_probs)
    #  subs[0].hist(x=emb_x_axis, bins=len(emb_x_axis), weights=emb_probs,density=True)
    subs[0].set_title('Embedded Markov Chian')
    subs[0].set_ylim(bottom=0,top=max_density)

    maxx = np.max(emb_x_axis)
    x = np.linspace(1,maxx,100)
    # Theres a different expression for n = 0. Will add later
    meep = lambda expo: (rates['lambda']/rates['mu'])**expo
    y = meep(x)
    y /= 1 + np.sum([meep(i-1) for i in range(1,1000)])
    y0 = 1/(1+ np.sum([meep(i-1) for i in range(1,1000)]))
    x = np.insert(x,0,0)
    y = np.insert(y,0,y0)
    subs[0].plot(x,y,c='r')

    #  subs[1].hist(x=race_x_axis, bins=len(race_x_axis), weights=race_probs,density=True)
    subs[1].bar(race_x_axis, race_probs)
    subs[1].set_title('Exponential Races')
    subs[1].set_ylim(bottom=0,top=max_density)
    #  subs[2].hist(x=poi_x_axis, bins=len(poi_x_axis), weights=poi_probs,density=True)
    subs[2].bar(poi_x_axis, poi_probs)
    subs[2].set_title('Poison Fight')
    subs[2].set_ylim(bottom=0,top=max_density)
    
    #
    #  # Expected Values
    #  emb_exp = emb_probs@emb_x_axis
    #  race_exp = race_probs@race_x_axis
    #  poi_exp = poi_probs@poi_x_axis

    #  print("ExpectedValue for Embedded MC: {}. ExpectedValue for Race: {}".format(emb_exp,race_exp))

    plt.show()

def compare_stationary_sample(length,sampling_rate,initial_state):
    # Craete Tapes for Embedded
    rates = {"lambda": 0.1,"mu":0.15} #This should keep us within the corner
    
    # Embeded Markov Chain Simulation
    np.random.seed(int(time.time()))
    embedded_sp = EmbeddedMarkC_BD(length,rates)
    emb_hold_tape, emb_state_tape = embedded_sp.generate_history(initial_state)

    # Create Tapes for Race
    race_sp = RaceOfExponentials(length,rates)
    race_hold_tape, race_state_tape = race_sp.generate_history(initial_state)
    
    # Compare them with event drive
    emb_x_axis, emb_hist = sampd_stationary_state(sampling_rate, emb_state_tape, emb_hold_tape)
    race_x_axis, race_hist = sampd_stationary_state(sampling_rate, race_state_tape, race_hold_tape)

    print("Histogram Count:",emb_hist)
    fig,subs = plt.subplots(1,2)

    print("For reference, last change was at: ",np.sum(race_hold_tape))

    subs[0].bar(emb_x_axis, emb_hist)
    subs[0].set_title('Embedded Markov Chian')
    subs[1].bar(race_x_axis, race_hist)
    subs[1].set_title('Races')

    plt.show()


if __name__ == '__main__':

    if len(sys.argv) != 4:
        print("Please provide appropriate parameters for simulations")
        exit(-1)

    # This does for now. Ill add argparse later if needed 
    length = int(sys.argv[1])
    sampling_rate = float (sys.argv[2])
    initial_state = int (sys.argv[3])

    #  compare_stationary_event(length,initial_state=initial_state)
    compare_stationary_sample(length,sampling_rate,0)

