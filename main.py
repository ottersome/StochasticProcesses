# Should be Run by default so it will be a script from the get go
import sys
import matplotlib.pyplot as plt
from sp_sims.simulators.embeddedmc import *
from sp_sims.statistics.statistics import *

def compare_stationary_event(length,initial_state):
    
    # Craete Tapes for Embedded
    rates = {"lambda": 0.1,"mu":0.1} #This should keep us within the corner
    embedded_sp = EmbeddedMarkC_BD(length,rates)
    emb_hold_tape, emb_state_tape = embedded_sp.generate_history(initial_state)

    # Create Tapes for Race
    race_sp = RaceOfExponentials(length,rates)
    race_hold_tape, race_state_tape = race_sp.generate_history(initial_state)

    # Poisson Ev
    poisson_sp = PoissonFight(length,1,rates)
    poi_hold_tape, poi_state_tape = race_sp.generate_history(initial_state)
    
    # Compare them with event drive
    emb_x_axis, emb_hist = eventd_stationary_state(emb_state_tape, emb_hold_tape)
    race_x_axis, race_hist = eventd_stationary_state(race_state_tape, race_hold_tape)
    poi_x_axis, poi_hist = eventd_stationary_state(poi_state_tape, poi_hold_tape)
    print("Histogram Count:",emb_hist)
    fig,subs = plt.subplots(1,3)

    subs[0].bar(emb_x_axis, emb_hist)
    subs[0].set_title('Embedded Markov Chian')
    subs[1].bar(race_x_axis, race_hist)
    subs[1].set_title('Exponential Races')
    subs[2].bar(poi_x_axis, poi_hist)
    subs[2].set_title('Poison Fight')
    
    # Get Probabilities
    emb_probs = emb_hist/np.sum(emb_hist)
    race_probs = race_hist/np.sum(race_hist)
    poi_probs = race_hist/np.sum(poi_hist)

    # Expected Values
    emb_exp = emb_probs@emb_x_axis
    race_exp = race_probs@race_x_axis
    poi_exp = race_probs@poi_x_axis

    print("ExpectedValue for Embedded MC: {}. ExpectedValue for Race: {}".format(emb_exp,race_exp))

    plt.show()

def compare_stationary_sample(length,sampling_rate,initial_state):
    # Craete Tapes for Embedded
    rates = {"lambda": 0.1,"mu":0.15} #This should keep us within the corner
    
    # Embeded Markov Chain Simulation
    np.random.seed(123)
    embedded_sp = EmbeddedMarkC_BD(length,rates)
    emb_hold_tape, emb_state_tape = embedded_sp.generate_history(initial_state)

    # Create Tapes for Race
    np.random.seed(123)
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
    # Create the two competitive processes
    print("Starting with numerical analysis")

    if len(sys.argv) != 3:
        print("Please provide lenth for simulations")
        exit(-1)

    length = int(sys.argv[1])
    sampling_rate = float (sys.argv[2])
    #  compare_stationary_event(length,initial_state=10000)
    compare_stationary_sample(length,sampling_rate,0)

