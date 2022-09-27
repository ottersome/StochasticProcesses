# Should be Run by default so it will be a script from the get go
import sys
from sp_sims.simulators.embeddedmc import *
from sp_sims.statistics.statistics import *

if __name__ == '__main__':
    # Create the two competitive processes
    print("Starting with numerical analysis")

    if len(sys.argv) != 2:
        print("Please provide lenth for simulations")
        exit(-1)

    length = int(sys.argv[1])

    # Craete Tapes for Embedded
    rates = {"lambda": 1,"mu":1.5} #This should keep us within the corner
    embedded_sp = EmbeddedMarkC_BD(length,rates)
    emb_tape = embedded_sp.generate_history(0)

    # Create Tapes for Race
    race_sp = RaceOfExponentials(length,rates)
    race_tape = race_sp.generate_history(0)

     


