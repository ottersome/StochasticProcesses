import numpy as np
from abc import ABC,abstractmethod

# TODO: This document ought to give usw ways to going through the history 
# and being able to summarize certain statistics. Such as waiting times and 
# Recurrence Times(Avg Time to go from one state to another)
# Queueing Size

# This guy right here receives realized sequence of i.i.d. r.v.  
# Then creates statistics on this allone(till I find it impossible)
class AbsStatisticGenerator(ABC):
    @abstractmethod
    def __init__(self):
        pass
    @abstractmethod # Will generate our statistics
    def generate(self):
        pass
    @abstractmethod
    def show_state_dist(self) :
        pass

class StatisticGenerator(AbsStatisticGenerator):

    # Matrix is in column major order. i.e. each sequence is a column
    def __init__(self, sps):
        # We will receive a list of list
        self.tapes =  sps

    # Will look at the tape and 
    # Im afraind I will end up abusing the amount of tapes. Lets see
    def state_transitions(self):
        # Because each run may have different states we will generate
        # unque elements per run

        unique_elements,inverse, counts, = np.unique(
                axis=0,
                return_counts=True,
                return_inverse=True)
        
        # of unique elements 
        num_unique_elements = [len(num_unique_elements) for ue in unique_elements]
        
        # With inverse we can form transitions
        init_states = inverse[:-1]
        next_states = inverse[1:]

        # State Transitions Matrices
        state_transitions = np.array(1,)

        # Form Tuples
        tuples = np.array([init_states, next_states])
        # Count the unique tuples
        tuple_count = np,unique()

    def stationary_state(self):
        # Still assuming tape matrix
        return np.unique(self.tapes,return_counts=True)

class SamplingGenerator(self):

    def __init__(self, tapes, dt):
        # We will receive a list of list
        self.transition_times = tapes['holding_times']
        self.states = tapes['states']
        self.delta_t = dt

    def stationary_state(self):
        # We just have to get a certain percentage of the states
        # Create array of unique elements
        unique_elem = np.unique(self.states)

        for i in np.arange(0,np.sum(self.transition_times),dt):
            # This is closer to uniform

    


