import numpy as np
from .abssimulator import StochasticSimulator,SPManager

# For now I assume distributions are
# np.random
#
class BinomialSP(StochasticSimulator):
    # n an p binomial parameters
    def __init__(length,n,p):
        super().__init__(length)
        self.n = n ; self.p = p;
    def generate_history():
        # Always generate new tape
        tape = np.random.binomial(self.n,self.p,self.length)

    # Add functions that are binomial stochastic process spicific
    def bp_specific():
        pass
   

class ExponentialSumSP(StochasticSimulator):
    def __init__(self,length, rate):
        super().__init__(length)
        self.rate = rate

    def generate_history(self):
        # Always generate new tape
        return np.random.exponential(1/self.rate,self.length)

    # Add functions that are exponential process spicific(if any)
    def _specific(self):
        pass
   

class ExponentialMinP(StochasticSimulator):
    def __init__(length,rates : list):
        super().__init__(length)
        self.rates = rates


    def generate_history():
        # Always generate new tape
        tape = np.array([self.length,len(self.rates)])
        for i,rate in enumerate(self.rates): tape[:,i]= np.random.exponential(self.n,self.p,self.length)
        final_tape = np.min(tapes,1)# Get the minimum exponential at each instance
        # This should be equal in distribution to the holdin gtimes


    # Add functions that are binomial stochastic process specific
    def _specific():
        pass
 


# SP manager would manage different stochastic processs
# Embedded Markov Chain Uses 
#   * A single holding time e
class EmbeddedMarkC_BD(SPManager):

    # Can we have an api for distributions?
    # we will assume for now all variables are identically distributed.
    def __init__(self, length,rates,state_limit):
        print("We are setting up our embedded markov chian")
        self.length = length

        self.a_rate = rates['lambda']
        self.s_rate = rates['mu']

        self.a_prob = self.a_rate/(self.a_rate+self.s_rate)
        self.s_prob = self.s_rate/(self.a_rate+self.s_rate)

        self.state_limit = state_limit


    # @@
    def generate_history(self, initial_state):
        exp_sp = ExponentialSumSP(self.length,self.a_rate+self.s_rate)
        self.holding_times = exp_sp.generate_history()
        
        # We can initialize this beforehand because the probability 
        # distribution at every point is the same
        birth_or_death = np.random.choice([-1,1],self.length-1,p=[self.s_prob, self.a_prob])
        states = [initial_state]

        # We go through all the n-1 transitions
        for i in range(self.length-1):
            if states[-1]==0 and birth_or_death[i] == -1: 
                # In case of being at state 0 
                # we only have the probability of moving right
                temp1 = np.random.exponential(scale=(1/self.a_rate))
                temp2 = np.random.exponential(scale=(1/self.s_rate))
                while temp1 < temp2:
                    temp1 = np.random.exponential(scale=(1/self.a_rate))
                    temp2 = np.random.exponential(scale=(1/self.s_rate))
                new_time = temp1
                self.holding_times[i] = new_time
                # self.holding_times[i] = np.random.exponential(scale=(1/self.a_rate))
                birth_or_death[i] = 1
            if states[-1]==self.state_limit and birth_or_death[i] == 1: 
                temp2 = np.random.exponential(scale=(1/self.s_rate))
                temp1 = np.random.exponential(scale=(1/self.a_rate))
                while temp2 < temp1:
                    temp1 = np.random.exponential(scale=(1/self.a_rate))
                    temp2 = np.random.exponential(scale=(1/self.s_rate))
                new_time = temp2
                self.holding_times[i] = new_time
                # self.holding_times[i] = np.random.exponential(scale=(1/self.s_rate))
                birth_or_death[i] = -1

            states.append(states[-1] + birth_or_death[i])
        
        # Manually fix last holding time
        if states[-1] == 0 : self.holding_times[-1] = np.random.exponential(scale=(1/self.a_rate))

        #This returns our tape to be later managed by statistics
        return (self.holding_times,states);

    def simulate_n_processes(self):
        pass
class PoissonPath(SPManager):
    # Homogeneous Poi -> Single Rate
    def __init__(self,num_jumps,rate):
        self.rate = rate
        self.amnt_events = num_jumps
    def generate_history(self):
        # Use an exponential to generate intervals so we stay in the real time.
        times = np.random.exponential(scale=1/self.rate,size=self.amnt_events)
        times_tape = [0]
        times_tape.extend(np.cumsum(times))
        states = np.arange(0,len(times_tape))

        return (times_tape, states)
    def simulate_n_processes(self):
        pass


class PoissonFight(SPManager):

    def __init__(self,length,sampling_interval,rates):
        # Sampling inteval will modify our rates

        self.rate_arr = rates['lambda']*sampling_interval
        self.rate_ser = rates['mu']*sampling_interval
        self.sampl_int = sampling_interval
        self.amnt_events = length


        # Was playing with some ideas here, Dont think I need them for now
        # V V V V V V V V V V V V V V V V V V V V V V V V V V V V V V V V V 
        # We fix rates to fit this intevals of time
        #  self.time_length = length*(1/(self.rate_arr+self.rate_ser))
        #  self.holding_variation = (1/(self.rate_arr+self.rate_ser)**2)
        #  self.time_length += length*(2*self.holding_variation)# Just to hold the variations as well

    def generate_history(self,initial_state):
        # We have two poisson processes fighting 
        birth = np.random.poisson(self.rate_arr,size=(self.amnt_events-1))
        death = np.random.poisson(self.rate_ser,size=(self.amnt_events-1))

        trend = birth-death

        states = [initial_state]
        for i in range(self.amnt_events-1):
            movement = trend[i]
            if(states[i] == 0):
                movement = birth[i]
            states.append(max(states[-1] + movement,0))

        holding_times = np.full_like(states, self.sampl_int)

        return (holding_times,states)
    def simulate_n_processes(self):
        pass


# This method of simulating Markov Chains take 
class RaceOfExponentials(SPManager):

    def __init__(self, length,rates, state_limit=-1):
        self.length = length
        self.a_rate = rates['lambda']
        self.s_rate = rates['mu']
        self.state_limit = state_limit

    def generate_history(self,initial_state):
        # Create two clocks racing for length
        race = np.zeros(shape=[self.length,2])
        # Death
        race[:,0] = np.random.exponential(scale=(1/self.s_rate),size=self.length)
        # Birth
        race[:,1] = np.random.exponential(scale=(1/self.a_rate),size=self.length)
        
        # Now get min and the index
        holding_times = np.min(race,axis=1)# Values
        bd = np.argmin(race,axis=1)
        bd[bd==0] = -1# Set to deaths

        states = [initial_state]

        # Generate the path
        for i in range(self.length-1):
            cur_state = states[-1]
            change = bd[i]
            if cur_state == 0 and change == -1:# We only take birth 
                # temp1 = np.random.exponential(scale=(1/self.a_rate))
                # temp2 = np.random.exponential(scale=(1/self.s_rate))
                # while temp2 < temp1:
                #     temp1 = np.random.exponential(scale=(1/self.a_rate))
                #     temp2 = np.random.exponential(scale=(1/self.s_rate))
                # new_time = temp1
                # holding_times[i] = new_time
                holding_times[i] = race[i,1]
                change = 1
            if cur_state == self.state_limit and change==1:
                # temp1 = np.random.exponential(scale=(1/self.a_rate))
                # temp2 = np.random.exponential(scale=(1/self.s_rate))
                # while temp1 < temp2:
                #     temp1 = np.random.exponential(scale=(1/self.a_rate))
                #     temp2 = np.random.exponential(scale=(1/self.s_rate))
                # new_time = temp2
                # holding_times[i] = new_time
                holding_times[i] = race[i,0]
                change = -1
            states.append(cur_state + change)

        # Make sure last state is representative
        if states[-1] == 0: holding_times = race[-1,1]

        return holding_times,states

    def simulate_n_processes(self):
        pass
        
# This method of simulating the true birth death process
class TrueBirthDeath(SPManager):

    def __init__(self, length,rates):
        self.length = length
        self.a_rate = rates['lambda']
        self.s_rate = rates['mu']

    def generate_history(self,initial_state):
        BirthPos = np.random.exponential(scale=(1/self.a_rate),size=self.length)
        Ages = np.random.exponential(scale=(1/self.s_rate),size=self.length)
        
        states = [initial_state]
        holding_times = []
        
        BirthLocs = np.cumsum(BirthPos)
        DeathLocs = BirthLocs + Ages
        DeathLocs = np.sort(DeathLocs)
        currTS = 0
        
        idxBirth = 0
        idxDeath = 0
        
        for i in range(self.length):
            if BirthLocs[idxBirth] < DeathLocs[idxDeath]:
                states.append(states[-1] + 1)
                holding_times.append(BirthLocs[idxBirth] - currTS)
                currTS = BirthLocs[idxBirth]
                idxBirth += 1
            else:
                states.append(states[-1] - 1)
                holding_times.append(DeathLocs[idxDeath] - currTS)
                currTS = DeathLocs[idxDeath]
                idxDeath += 1
                
        # xaxis = np.cumsum(holding_times)
        # xaxis = np.expand_dims(xaxis, 1)
        # xaxis_full = np.concatenate((xaxis, xaxis), axis=1)
        # xaxis_full = xaxis_full.flatten()
        
        # xaxis_full = np.concatenate(([0], xaxis_full))
        
        # states_one = np.array(states)
        # states_one = np.expand_dims(states_one, 1)
        # states_full = np.concatenate((states_one, states_one), axis=1)
        # states_full = states_full.flatten()
        # states_full = states_full[0:-1]
        
        # plt.plot(xaxis_full[0:1000], states_full[0:1000])
        
        states = states[0:-1]
        
        return holding_times,states

    def simulate_n_processes(self):
        pass
