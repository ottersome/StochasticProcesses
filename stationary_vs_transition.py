import numpy as np
import sys
import matplotlib.pyplot as plt
import datetime
import argparse
from sklearn.metrics import roc_curve
from math import factorial
from scipy.linalg import expm
from sp_sims.simulators.stochasticprocesses import *
from sp_sims.statistics.statistics import *
from sp_sims.estimators.algos import *
from sp_sims.sanitycheck.truebirthdeath import *
from sp_sims.utils.utils import *
from sp_sims.curve_filters.lowpass import *
import random
from tqdm import tqdm
from time import sleep




def get_true_trans_probs(Q):
    #  P = power_series_exp(Q)
    P = expm(Q)
    # Get the Norm 
    #  print(np.linalg.norm(Q,ord='fro'))
    return P


def print_mat_text(mat, axs):
    axs.imshow(mat)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            axs.text(j,i,"%.2f " % mat[i,j],ha="center",va="center",color="w",fontsize=8)

# This function will take a guess at which process generated the entire thing.
# Under the very unsafe assumption that we can take observations as independent
def take_a_stationary_guess(tape, p0, p1):
    num = 0
    denum = 0
    # All we need for this is the amount of states
    for i in range(len(tape)):
        state = tape[i]
        num += np.log(p0[state])
        denum += np.log(p1[state])

    return 0 if num > denum else 1

def take_a_guess(tape, p0, p1):
    num = 0
    denum = 0
    # num = 1
    # denum = 1
    for i in range(len(tape)-1):
        from_state = tape[i]
        to_state = tape[i+1]
        # num  *= p0[from_state,to_state]
        # denum *= p1[from_state,to_state]
        num += np.log(p0[from_state,to_state])
        denum += np.log(p1[from_state,to_state])

    return 0 if num > denum else 1

def rgt():
    return (random.random(),random.random(),random.random())

def return_ls(tape, p0, p1):
    num = 1
    denum = 1
    for i in range(len(tape)-1):
        from_state = tape[i]
        to_state = tape[i+1]
        num  *= p0[from_state,to_state]
        denum *= p1[from_state,to_state]
    return num,denum

def return_lr(tape, p0, p1):
    resl = 1
    for i in range(len(tape)-1):
        from_state = tape[i]
        to_state = tape[i+1]
        resl  *= p0[from_state,to_state]
        resl /= p1[from_state,to_state]
    return resl


def return_lls(tape, p0, p1):
    num = 0
    denum = 0
    for i in range(len(tape)-1):
        from_state = tape[i]
        to_state = tape[i+1]
        num += np.log(p0[from_state,to_state])
        denum += np.log(p1[from_state,to_state])

    return -num,-denum

def test_estimator(rates,args):
    # Get the Probabilities
    roe = RaceOfExponentials(args.length,rates,state_limit=args.state_limit)
    holdTimes_tape, state_tape = roe.generate_history(args.init_state)
    sampled_tape = simple_sample(args.samprate, state_tape, holdTimes_tape)
    p_hat = state_transitions(np.full_like(sampled_tape, 1/args.samprate), sampled_tape)
    
    # A Figure
    fig,axs = plt.subplots(1,2)
    fig.tight_layout()
    fig.set_size_inches(10,10)

    # Building True Statistics
    tgm = generate_true_gmatrix(rates, args.state_limit)
    true_p = get_true_trans_probs(Q=tgm*(1/args.samprate))


    # print_mat_text(p_hat, axs[0])
    axs[0].set_title('Estimated Probability Matrix')
    # print_mat_text(true_p, axs[1])
    axs[1].set_title('True Probability Matrix')

    plt.show()

def manual_init():
    import pandas as pd
    args = pd.DataFrame()
    args.detection_guesses = 1000
    args.length = 1000
    args.xres = 16
    args.state_limit = 1
    args.init_state = 0
    args.num_samples = 10
    args.figtitle = 'L1_L2_EquallyLikely'


if __name__ == '__main__':
    # Go through arguments
    args = argparser()
    #np.random.seed(123)

    # Created Tapes
    #  rates0 = {"lam": 4/10,"mu":12/10}
    # rates1 = {"lam": 100/10,"mu":150/10}
    rates0 = {"lam": 4/10,"mu":14/10}
    rates1 = {"lam": 8/10,"mu":12/10}
    print("Null Rates ", rates0)
    print("Alternative Rates ", rates1)

    print("Holding rates are : {} {}".format(rates0['lam']+rates0['mu'],rates1['lam']+rates1['mu']))
    print("Holding time intervals are : {} {}".format(1/(rates0['lam']+rates0['mu']),1/(rates1['lam']+rates1['mu'])))
    
    # Sanity Check
    # test_estimator(rates1,args)
    
    # We dont even need that. We just neeed a Q matrix ||Q|| < ln 2

    # Create(by hand) Q matrix
    #rates = [{"lam":args.lam, "mu":args.mu},{"lam":1.4, "mu":3.9}]
    rates = [rates0,rates1]
    # We now have the analytical results we need

    # We will create multiple different samples here
    # samp_rates = [args.samprate *2 ** j for j in range(10)]
    # samp_rates = np.linspace(0.001,16,1000)
    samp_rates = np.logspace(-3,7,args.xres, base=2)
    # samp_rates = 100/np.array([100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 1])
    # Decimation Using 10 

    tgm0 = np.array([[-rates0['lam'],rates0['lam']],[rates0['mu'],-rates0['mu']]])
    tgm1 = np.array([[-rates1['lam'],rates1['lam']],[rates1['mu'],-rates1['mu']]])

    hit_rates_transition = [] 
    hit_rates_stationary= [] 
    l0s,l1s = ([],[])# Likelihood 
    v0s,v1s = ([],[])# Variances
    mi0s,mi1s = ([],[])
    ma0s,ma1s = ([],[])

    j = 0

    true_values = np.random.choice(2,args.detection_guesses)
    #true_values = np.ones(args.detection_guesses).astype(int)
    hts, sts = ([],[])
    last_times  = []
    for i in range(args.detection_guesses):
        roe = RaceOfExponentials(args.length,rates[true_values[i]],state_limit=args.state_limit)
        holdTimes_tape, state_tape = roe.generate_history(args.init_state)
        hts.append(holdTimes_tape); sts.append(state_tape)
        last_times.append(np.cumsum(holdTimes_tape)[-1])
    
    print("Max:{} min:{} and mean:{} last times".format(np.max(last_times), np.min(last_times), np.mean(last_times)))

    fprs = [] # False Positive Rates
    fnrs = [] # False Negative Rates

    sensitivities = []
    invspecificities = []

    base_samp_rate = samp_rates[-1]
    
    guesses_transition = np.zeros((len(samp_rates), args.detection_guesses))
    guesses_stationary = np.zeros((len(samp_rates), args.detection_guesses))
    # Likelihoods for # SampRates X # Detection Guesses
    l0_smpRates_x_gusses = np.zeros((len(samp_rates), args.detection_guesses))
    l1_smpRates_x_gusses = np.zeros((len(samp_rates), args.detection_guesses))
    
    # Transition Probabilities
    true_p0s = []
    true_p1s = []

    # Stationary Probabilities 
    p0_stationary = [rates0['mu']/np.sum(list(rates0.values())),rates0['lam']/np.sum(list(rates0.values()))]
    p1_stationary = [rates1['mu']/np.sum(list(rates1.values())),rates1['lam']/np.sum(list(rates1.values()))]
    

    for srIdx, cur_samp_rate in enumerate(samp_rates):
        true_p0s.append(get_true_trans_probs(Q=tgm0*(1/cur_samp_rate)))
        true_p1s.append(get_true_trans_probs(Q=tgm1*(1/cur_samp_rate)))

    #############################################
    # Take Detection Guesses
    #    (and return Likelihood Ratio)
    #############################################
    for i in tqdm(np.arange(args.detection_guesses)):
        sampled_tape = quick_sample(base_samp_rate, sts[i],hts[i])
        
        for srIdx, cur_samp_rate in enumerate(samp_rates):
            true_p0 = true_p0s[srIdx]
            true_p1 = true_p1s[srIdx]
            true_ps = [true_p0,true_p1]
            
            decimateInterval = int(base_samp_rate/cur_samp_rate)
            tmpSampTape = sampled_tape[0::decimateInterval]
            ###### Option 1 without limit on the number of samples ############
            # guesses[srIdx, i] = take_a_guess(tmpSampTape, true_p0, true_p1)
            # # l0, l1 = return_ls(tmpSampTape, true_p0, true_p1) # Shall not use this as it results in very small quantities
            # l0 = return_lr(tmpSampTape, true_p0, true_p1)
            # l1 = 1
            ###################################################################
            ############ Option 2 with limited number of samples ##############
            guesses_transition[srIdx, i] = take_a_guess(tmpSampTape[0:args.num_samples], true_p0, true_p1)
            guesses_stationary[srIdx, i] = take_a_stationary_guess(tmpSampTape[0:args.num_samples], p0_stationary, p1_stationary)
            # l0, l1 = return_ls(tmpSampTape[0:args.num_samples], true_p0, true_p1) # Shall not use this as it results in very small quantities
            l0 = return_lr(tmpSampTape[0:args.num_samples], true_p0, true_p1)
            l1 = 1
            ###################################################################

            l0_smpRates_x_gusses[srIdx, i] = l0
            l1_smpRates_x_gusses[srIdx, i] = l1
            


    num_negs = np.sum(true_values == 0)#TN + FP
    num_pos = np.sum(true_values == 1)#TP + FN
            
    #############################################
    # Likelihoods
    #############################################
    for srIdx, cur_samp_rate in enumerate(samp_rates):
        # All Guesses for this particular sampling rate
        guess_transition = guesses_transition[srIdx]
        guess_stationary = guesses_stationary[srIdx]

        l0c = l0_smpRates_x_gusses[srIdx]
        l1c = l1_smpRates_x_gusses[srIdx]
        hits_index = (true_values == guess_transition)
        tp = (true_values[hits_index] == 1).sum()
        tn = (true_values[hits_index] == 0).sum()

        sensitivities.append(tp/num_pos)
        invspecificities.append(1-(tn/num_negs))

        fprs.append((num_negs-tn)/(num_negs))
        fnrs.append((num_pos-tp)/(num_pos))# Type 2 Error
        
        #This section for H0 case
        idxLocs = (true_values == 0)
        lInterest = l0c[idxLocs]
        l0s.append(np.mean(lInterest))
        v0s.append(np.std(lInterest))
        mi0s.append(np.min(lInterest))
        ma0s.append(np.max(lInterest))
        
        #This section for H1 case
        idxLocs = (true_values == 1)
        lInterest = 1/(l0c[idxLocs])
        l1s.append(np.mean(lInterest))
        v1s.append(np.std(lInterest))
        mi1s.append(np.min(lInterest))
        ma1s.append(np.max(lInterest))

        num_hits_transition = (true_values == guess_transition).sum()
        num_hits_stationary = (true_values == guess_stationary).sum()
        hit_rates_transition.append(num_hits_transition/args.detection_guesses)
        hit_rates_stationary.append(num_hits_stationary/args.detection_guesses)

#    smoothed_hits = savitzky_golay(hit_rates, 21, 3)
    
    #############################################
    # Plotting
    #############################################
    fig, axs = plt.subplots(1,2)
    fig.tight_layout()
    fig.set_size_inches(16,10)

    plt.rcParams['text.usetex'] = True
    
    # Accuracy Plotting
    axs[0].plot(samp_rates,hit_rates_transition,label='Transition Prob Accuracy',color='red',alpha=1,linewidth=2) 
    axs[0].plot(samp_rates,hit_rates_stationary,label='Stationary Prob Accuracy',color='orange',alpha=1,linewidth=2) 
    # Plot the Scalar Rates
    for i,rate in enumerate(rates):
        axs[0].axvline(rate['lam'],label='$\lambda_'+str(i)+'=$'+str(rate['lam']),c=rgt())
        axs[0].axvline(rate['mu'],label='$\mu_'+str(i)+'=$'+str(rate['mu']),c=rgt())
    axs[0].set_title('Accuracy with Respect to Sampling Rate')
    axs[0].set_xscale("log",base=2)
    axs[0].legend()
    
    l0s,l1s,v0s,v1s = (np.array(l0s), np.array(l1s), np.array(v0s), np.array(v1s))
    # Min-Max Values of Likelihoods
    axs[1].fill_between(samp_rates, mi0s,ma0s, color='blue', alpha=0.2)
    axs[1].fill_between(samp_rates, mi1s,ma1s, color='green', alpha=0.2)
    # Likelihoods
    axs[1].plot(samp_rates, l0s,label='$L_0$', c='blue')
    axs[1].plot(samp_rates, l1s,label='$L_1$',c='green')
    # Set Our Scales
    axs[1].set_title('Likelihoods (H0: $\\frac{L_0}{L_1}$; H1: $\\frac{L_1}{L_0}$)')
    axs[1].set_xscale("log",base=2)
    axs[1].set_yscale("log",base=10)
    axs[1].legend()
        
    now = datetime.datetime.now()
    time_frm = now.strftime('%Y-%m-%dT-%H_%M_%S')

    title = 'Images/Ernest/d{}_'.format(time_frm)+args.figtitle if args.figtitle != None else 'Images/Ernest/run_{}'.format(time_frm)
    plt.savefig(title)
    plt.show()

    # ROC Cruve
    plt.plot(invspecificities, sensitivities, '-o')
    plt.title('ROC Curve')
    plt.xlabel('1-Specificity')
    plt.ylabel('Sensitivity')
    plt.xlim([0,1])
    plt.ylim([0,1])
    title = 'Images/Ernest/d{}_'.format(time_frm)+'ROC'
    plt.savefig(title)
    plt.show()

