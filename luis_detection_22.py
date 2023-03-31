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
    # rates0 = {"lam": 4/10,"mu":12/10}
    # rates1 = {"lam": 100/10,"mu":122/10}
    rates0 = {"lam": 4/10,"mu":4/10}
    rates1 = {"lam": 100/10,"mu":100/10}
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
    samp_rates = np.logspace(-3,8,args.xres, base=2)
    # samp_rates = 100/np.array([100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 1])

    tgm0 = np.array([[-rates0['lam'],rates0['lam']],[rates0['mu'],-rates0['mu']]])
    tgm1 = np.array([[-rates1['lam'],rates1['lam']],[rates1['mu'],-rates1['mu']]])


    # Chose a random rate to test:
    #ran_index = np.random.randint(780,1000,1)
    #print("Random Index is ", ran_index)

    hit_rates = [] 
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


    sensitivities = []
    invspecificities = []

    base_samp_rate = samp_rates[-1]
    
    guesses = np.zeros((len(samp_rates), args.detection_guesses))
    # LIkelihoods in 2D. On First Axis Sampling Rates, On Second Axis number of guesses
    l0cs = np.zeros((len(samp_rates), args.detection_guesses))
    l1cs = np.zeros((len(samp_rates), args.detection_guesses))
    
    true_p0s = []
    true_p1s = []
    # Crete all the *known* probability functions.
    for srIdx, cur_samp_rate in enumerate(samp_rates):
        true_p0s.append(get_true_trans_probs(Q=tgm0*(1/cur_samp_rate)))
        true_p1s.append(get_true_trans_probs(Q=tgm1*(1/cur_samp_rate)))

    # Go Over the Detection Guesses
    for i in tqdm(np.arange(args.detection_guesses)):
        
        # Prepare for Decimation: First sample with the quickest rate. -> Smallest Unit of Time Interval
        sampled_tape = quick_sample(base_samp_rate, sts[i],hts[i])
        # Go Over the 
        for srIdx, cur_samp_rate in enumerate(samp_rates):
            # Fetch Values for Current Rate
            true_p0 = true_p0s[srIdx]
            true_p1 = true_p1s[srIdx]
            true_ps = [true_p0,true_p1]
            
            #LG: How many of the baseline intervals fit into the slow rate interval
            decimateInterval = int(base_samp_rate/cur_samp_rate)
            tmpSampTape = sampled_tape[0::decimateInterval]
            ###################################################
            # Option 1 without limit on the number of samples #
            ############################################
            # guesses[srIdx, i] = take_a_guess(tmpSampTape, true_p0, true_p1)
            # # l0, l1 = return_ls(tmpSampTape, true_p0, true_p1) # Shall not use this as it results in very small quantities
            # l0 = return_lr(tmpSampTape, true_p0, true_p1)
            # l1 = 1
            ############################################
            # Option 2 with limited number of samples ##
            ############################################
            limited_sampled_tape = tmpSampTape[0:args.num_samples] 
            guesses[srIdx, i] = take_a_guess(limited_sampled_tape, true_p0, true_p1)
            l0, l1 = return_ls(tmpSampTape[0:args.num_samples], true_p0, true_p1) # Shall not use this as it results in very small quantities
            # l0 = return_lr(limited_sampled_tape, true_p0, true_p1)
            # l1 = 1
            ###################################################################

            # Add Per Sample
            l0cs[srIdx, i] = l0
            l1cs[srIdx, i] = l1
            
    num_negs = np.sum(true_values == 0)#TN + FP
    num_pos = np.sum(true_values == 1)#TP + FN

    fprs = [] # False Positive Rates
    fnrs = [] # False Negative Rates

    # At this point we have our guesses saved
    # At this point we can start looking at 
    # * Sensitivities
    # Inv Specificities
    # Loop through all Sampling Rates we have used
    for srIdx, cur_samp_rate in enumerate(samp_rates):
        # For Plotting ROC Curve
        guess = guesses[srIdx]
        # Take likelihoods for this sampling rate
        l0c = l0cs[srIdx]
        l1c = l1cs[srIdx]

        # Hits and True Probabilities
        hits_index = (true_values == guess)
        tp = (true_values[hits_index] == 1).sum()
        tn = (true_values[hits_index] == 0).sum()

        # Correct
        sensitivities.append(tp/num_pos)
        invspecificities.append(1-(tn/num_negs))

        # False Positive and Negative Rates
        fprs.append((num_negs-tn)/(num_negs))
        fnrs.append((num_pos-tp)/(num_pos))# Type 2 Error

        
        #This section for H0 case
        # I dont knwo why Ernest focuses on when we get it right
        # idxLocs = (true_values == 0)
        # lInterest = l0c[idxLocs]# Just get the one of interest
        lInterest = l0c

        l0s.append(np.mean(lInterest))
        v0s.append(np.std(lInterest))
        mi0s.append(np.min(lInterest))
        ma0s.append(np.max(lInterest))
        
        #This section for H1 case
        # # Again same as above
        # idxLocs = (true_values == 1)
        # lInterest = l1c[idxLocs] # Get Likelihood 1
        lInterest = l1c

        l1s.append(np.mean(lInterest))
        v1s.append(np.std(lInterest))

        mi1s.append(np.min(lInterest))
        ma1s.append(np.max(lInterest))

        num_hits = (true_values == guess).sum()
        hit_rates.append(num_hits/args.detection_guesses)


    fig, axs = plt.subplots(1,2)
    fig.tight_layout()
    fig.set_size_inches(16,10)

    plt.rcParams['text.usetex'] = True
    
    axs[0].plot(samp_rates,hit_rates,label='Accuracy',color='green',alpha=1,linewidth=2) 
    axs[0].plot(samp_rates,fprs,label='Type I Error',color='orange',alpha=1,linewidth=2) 
    axs[0].plot(samp_rates,fnrs,label='Type II Error',color='red',alpha=1,linewidth=2) 
    
    for i,rate in enumerate(rates):
        axs[0].axvline(rate['lam'],label='$\lambda_'+str(i)+'=$'+str(rate['lam']),c=rgt())
        axs[0].axvline(rate['mu'],label='$\mu_'+str(i)+'=$'+str(rate['mu']),c=rgt())
    axs[0].set_title('Accuracy with Respect to Sampling Rate')
    axs[0].set_xscale("log",base=2)
    axs[0].legend()
    
    l0s,l1s,v0s,v1s = (np.array(l0s), np.array(l1s), np.array(v0s), np.array(v1s))
    axs[1].fill_between(samp_rates, l0s-v0s,l0s+v0s, color='blue', alpha=0.2)
    axs[1].fill_between(samp_rates, l1s-v1s,l1s+v1s, color='green', alpha=0.2)

    for i,rate in enumerate(rates):
        axs[1].axvline(rate['lam'],label='$\lambda_'+str(i)+'=$'+str(rate['lam']),c=rgt())
        axs[1].axvline(rate['mu'],label='$\mu_'+str(i)+'=$'+str(rate['mu']),c=rgt())
    axs[1].plot(samp_rates, l0s,label='$L_0$', c='blue')
    axs[1].plot(samp_rates, l1s,label='$L_1$',c='green')
    axs[1].set_title('Individual Likelihoods')
    axs[1].set_xscale("log",base=2)
    axs[1].set_yscale("log",base=10)
    axs[1].legend()
        
    now = datetime.datetime.now()
    time_frm = now.strftime('%Y-%m-%dT-%H_%M_%S')

    title = 'Images/Detection/d{}_'.format(time_frm)+args.figtitle if args.figtitle != None else 'Images/Detection/run_{}'.format(time_frm)
    plt.savefig(title)
    plt.show()

    # ROC Cruve
    #plt.plot(invspecificities, sensitivities, '-o')
    plt.scatter(invspecificities, sensitivities, s=5+np.log(samp_rates)+np.log(samp_rates[0]))
    plt.title('ROC Curve')
    plt.xlabel('1-Specificity')
    plt.ylabel('Sensitivity')
    plt.xlim([0,1])
    plt.ylim([0,1])
    title = 'Images/Detection/d{}_'.format(time_frm)+'ROC'
    plt.savefig(title)
    plt.show()

