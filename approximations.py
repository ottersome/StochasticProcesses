import numpy as np
import sys
import matplotlib.pyplot as plt
import argparse
import scipy
from sp_sims.simulators.stochasticprocesses import *
from sp_sims.statistics.statistics import *
from sp_sims.estimators.algos import *
from sp_sims.sanitycheck.truebirthdeath import *
from sp_sims.utils.utils import *

def log_matrix_approx(state_tape, holdTimes_tape,args):
    # Setting up Figures fig,axs = plt.subplots(1,2)
    font_size = 8
    samp_time = 1/args.samprate
    fig,axs = plt.subplots(1,2)
    fig.tight_layout()
    fig.set_size_inches(10,10)

    # Get a tape of samples at a particular rate
    print(r"Calculating the $\Delta t = ${}-Sampeld Transition Matrix MLE...".format(args.samprate))
    sampled_tape = simple_sample(args.samprate, state_tape, holdTimes_tape)
    p_hat = state_transitions(np.full_like(sampled_tape, samp_time), sampled_tape)

    # Show the p-hat matrix
    axs[0].imshow(p_hat)
    for i in range(p_hat.shape[0]):
        for j in range(p_hat.shape[1]):
            axs[0].text(j,i,"%2.2f " % p_hat[i,j],ha="center",va="center",color="w",fontsize=font_size)
    axs[0].set_title(r'Transition matrix for $\Delta t=1/$ {}'.format(args.samprate))

    # Sanity Check
    print("Determinant of the transition matrix : {}".format(np.linalg.det(p_hat)))
    # Compute the Solution
    Qdt = power_series_log(p_hat, 3)
    #  Qdt = scipy.linalg.logm(p_hat)
    Q = Qdt / (samp_time)

    # Show the Gneeratort matrix recovered from it
    axs[1].imshow(Q)
    for i in range(Q.shape[0]):
        for j in range(Q.shape[1]):
            axs[1].text(j,i,"%2.2f " % Q[i,j],ha="center",va="center",color="w",fontsize=font_size)

    axs[1].set_title(
            r"Estimated Generator Matrix through samples at $\Delta t = ${}. $\mu$:{} $\lambda$:{} ".format(args.samprate,args.mu,args.lam))
    plt.savefig('Images/MLE_sample_log_dt{}_lam{},_mu{}.png'.format(
        args.samprate,args.lam,args.mu, format='eps',dpi=200))
    plt.show()
    print("Calculated. Displaying...")

def determinant_and_sampling(state_tape, holdTimes_tape,args, doublings=8, initial_rate=1):

    fig,axs = plt.subplots(1,1)
    fig.tight_layout()
    fig.set_size_inches(10,10)
    samp_time = 1/initial_rate

    dets = []
    for i in range(doublings+1):
        rate = initial_rate*2**i
        print(f'Using rate {rate}')
        sampled_tape = simple_sample(rate, state_tape, holdTimes_tape)
        p_hat = state_transitions(np.full_like(sampled_tape, 1/rate), sampled_tape)
        dets.append(np.linalg.det(p_hat))

    axs.plot([initial_rate*2**i for i in range(doublings+1)],dets)
    axs.set_xlabel("Sampling Rate(Unit Time)")
    axs.set_ylabel("Determinant of Sampled Transition Matrix")
    plt.title("Determinant vs the amount of Sampling")
    plt.show()

# Maximum of eigenvalue function
def max_eigenf_value(state_tape, holdTimes_tape,args, doublings=8, initial_rate=1):

    fig,axs = plt.subplots(1,1)
    fig.tight_layout()
    fig.set_size_inches(10,10)
    samp_time = 1/initial_rate


    max_eig_vals = []
    for i in range(doublings+1):
        rate = initial_rate*2**i
        print(f'Using rate {rate}')
        sampled_tape = simple_sample(rate, state_tape, holdTimes_tape)
        p_hat = state_transitions(np.full_like(sampled_tape, 1/rate), sampled_tape)
        eigvals,_ = np.linalg.eig(p_hat)
        print("Eigen Vals for Matrix at rate {} are \n\t{}".format(rate,eigvals))
        a = np.real(eigvals)
        b = np.imag(eigvals)
        pre_s = (a-1)**2  + b**2
        S = np.max(pre_s)
        max_eig_vals.append(S)

    axs.plot([initial_rate*2**i for i in range(doublings+1)],max_eig_vals)
    axs.set_xlabel("Sampling Time")
    axs.set_ylabel(r'$s=(a-1)^2 + b^2$')
    plt.title("S vs Sampling Rate")
    plt.show()


def show_event_driven_mle(state_tape,holdTimes_tape,args):

    fig, axs = plt.subplots(1,1)
    font_size = 8
    fig.tight_layout()
    fig.set_size_inches(10,10)

    print("Calculating the Event Driven Matrix MLE...")
    gen_matrix = event_driven_mle(state_tape,holdTimes_tape)
    print("Calculated. Displaying...")

    axs.imshow(gen_matrix)
    for i in range(gen_matrix.shape[0]):
        for j in range(gen_matrix.shape[1]):
            axs.text(j,i,"%2.2f " % gen_matrix[i,j],ha="center",va="center",color="w",fontsize=font_size)
    axs.set_title(r"Estimated Generator Matrix through event driven transitions. $\mu$:{} $\lambda$:{} ".format(args.mu,args.lam))
    plt.savefig('Images/MLE_event_driven'.format(
        args.samprate,args.lam,args.mu, format='eps',dpi=200))
    plt.show()

def GeneratorFromTransition(holdTimes_tape, state_tape,samp_rate,args):

    # Configure Figures
    fig, ax = plt.subplots(2,2)
    font_size = 8
    fig.tight_layout()
    fig.set_size_inches(10,10)

    # Helper Values
    samp_time = 1/samp_rate
    alt_rate = samp_rate*(2**6)
    # Alt rate is to visualize the behavior under different(usually faster) parameters
    alt_samp_time = 1/alt_rate
    
    # Sample the Tapes
    sampled_tape_ori = simple_sample(samp_rate, state_tape, holdTimes_tape)
    sampled_tape_alt = simple_sample(alt_rate, state_tape, holdTimes_tape)
   
    # Sampled Transition Probabilities
    print(f"Calculating Transition matrix at {samp_rate}")
    trans_matx_samp = state_transitions(np.full_like(sampled_tape_ori,args.samprate), sampled_tape_ori)
    print(f"Calculating Transition matrix at {alt_rate}")
    trans_matx_samp_alt = state_transitions(np.full_like(sampled_tape_alt,args.samprate/2,dtype=np.float16), sampled_tape_alt)
    print(f"Calculating Generator matrix at {samp_rate}")
    Q_ori = (trans_matx_samp/samp_time)-np.eye(trans_matx_samp.shape[0],trans_matx_samp.shape[1])/samp_time # + tineh
    print(f"Calculating Generator matrix at {alt_rate}")
    Q_alt = (trans_matx_samp_alt/alt_samp_time)-np.eye(trans_matx_samp_alt.shape[0],trans_matx_samp_alt.shape[1])/alt_samp_time # + tineh
    
    # Organize Them for the incoming iteration 
    matrices = [trans_matx_samp, Q_ori, trans_matx_samp_alt,Q_alt]
    titles = [f"Transitions from sampled states at sr:{args.samprate}",
              f"Generator Matrix from sampled states at sr:{args.samprate}",
              f"Transitions from sampled states at sr:{alt_rate}",
              f"Generator Matrix from sampled states at sr:{alt_rate}"]

    # Draw the Matrices and their information
    for idx,mat in enumerate(matrices):
        m = idx // 2
        n = int(idx % 2)
        im = ax[m,n].imshow(mat)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax[m,n].text(j,i,"%.2f " % mat[i,j],ha="center",va="center",color="w",fontsize=font_size)
        ax[m,n].set_title(titles[idx])

    fig.suptitle(r'Transition Matrices and Generator Matrices for $\mu=${},$\lambda=${}'
                 .format(args.mu,args.lam))
    plt.savefig('./Images/GenMatrices_r{}_l{}_m{}.png'.format(
        args.samprate,args.lam,args.mu,
        format='eps',dpi=300
        ))
    plt.show()


if __name__ == '__main__':
    # Go through arguments
    args = argparser()

    # Created Tapes
    rates = {"lambda": args.lam,"mu":args.mu} 
    print(f"Working with parameters mu:{args.mu} lambda:{args.lam}")

    # roe = RaceOfExponentials(args.length,rates,state_limit=args.state_limit)
    # holdTimes_tape, state_tape = roe.generate_history(args.init_state)
    
    #  tbd = TrueBirthDeath(args.length,rates)
    #  holdTimes_tape, state_tape = tbd.generate_history(args.init_state)

    roe = EmbeddedMarkC_BD(args.length,rates, state_limit=args.state_limit)
    holdTimes_tape, state_tape = roe.generate_history(args.init_state)

    # For Sanity Check Purposes
    #  if args.show_cont_tmatx:  show_trans_matrx(holdTimes_tape, state_tape)
    if args.show_cont_tmatx and args.state_limit > 0:
        # In case you want to check the determinant
        #  determinant_and_sampling(state_tape, holdTimes_tape, args,initial_rate=args.samprate)
        max_eigenf_value(state_tape, holdTimes_tape, args, initial_rate=args.samprate)

        
        tgm = generate_true_gmatrix({"lam":args.lam, "mu":args.mu}, args.state_limit)
        osm = one_step_matrx(tgm)
        osm = one_step_matrx(tgm)
        stat_dist = get_stat_dist(tgm)
        #  stat_dist = stat_dist.flatten()
        sampled_tape = simple_sample(args.samprate,state_tape,holdTimes_tape)
        emp_state, x_axis = emp_steady_state_distribution(sampled_tape)

        show_sanity_matrxs(
                [tgm,osm, [stat_dist, emp_state]],
                    ["True Generator Matrx",
                    "Corresponding Single-Step matrx",
                     "True Vs Empirical Stationary Distribution"])

    print("Method being used : ",args.method)

    if args.method == 'event_driven_mle':
        show_event_driven_mle(state_tape,holdTimes_tape,args)
    elif args.method == 'log_mat':
        log_matrix_approx(state_tape, holdTimes_tape, args)
    elif args.method == 'fixed_delta_t':
        GeneratorFromTransition(holdTimes_tape, state_tape, args.samprate,args)
