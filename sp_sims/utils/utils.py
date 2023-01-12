import matplotlib.pyplot as plt
import numpy as np
import argparse
from  ..statistics.statistics import trans_matrix

# Extremely ad-hoc 
def show_sanity_matrxs(matrices, titles):
    # Show Transition Matrices
    fig, ax = plt.subplots(1,len(matrices))
    fig.subplots_adjust(top=1.30)
    fig.suptitle(titles[-1])
    labels = ['True','Empirical']
    for m,data in enumerate(matrices):
        ax[m].title.set_text(titles[m])
        if hasattr(data, "shape") and len(data.shape) == 2:
            ax[m].imshow(data)
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    ax[m].text(j,i,"%.2f " % data[i,j],ha="center",va="center",color="w",fontsize=6)
        else:# Assume its 1d data
            for i,bar in enumerate(data):
                ax[m].bar(np.arange(0,len(bar),dtype=float)+(0.40*i),bar,0.40,label=labels[i])
            ax[m].legend()
            state_dist = ax[m].set_title(titles[m])

    plt.show()
    plt.close()

def show_trans_matrx(holdTimes_tape,state_tape):
    # Show Transition Matrices
    fig, ax = plt.subplots(1,1)
    trans = trans_matrix(holdTimes_tape,state_tape)
    plt.imshow(trans)
    for i in range(trans.shape[0]):
        for j in range(trans.shape[1]):
            ax.text(j,i,"%.2f " % trans[i,j],ha="center",va="center",color="w",fontsize=4)
    ax.set_title("Preliminary Transition transrix")
    plt.show()
    plt.close()

def argparser():
    parser  = argparse.ArgumentParser()
    parser.add_argument('--length',
                        dest='length',
                        default=10000,
                        type=int,
                        help='Length of episode in discrete realizations.')
    parser.add_argument('--mu',
                        dest='mu',
                        default = .15,
                        type=float,
                        help='Service Rate')
    parser.add_argument('--lambda',
                        dest='lam',
                        default=.10,
                        type=float,
                        help='Birth Rate')
    parser.add_argument('--samprate',
                        dest='samprate',
                        default=1.0,
                        type=float,
                        help='Rate at which we sample real line.')
    parser.add_argument('--state_limit',
                        dest='state_limit',
                        default=-1,
                        type=int,
                        help='Make the State Space Limited in Positive Integer Space by Providing Max Value.')
    parser.add_argument('--detection_guesses',
                        dest='detection_guesses',
                        default=-100,
                        type=int,
                        help='How many guesses we will take for detection')
    parser.add_argument('--init_state',
                        dest='init_state',
                        type=int,
                        default = 0,
                        help='Initial State in the real line.(Amnt of current events)')
    parser.add_argument('--show_sanity_check',
                        dest='show_cont_tmatx',
                        type=bool,
                        default = False,
                        help='Shows the transition matrix estimated from  continuous labels.')
    parser.add_argument('--method',
                        dest='method',
                        choices=['event_driven_mle','log_mat','fixed_delta_t'],
                        default = 'fixed_sampled_rate',
                        help='Initial State in the real line.(Amnt of current events)')
    return parser.parse_args()
