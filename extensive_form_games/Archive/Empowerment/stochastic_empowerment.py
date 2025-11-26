import numpy as np 
from functools import reduce
import itertools
import random
from extensive_form_games.Empowerment.information_theory import blahut_arimoto

def compute_empowerment(T, det, n_step, state, n_samples=1000, epsilon = 1e-6):
    """
    Compute the empowerment of a state,
    T : numpy array, shape (n_states, n_actions, n_states)
        Transition matrix describing the probabilistic dynamics of a markov decision process
        (without rewards). Taking action a in state s, T describes a probability distribution
        over the resulting state as T[:,a,s]. In other words, T[s',a,s] is the probability of
        landing in state s' after taking action a in state s. The indices may seem "backwards"
        because this allows for convenient matrix multiplication.  
        - I want to store the transition probabalities indexed by [ai,si]. [ai,si] -> [s1, .4], [s2, .3], [s4, .2]
                                                                           [aj, si]->  [s1, .7], [s2, .2], [s4, .5]   
                                                                            [ai, sj]       
    det : bool
        True if the dynamics are deterministic.
    n_step : int 
        Determines the "time horizon" of the empowerment computation. The computed empowerment is
        the influence the agent has on the future over an n_step time horizon. 
    n_samples : int
        Number of samples for approximating the empowerment in the deterministic case.
    state : int 
        State for which to compute the empowerment.
    """
    n_states, n_actions, _  = T.shape
    if det:
        # only sample if too many actions sequences to iterate through
        if n_actions**n_step < 5000:
            nstep_samples = np.array(list(itertools.product(range(n_actions), repeat = n_step)))
        else:
            nstep_samples = np.random.randint(0,n_actions, [n_samples,n_step] )
        # fold over each nstep actions, get unique end states
        tmap = lambda s, a : np.argmax(T[:,a,s]) 
        seen = set()
        for i in range(len(nstep_samples)):
            aseq = nstep_samples[i,:]
            seen.add(reduce(tmap, [state,*aseq]))
        # empowerment = log # of reachable states 
        return np.log2(len(seen))
    else:
        nstep_actions = list(itertools.product(range(n_actions), repeat = n_step))
        Bn = np.zeros([n_states, len(nstep_actions), n_states])
        for i, an in enumerate(nstep_actions):
            Bn[:, i , :] = reduce((lambda x, y : np.dot(y, x)), map((lambda a : T[:,a,:]), an))
        return blahut_arimoto(Bn[:,:,state], epsilon=epsilon)

def rand_sample(p_x):
    """ 
    Randomly sample a value from a probability distribution p_x
    """
    cumsum = np.cumsum(p_x)
    rnd = np.random.rand()
    return np.argmax(cumsum > rnd)

def normalize(X):
    """ 
    Normalize vector or matrix columns X
    """
    return X / X.sum(axis=0)

def softmax(x, tau):
    """
    Returns the softmax normalization of a vector x using temperature tau.
    """
    return normalize(np.exp(x / tau)) 


