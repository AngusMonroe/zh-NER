import theano
import theano.tensor as T
from utils import shared
import numpy as np
from theano.tensor import _shared


def log_sum_exp(x, axis=None):
    """
    Sum probabilities in the log-space.
    """
    xmax = x.max(axis=axis, keepdims=True)
    xmax_ = x.max(axis=axis)
    return xmax_ + T.log(T.exp(x - xmax).sum(axis=axis))


def forward(observations, transitions, viterbi=True,
            return_alpha=False, return_best_sequence=True):
    """
    Takes as input:
        - observations, sequence of shape (n_steps, n_classes)
        - transitions, sequence of shape (n_classes, n_classes)
    Probabilities must be given in the log space.
    Compute alpha, matrix of size (n_steps, n_classes), such that
    alpha[i, j] represents one of these 2 values:
        - the probability that the real path at node i ends in j
        - the maximum probability of a path finishing in j at node i (Viterbi)
    Returns one of these 2 values:
        - alpha
        - the final probability, which can be:
            - the sum of the probabilities of all paths
            - the probability of the best path (Viterbi)
    """
    assert not return_best_sequence or (viterbi and not return_alpha)
    print ('running')

    states = transitions.shape[1]
    assert observations.shape[1] == transitions.shape[0]
    n_step = observations.shape[0]-1

    def recurrence(obs, steps, previous, transitions):
        # make a column out of a 1d vector
        # transitions: states * states
        # (states*topk) \times 1
        print (states, 'states num')

        cut = states*(steps*2+1)
        previous = previous[:cut]
        transitions = transitions[:cut]

        previous = previous.dimshuffle(0, 'x')
        # make a row out of a 1d vector
        # 1 * states
        obs = obs.dimshuffle('x', 0)
        # 
        if viterbi:
            # each column is a state i
            # previous + obs = states*topk \times states
            scores = previous + obs + transitions
            # return top3 
            # return top3
            # out2 will record the previous node ( size = (1, tag_size) ) 
            
            if return_best_sequence:
                out2 = scores.argsort(axis=0)[::-1][:3].flatten()
                scores = scores.sort(axis=0)
                out = scores[::-1][:3].flatten()
                return out, out2
        else:
            return log_sum_exp(previous + obs + transitions, axis=0)

    obs = theano.tensor.matrix("observations")
    ini = theano.tensor.vector("initial")
    tra = theano.tensor.matrix("transitions")
    index = T.vector('v', 'int64')
    # if return best sequence, it will return alpha[0] and alpha[1], 
    alpha, _ = theano.scan(
        fn=recurrence,
        # to place the second return if return_best_sequence
        # does the parameter correctly passed to the next iteration?
        outputs_info=[T.concatenate([ini, ini, ini]), None] if return_best_sequence else ini,
        sequences=[obs, index],
        non_sequences=T.concatenate([tra, tra, tra], axis = 0)
    )
# alpha[0] size is sen_length*tag
    # alpha[1] size is also sen_length*tag
    # the relevance between viterbi and beam search

    func = theano.function(inputs=[obs, index, ini, tra], outputs=alpha)
    initial = observations[0]
    alpha = func(observations[1:], np.array(list(range(n_step))), initial, transitions)

    if return_best_sequence:
        sequences = []
        topk = 3
        small = -10000
        scores = np.array(alpha[0][-1])
        print ('score', alpha[0])
        print ('back_pointer', alpha[1])
        indices = scores.argsort()[-3:][::-1] 
        print ('indices', indices)
        for indice in indices:
            ind = theano.tensor.scalar("k", 'int64')
            back = theano.tensor.matrix("A", 'int64')
            sequence, _ = theano.scan(
                fn=lambda beta_i, previous: beta_i[previous],
# argmax the final tag corresponding to the largest probability
# add something here
                outputs_info=ind,
                # reverse the sequence length
                sequences=back
            )
            sequence = T.concatenate([sequence[::-1], [ind]])
# [T.argmax(alpha[0][-1])]])
            final_result = theano.function(inputs=[back, ind], outputs=sequence)
# when passing the parameters, no need to apply T.cast
            sequence = final_result(alpha[1][::-1], indice)
            sequences.append(sequence)
        print ('finish')
        print sequences 

if __name__ == '__main__':
    """
        - observations, sequence of shape (n_steps, n_classes)
        - transitions, sequence of shape (n_classes, n_classes)
    """
    # construct observations and transitions
    n_steps = 4 
    n_classes = 4 
    # observations = np.random.rand(n_steps, n_classes)
    # transitions = np.random.rand(n_classes, n_classes)
    observations = np.array([[ 0.59900003,  0.05283449,  0.92982759,  0.21338291],
               [ 0.09610972,  0.06971136,  0.55588373,  0.53080635],
                      [ 0.14908922,  0.89488346,  0.59575705,  0.99684985],
                             [ 0.88628938,  0.21223815,  0.75238096,  0.52368204]])
    transitions = np.array([[ 0.879807  ,  0.23694166,  0.06132776,  0.30704196],
               [ 0.45483884,  0.63702597,  0.75822334,  0.49718952],
                      [ 0.99380664,  0.1713296 ,  0.30975677,  0.53978551],
                             [ 0.34455583,  0.3239001 ,  0.31954717,  0.43296484]])
    print (observations, transitions)
    forward(observations, transitions)
