import theano
import theano.tensor as T
from utils import shared
import numpy as np
from theano.tensor import _shared


class HiddenLayer(object):
    """
    Hidden layer with or without bias.
    Input: tensor of dimension (dims*, input_dim)
    Output: tensor of dimension (dims*, output_dim)
    """
    def __init__(self, input_dim, output_dim, bias=True, activation='sigmoid',
                 name='hidden_layer'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias
        self.name = name
        if activation is None:
            self.activation = None
        elif activation == 'tanh':
            self.activation = T.tanh
        elif activation == 'sigmoid':
            self.activation = T.nnet.sigmoid
        elif activation == 'softmax':
            self.activation = T.nnet.softmax
        else:
            raise Exception("Unknown activation function: " % activation)

        # Initialize weights and bias
        self.weights = shared((input_dim, output_dim), name + '__weights')
        self.bias = shared((output_dim,), name + '__bias')

        # Define parameters
        if self.bias:
            self.params = [self.weights, self.bias]
        else:
            self.params = [self.weights]

    def link(self, input):
        """
        The input has to be a tensor with the right
        most dimension equal to input_dim.
        """
        self.input = input
        self.linear_output = T.dot(self.input, self.weights)
        if self.bias:
            self.linear_output = self.linear_output + self.bias
        if self.activation is None:
            self.output = self.linear_output
        else:
            self.output = self.activation(self.linear_output)
        return self.output


class EmbeddingLayer(object):
    """
    Embedding layer: word embeddings representations
    Input: tensor of dimension (dim*) with values in range(0, input_dim)
    Output: tensor of dimension (dim*, output_dim)
    """

    def __init__(self, input_dim, output_dim, name='embedding_layer'):
        """
        Typically, input_dim is the vocabulary size,
        and output_dim the embedding dimension.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name = name

        # Randomly generate weights
        self.embeddings = shared((input_dim, output_dim),
                                 self.name + '__embeddings')

        # Define parameters
        self.params = [self.embeddings]

    def link(self, input):
        """
        Return the embeddings of the given indexes.
        Input: tensor of shape (dim*)
        Output: tensor of shape (dim*, output_dim)
        """
        self.input = input
        self.output = self.embeddings[self.input]
        return self.output


class DropoutLayer(object):
    """
    Dropout layer. Randomly set to 0 values of the input
    with probability p.
    """
    def __init__(self, p=0.5, name='dropout_layer'):
        """
        p has to be between 0 and 1 (1 excluded).
        p is the probability of dropping out a unit, so
        setting p to 0 is equivalent to have an identity layer.
        """
        assert 0. <= p < 1.
        self.p = p
        self.rng = T.shared_randomstreams.RandomStreams(seed=123456)
        self.name = name

    def link(self, input):
        """
        Dropout link: we just apply mask to the input.
        """
        if self.p > 0:
            mask = self.rng.binomial(n=1, p=1-self.p, size=input.shape,
                                     dtype=theano.config.floatX)
            self.output = input * mask
        else:
            self.output = input

        return self.output


class LSTM(object):
    """
    Long short-term memory (LSTM). Can be used with or without batches.
    Without batches:
        Input: matrix of dimension (sequence_length, input_dim)
        Output: vector of dimension (output_dim)
    With batches:
        Input: tensor3 of dimension (batch_size, sequence_length, input_dim)
        Output: matrix of dimension (batch_size, output_dim)
    """
    def __init__(self, input_dim, hidden_dim, with_batch=True, name='LSTM'):
        """
        Initialize neural network.
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.with_batch = with_batch
        self.name = name

        # Input gate weights
        self.w_xi = shared((input_dim, hidden_dim), name + '__w_xi')
        self.w_hi = shared((hidden_dim, hidden_dim), name + '__w_hi')
        self.w_ci = shared((hidden_dim, hidden_dim), name + '__w_ci')

        # Forget gate weights
        # self.w_xf = shared((input_dim, hidden_dim), name + '__w_xf')
        # self.w_hf = shared((hidden_dim, hidden_dim), name + '__w_hf')
        # self.w_cf = shared((hidden_dim, hidden_dim), name + '__w_cf')

        # Output gate weights
        self.w_xo = shared((input_dim, hidden_dim), name + '__w_xo')
        self.w_ho = shared((hidden_dim, hidden_dim), name + '__w_ho')
        self.w_co = shared((hidden_dim, hidden_dim), name + '__w_co')

        # Cell weights
        self.w_xc = shared((input_dim, hidden_dim), name + '__w_xc')
        self.w_hc = shared((hidden_dim, hidden_dim), name + '__w_hc')

        # Initialize the bias vectors, c_0 and h_0 to zero vectors
        self.b_i = shared((hidden_dim,), name + '__b_i')
        # self.b_f = shared((hidden_dim,), name + '__b_f')
        self.b_c = shared((hidden_dim,), name + '__b_c')
        self.b_o = shared((hidden_dim,), name + '__b_o')
        self.c_0 = shared((hidden_dim,), name + '__c_0')
        self.h_0 = shared((hidden_dim,), name + '__h_0')

        # Define parameters
        self.params = [self.w_xi, self.w_hi, self.w_ci,
                       # self.w_xf, self.w_hf, self.w_cf,
                       self.w_xo, self.w_ho, self.w_co,
                       self.w_xc, self.w_hc,
                       self.b_i, self.b_c, self.b_o,  # self.b_f,
                       self.c_0, self.h_0]

    def link(self, input):
        """
        Propagate the input through the network and return the last hidden
        vector. The whole sequence is also accessible via self.h, but
        where self.h of shape (sequence_length, batch_size, output_dim)
        """
        def recurrence(x_t, c_tm1, h_tm1):
            i_t = T.nnet.sigmoid(T.dot(x_t, self.w_xi) +
                                 T.dot(h_tm1, self.w_hi) +
                                 T.dot(c_tm1, self.w_ci) +
                                 self.b_i)
            # f_t = T.nnet.sigmoid(T.dot(x_t, self.w_xf) +
            #                      T.dot(h_tm1, self.w_hf) +
            #                      T.dot(c_tm1, self.w_cf) +
            #                      self.b_f)
            c_t = ((1 - i_t) * c_tm1 + i_t * T.tanh(T.dot(x_t, self.w_xc) +
                   T.dot(h_tm1, self.w_hc) + self.b_c))
            o_t = T.nnet.sigmoid(T.dot(x_t, self.w_xo) +
                                 T.dot(h_tm1, self.w_ho) +
                                 T.dot(c_t, self.w_co) +
                                 self.b_o)
            h_t = o_t * T.tanh(c_t)
            return [c_t, h_t]

        # If we use batches, we have to permute the first and second dimension.
        if self.with_batch:
            self.input = input.dimshuffle(1, 0, 2)
            outputs_info = [T.alloc(x, self.input.shape[1], self.hidden_dim)
                            for x in [self.c_0, self.h_0]]
        else:
            self.input = input
            outputs_info = [self.c_0, self.h_0]

        [_, h], _ = theano.scan(
            fn=recurrence,
            sequences=self.input,
            outputs_info=outputs_info,
            n_steps=self.input.shape[0]
        )
        self.h = h
        self.output = h[-1]

        return self.output


def log_sum_exp(x, axis=None):
    """
    Sum probabilities in the log-space.
    """
    xmax = x.max(axis=axis, keepdims=True)
    xmax_ = x.max(axis=axis)
    return xmax_ + T.log(T.exp(x - xmax).sum(axis=axis))


def forward(observations, transitions, viterbi=False,
            return_alpha=False, return_best_sequence=False):
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
#    assert observations.shape[1] == transitions.shape[0]

    def recurrence(obs, steps, previous, transitions):
        # make a column out of a 1d vector
        # transitions: states * states
        # (states*topk) \times 1
        print (states, 'states num')

        # debug for 1.5 hour...
        # cannot get the current value and get branches
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

    initial = observations[0]
    # if return best sequence, it will return alpha[0] and alpha[1], 
    alpha, _ = theano.scan(
        fn=recurrence,
        # to place the second return if return_best_sequence
        outputs_info=(T.concatenate([initial, initial, initial]), None) if return_best_sequence else initial,
        sequences=[observations[1:], T.arange(observations.shape[0]-1)],
        non_sequences=T.concatenate([transitions, transitions, transitions], axis = 0)
    )
# alpha[0] size is sen_length*tag
    # alpha[1] size is also sen_length*tag
    # the relevance between viterbi and beam search

    if return_alpha:
        print ('1')
        return alpha
# return best sequence
    elif return_best_sequence:
        sequences = []
        print ('this branch')
        topk = 3
        small = -10000
        scores = alpha[0][-1]
        # the only explanation is that it has only one score...
        #indices = scores.argsort()[-3:] # [::-1] 
        indices = scores.argsort()[::-1][:3].flatten()
        scores = scores.sort()
        top_3_scores = scores[::-1][:3].flatten()
        # for indice in indices:_
        # cannot cast to numpy array and perform
        # cannot iterate the theano variable
# hard code
        indices_auto = [states-1, states*2-1, states*3-1]
        for indice in indices_auto:
            sequence, _ = theano.scan(
                fn=lambda beta_i, previous: beta_i[previous],
# argmax the final tag corresponding to the largest probability
# add something here
                outputs_info=indice,
                # reverse the sequence length
                sequences=T.cast(alpha[1][::-1], 'int64')
            )
            sequence = T.concatenate([sequence[::-1], [indice]])
            sequences.append(sequence)
        sequences.append(top_3_scores)
        return sequences
    else:
        if viterbi:
            return alpha[-1].max(axis=0)
        else:
            return log_sum_exp(alpha[-1], axis=0)
