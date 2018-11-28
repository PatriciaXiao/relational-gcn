from __future__ import print_function

from keras.layers import Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

'''
from rgcn.layers.graph import GraphConvolution
from rgcn.layers.input_adj import InputAdj
from rgcn.utils import *
'''
from layers.graph import GraphConvolution
from layers.input_adj import InputAdj
from utils import *

import pickle as pkl

import os
import sys
import time
import argparse

import keras.backend as K

import theano.tensor as T
def myloss(output, target):
    '''
        categorical_crossentropy, customize function
        limitation: theano functions only, no tensorflow
    '''
    return T.nnet.categorical_crossentropy(target, output)

np.random.seed()

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, default="aifb",
                help="Dataset string ('aifb', 'mutag', 'bgs', 'am')")
ap.add_argument("-e", "--epochs", type=int, default=50,
                help="Number training epochs")
ap.add_argument("-hd", "--hidden", type=int, default=16,
                help="Number hidden units")
ap.add_argument("-do", "--dropout", type=float, default=0.,
                help="Dropout rate")
ap.add_argument("-b", "--bases", type=int, default=-1,
                help="Number of bases used (-1: all)")
ap.add_argument("-lr", "--learnrate", type=float, default=0.01,
                help="Learning rate")
ap.add_argument("-l2", "--l2norm", type=float, default=0.,
                help="L2 normalization of input weights")

fp = ap.add_mutually_exclusive_group(required=False)
fp.add_argument('--validation', dest='validation', action='store_true')
fp.add_argument('--testing', dest='validation', action='store_false')
ap.set_defaults(validation=True)

args = vars(ap.parse_args())
print(args)

# Define parameters
DATASET = args['dataset']
NB_EPOCH = args['epochs']
VALIDATION = args['validation']
LR = args['learnrate']
L2 = args['l2norm']
HIDDEN = args['hidden']
BASES = args['bases'] # bases defined
DO = args['dropout']

DEBUG_DATA = False

dirname = os.path.dirname(os.path.realpath(sys.argv[0]))

with open(dirname + '/' + DATASET + '.pickle', 'rb') as f:
    data = pkl.load(f, encoding='iso-8859-1') # for python 3
    # data = pkl.load(f)

A = data['A']
y = data['y']
train_idx = data['train_idx']
test_idx = data['test_idx']


if DEBUG_DATA:
    print("Relations used and their frequencies " + str([a.sum() for a in A]))
    print("First relation shape in A: {0}".format(A[0].shape))
    print("First entry of first relation in A: {0}".format(A[0][0, 0]))
    print("Another entry of first relation in A: {0}".format(A[0][10, 1]))
    print("This is y {0}".format(y)) # {((row, col), val)} # row: item; col: category; value: 1 for yes
    # print("This is what X supposed to be {0}".format(sp.identity(A[0].shape[0], format='csr'))) # Identity matrix
    print("test_idx: {0}".format(test_idx))
    print("this is y: {0}".format(y.toarray()))
    print("this is the amount of 1s in y: {0}".format(int(y.sum())))
    print("y shape: ({0}, {1})".format(len(y.toarray()), len(y.toarray()[1]) ))
    print("train index unique set? ", len(train_idx) == len(set(train_idx)), len(train_idx) ) # same as the train tsv
    print("test index unique set? ", len(test_idx) == len(set(test_idx)), len(test_idx) ) # same as the test tsv
    exit(0)

# Get dataset splits
y_train, y_val, y_test, idx_train, idx_val, idx_test = get_splits(y, train_idx,
                                                                  test_idx,
                                                                  VALIDATION)
train_mask = sample_mask(idx_train, y.shape[0])

num_nodes = A[0].shape[0]
support = len(A)

# Define empty dummy feature matrix (input is ignored as we set featureless=True)
# In case features are available, define them here and set featureless=False.
X = sp.csr_matrix(A[0].shape)

# Normalize adjacency matrices individually
for i in range(len(A)):
    d = np.array(A[i].sum(1)).flatten()
    d_inv = 1. / (d + 1e-5)
    d_inv[np.isinf(d_inv)] = 0.
    D_inv = sp.diags(d_inv)
    A[i] = D_inv.dot(A[i]).tocsr()


A_in = [InputAdj(sparse=True) for _ in range(support)]
X_in = Input(shape=(X.shape[1],), sparse=True)

# Define model architecture
H = GraphConvolution(HIDDEN, support, num_bases=BASES, featureless=True,
                     activation='relu',
                     W_regularizer=l2(L2))([X_in] + A_in)

H = Dropout(DO)(H)
# print ("A_in.shape=({0},)".format(len(A_in),)) # (47,)
# print ("H.shape={0}".format(H.shape)) # (23644, 16)
# print ("support={0}".format(support)) # 47
Y = GraphConvolution(y_train.shape[1], support, num_bases=BASES,
                     activation='softmax')([H] + A_in)

# Compile model
model = Model(input=[X_in] + A_in, output=Y)
model.compile(loss=myloss, optimizer=Adam(lr=LR))

preds = None

# Fit
for epoch in range(1, NB_EPOCH + 1):

    # Log wall-clock time
    t = time.time()

    # Single training iteration
    model.fit([X] + A, y_train, sample_weight=train_mask,
              batch_size=num_nodes, nb_epoch=1, shuffle=False, verbose=0)

    if epoch % 1 == 0:

        # Predict on full dataset
        preds = model.predict([X] + A, batch_size=num_nodes)

        # Train / validation scores
        train_val_loss, train_val_acc = evaluate_preds(preds, [y_train, y_val],
                                                       [idx_train, idx_val])

        print("Epoch: {:04d}".format(epoch),
              "train_loss= {:.4f}".format(train_val_loss[0]),
              "train_acc= {:.4f}".format(train_val_acc[0]),
              "val_loss= {:.4f}".format(train_val_loss[1]),
              "val_acc= {:.4f}".format(train_val_acc[1]),
              "time= {:.4f}".format(time.time() - t))

    else:
        print("Epoch: {:04d}".format(epoch),
              "time= {:.4f}".format(time.time() - t))

# Testing
test_loss, test_acc = evaluate_preds(preds, [y_test], [idx_test])
print("Test set results:",
      "loss= {:.4f}".format(test_loss[0]),
      "accuracy= {:.4f}".format(test_acc[0]))
