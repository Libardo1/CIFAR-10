__author__ = 'Dipendra'
import numpy as np
import csv, sys, copy
import gzip, cPickle, numpy
from numpy import genfromtxt
import random
import theano
import theano.tensor as T
import os

# to unpickle from data file
def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

#Read data files and merge into one
fls = os.listdir('data')
train_data = {}
for f in fls:
	if 'data' in f:
		#print f
		d = unpickle('data/'+f)
		#print type(d['data'])
		#print d['data']
		d['data']=np.true_divide(d['data'],255.0)
		#print d['data']
		if not train_data.has_key('data'):
			train_data['data'] = d['data']
			train_data['labels'] = d['labels']
		else:
			train_data['data'] = numpy.concatenate((train_data['data'],d['data']), axis=0)
			train_data['labels'] = numpy.concatenate((train_data['labels'], d['labels']), axis=0)
print train_data['data'].shape[0], train_data['data'].shape[1]

test_data = unpickle('data/test_batch')
print test_data['data'].shape[0], test_data['data'].shape[1]

test_data['data'] = np.true_divide(test_data['data'],255.0)
num_rows = train_data['data'].shape[0] # Number of data samples

data = np.array(train_data['data'],dtype=float)
label = np.array(train_data['labels'],dtype=float)


print data.shape
print label.shape

train_num = int(num_rows * 0.8)

DataSetState = "This dataset has " + repr(data.shape[0]) + " samples of length " + repr(data.shape[1]) + ". The number of training examples is " + repr(train_num)
print DataSetState


train_set_x = data[:train_num]
train_set_y = label[:train_num]

val_set_x = data[train_num:]
val_set_y = label[train_num:]

test_set_x = np.array(test_data['data'],float)
test_set_y = np.array(test_data['labels'], float)


# Divided dataset into 3 parts. split by percentage.

train_set = train_set_x, train_set_y
val_set = val_set_x, val_set_y
test_set = test_set_x, test_set_y

print train_set_x.shape, train_set_y.shape
print val_set_x.shape, val_set_y.shape
print test_set_x.shape, test_set_y.shape

dataset = [train_set, val_set, test_set]

print 'creating tar file'

#sys.exit(0)
f = gzip.open('CIFAR10'+'.pkl.gz','wb')
cPickle.dump(dataset, f, protocol=2)
f.close()
