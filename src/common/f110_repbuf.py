#!/usr/bin/env python
from __future__ import print_function
from collections import deque
import random, os, sys

__author__ = 'dhruv karthik <dhruvkar@seas.upenn.edu>'

class f110_ReplayBuffer(object):
    """
    Generic Replay Buff implementation. Stores experiences from the F110 & returns sample batches
    """
    def __init__(self, maxsize=500000, batch_size=20):
        super(f110_ReplayBuffer, self).__init__()
        self.maxsize, self.bs = maxsize, batch_size
        self.buffer = deque(maxlen=maxsize)
        self.count = 0 #keep track of elements
    
    def add(self, obs_dict, action, reward, done):
        """
        Add an experience to Replay Buffer
        """
        self.buffer.append((obs_dict, action, reward, done))
        self.count = min(self.maxsize, self.count+1)
        print('||| BUFSIZE:', self.count)

    def sample(self):
        """
        Uniformly samples the buffer for 'batch_size' experiences & returns them
        """
        if self.count <= self.bs:
            raise Exception('Not Enough Elements to Sample batch')
        else:
	    ob, ac, re, do = zip(*random.sample(self.buffer, self.bs))
	    obs_batch, action_batch, reward_batch, done_batch  = map(lambda x: list(x), [ob, ac, re, do])
	    return obs_batch, action_batch, reward_batch, done_batch
