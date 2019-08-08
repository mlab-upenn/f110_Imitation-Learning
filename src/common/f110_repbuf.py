#!/usr/bin/env python
from __future__ import print_function
from collections import deque
import numpy as np
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

class f110_PrioritizedReplayBuffer(object):
    """
    Prioritized Replay Buff implementation. Stores experiences from the F110 & returns sample batches
    """
    def __init__(self, maxsize=500, batch_size=20):
        super(f110_PrioritizedReplayBuffer, self).__init__()
        self.maxsize, self.bs = maxsize, batch_size
        self.buffer = deque(maxlen=maxsize)
        self.priorities = []
        self.count = 0 #keep track of elements
    
    def add(self, obs_dict, action, reward, done, priority):
        """ Add an experience to Replay Buffer
        """
        self.buffer.append((obs_dict, action, reward, done))
        self.count = min(self.maxsize, self.count+1)
        self.priorities.append(priority)
        print('||| BUFSIZE:', self.count)

    def getprobs(self):
        unorm_probs = np.array(self.priorities)
        norm_probs = unorm_probs/unorm_probs.sum()
        return norm_probs
        
    def batch_from_idxs(self, idxs):
        obs_batch, action_batch, reward_batch, done_batch = [],[],[],[] 
        for idx in idxs:
            e = self.buffer[idx]
            obs_batch.append(e[0])
            action_batch.append(e[1])
            reward_batch.append(e[2])
            done_batch.append(e[3])
        return obs_batch, action_batch, reward_batch, done_batch

    def sample(self):
        """ Uniformly samples the buffer for 'batch_size' experiences & returns them
        """
        if self.count <= self.bs:
            raise Exception('Not Enough Elements to Sample batch')
        else:
            idxs = np.random.choice(len(self.buffer), self.bs, p=self.getprobs())
            obs_batch, action_batch, reward_batch, done_batch = self.batch_from_idxs(idxs)
            return obs_batch, action_batch, reward_batch, done_batch
