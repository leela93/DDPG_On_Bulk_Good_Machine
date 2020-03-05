# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 17:08:55 2020

@author: leela
"""
import numpy as np
import torch
from collections import deque
import torch.nn as nn
import torch.nn.functional as F
import random

class MemoryBuffer:

    def __init__(self, size):
                    self.buffer = deque(maxlen=size)
                    self.maxSize = size
                    self.len = 0

    def sample(self, count = 100):
        
        """
        samples a random batch from the replay memory buffer
        :param count: batch size
        :return: batch (numpy array)
        """
        batch = []
        count = min(count, self.len)
        batch = random.sample(self.buffer, count)

        s_arr = np.array([arr[0] for arr in batch])
        d_arr = np.array([arr[1] for arr in batch])
        a_arr = np.array([arr[2] for arr in batch])
        #print('\nbatch in sampling',batch)#np.float32
        r_arr = np.array([arr[3] for arr in batch])
        s1_arr = np.array([arr[4] for arr in batch])
        done_arr = np.array([arr[5] for arr in batch])
        return s_arr, d_arr,a_arr, r_arr, s1_arr, done_arr

    def len(self):
        return self.len

    def add(self, s,d, a, r, s1, done):
        """
        adds a particular transaction in the memory buffer
        :param s: current state
        :param d: desired action
        :param a: action taken
        :param r: reward received
        :param s1: next state
        :return:
        """
        transition = (s,d,a,r,s1,done)
        self.len += 1
        #print("transition_len",self.len)
        if self.len > self.maxSize:
            self.len = self.maxSize
        self.buffer.append(transition)
        
    def clearmemory(self):
        self.buffer.clear()

EPS = 0.003

def fanin_init(size, fanin=None):
	fanin = fanin or size[0]
	v = 1. / np.sqrt(fanin)
	return torch.Tensor(size).uniform_(-v, v)

class Critic(nn.Module):

	def __init__(self, state_dim, action_dim):
		"""
		:param state_dim: Dimension of input state (int)
		:param action_dim: Dimension of input action (int)
		:return:
		"""
		super(Critic, self).__init__()

		self.state_dim = state_dim
		self.action_dim = action_dim

		self.fcs1 = nn.Linear(state_dim,25) #previous:7
		self.fcs1.weight.data = fanin_init(self.fcs1.weight.data.size())
		#self.fcs1_bn = nn.BatchNorm1d(7)  
        
		self.fcs2 = nn.Linear(25,25) #(7,12)
		self.fcs2.weight.data = fanin_init(self.fcs2.weight.data.size())
		#self.fcs2_bn = nn.BatchNorm1d(12)
		#print("action_dim",action_dim)        
		self.fca1 = nn.Linear(action_dim,25) #(12)
		self.fca1.weight.data = fanin_init(self.fca1.weight.data.size())
		#self.fca1_bn = nn.BatchNorm1d(16)
        
		self.fc2 = nn.Linear(50,25) #(24,8)
		self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
		#self.fc2_bn = nn.BatchNorm1d(16)
        
		self.fc3 = nn.Linear(25,1) #(8,1)
		self.fc3.weight.data.uniform_(-EPS,EPS)

	def forward(self, state, action):
		"""
		returns Value function Q(s,a) obtained from critic network
		:param state: Input state (Torch Variable : [n,state_dim] )
		:param action: Input Action (Torch Variable : [n,action_dim] )
		:return: Value function : Q(S,a) (Torch Variable : [n,1] )
		"""
		##print("\nsstate in before fwd",state.size(),"action in before fwd",action.size())
		s1 = F.relu(self.fcs1(state))
		s2 = F.relu(self.fcs2(s1))
		a1 = F.relu(self.fca1(action))
		#print("\ns2 in fwd",s2.size(),"a1 in fwd",a1.size())
		x = torch.cat((s2,a1),dim=1)
		#print('\nx after cat',x.size())
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		#x = (x - min_action)/(max_action - min_action)
		#N = nn.Sigmoid()
		#x = N(x)
		#print('\nQ value', x)
		return x


class Actor(nn.Module):

	def __init__(self, state_dim, action_dim): #, action_lim
		"""
		:param state_dim: Dimension of input state (int)
		:param action_dim: Dimension of output action (int)
		:param action_lim: Used to limit action in [-action_lim,action_lim]
		:return:
		"""
		super(Actor, self).__init__()

		self.state_dim = state_dim
		self.action_dim = action_dim
		#self.action_lim = action_lim

		self.fc1 = nn.Linear(state_dim,25)#(7)
		self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
		#self.fc1_bn = nn.BatchNorm1d(10)
		self.fc2 = nn.Linear(25,50) #(7,12)
		self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
		#self.fc2_bn = nn.BatchNorm1d(16)
		self.fc3 = nn.Linear(50,25) #(12,5)
		self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())
		#self.fc3_bn = nn.BatchNorm1d(6)
		self.fc4 = nn.Linear(25,action_dim) #(5)
		self.fc4.weight.data.uniform_(-EPS,EPS)
		
        
	def forward(self, state):
		"""
		returns policy function Pi(s) obtained from actor network
		this function is a gaussian prob distribution for all actions
		with mean lying in (-1,1) and sigma lying in (0,1)
		The sampled action can , then later be rescaled
		:param state: Input state (Torch Variable : [n,state_dim] )
		:return: Output action (Torch Variable: [n,action_dim] )
		"""
		#x = state
		x1 = F.relu(self.fc1(state))
		x2 = F.relu(self.fc2(x1))
		x3 = F.relu(self.fc3(x2))
		#x = F.relu(self.fc1(state))        
		#x = F.relu(self.fc2(self.fc1_bn(x)))
		#x = F.relu(self.fc3(self.fc2_bn(x)))
		action = F.relu(self.fc4(x3))
		m = nn.Sigmoid()
		action = m(action)
		#action = action * self.action_lim

		return action
