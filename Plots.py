#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 11:41:14 2019

@author: at-lab
"""

import torch
import matplotlib.pyplot as plt 

Energy_1 = torch.load("Energy_Consumed_Per_Episode.pt")
#actor_1 = torch.load("Reward_Per_Episode_1.pt",)
#actor_2 = torch.load("loss_mse_1.pt",)
#actor_3 = torch.load("loss_mse_5.pt",)

N_1 = 1
cumsum_1, moving_aves_1 = [0],[]

for i, x in enumerate(Energy_1,1):
      cumsum_1.append(cumsum_1[i-1]+x)
      if i > N_1:
          moving_ave_1 = (cumsum_1[i]-cumsum_1[i-N_1])/N_1
          moving_aves_1.append(moving_ave_1)
#        
# N_1 = 10
# cumsum_1, moving_aves_1 = [0],[]

# for i, x in enumerate(actor_1):
#     cumsum_1.append(cumsum_1[i-1]+x)
#     if i > N_1:
#         moving_ave_1 = (cumsum_1[i]-cumsum_1[i-N_1])/N_1
#         moving_aves_1.append(moving_ave_1)        
# ## # #        
# N_2 = 100
# cumsum_2, moving_aves_2 = [0],[]
# for i, x in enumerate(actor_2,1):
#     cumsum_2.append(cumsum_2[i-1]+x)
#     if i > N_2:
#         moving_ave_2 = (cumsum_2[i]-cumsum_2[i-N_2])/N_2
#         moving_aves_2.append(moving_ave_2)
        
# N_3 = 100
# cumsum_3, moving_aves_3 = [0],[]
# for i, x in enumerate(actor_3,1):
#     cumsum_3.append(cumsum_3[i-1]+x)
#     if i > N_3:
#         moving_ave_3 = (cumsum_3[i]-cumsum_3[i-N_3])/N_3
#         moving_aves_3.append(moving_ave_3)



plt.figure(figsize=(16,8))
plt.plot(moving_aves_1,"r",label = 'Energy_Steps:700,Episodes:500')
#plt.plot(moving_aves_1,"r",label = 'reward_5, Steps-700, Episodes-1000')
#plt.bar(moving_aves_2,"b",label = 'mse_1,Steps:1000,Episodes:1000')
#plt.plot(moving_aves_3,"m",label = 'mse_5,Steps:700,Episodes:500')


#plt.legend()
# plt.title("hopper_3")
# plt.ylabel('Height')
# plt.xlabel('Episodes')
#plt.savefig("loss_mse_1_700")