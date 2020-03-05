

################# libraries #################
from __future__ import division
import torch
import gc
import numpy as np
from random import randrange
import datetime
import torch.nn as nn
import shutil
from Environment import bulk_good
from plc import plc
from actor_critic_memory import MemoryBuffer, Actor, Critic
import matplotlib.pyplot as plt
#import time
    
############ Train ###########

GAMMA = 0.99
TAU = 0.001

#Loss_Critic = []
#Loss_Actor = []

class Trainer:

	def __init__(self, state_dim, action_dim, ram,batch_size): #, self.action_lim
		"""
		:param state_dim: Dimensions of state (int)
		:param action_dim: Dimension of action (int)
		:param action_lim: Used to limit action in [-action_lim,action_lim]
		:param ram: replay memory buffer object
		:return:
		"""
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.batch_size = batch_size
		self.ram = ram
		self.iter = 0
		self.noise = OrnsteinUhlenbeckActionNoise(self.action_dim)
		self.learning_actor = 0.0001
		self.learning_critic = 0.001
		self.mse = nn.MSELoss()
		self.l_loss = nn.SmoothL1Loss()
		self.lambda_mse = 0.5

		self.actor = Actor(self.state_dim, self.action_dim) # , self.action_lim
		self.target_actor = Actor(self.state_dim, self.action_dim) #, self.action_lim
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),self.learning_actor)

		self.critic = Critic(self.state_dim, self.action_dim)
		self.target_critic = Critic(self.state_dim, self.action_dim)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),self.learning_critic)

		hard_update(self.target_actor, self.actor)
		hard_update(self.target_critic, self.critic)
		self.loss_l1_list = []
		self.loss_mse_list = []
		self.loss_final_list = []
		self.mse_logic = []
		self.actor_logic = []
        
	def get_exploitation_action(self, state):
		"""
		gets the action from target actor added with exploration noise
		:param state: state (Numpy array)
		:return: sampled action (Numpy array)
		"""
		
		action = self.target_actor.forward(state).detach()
		#print('\nExploitation action',action)
		return action.data.numpy()

		        
		        
	def get_exploration_action(self, state):
		"""
		gets the action from actor added with exploration noise
		:param state: state (Numpy array)
		:return: sampled action (Numpy array)
		"""
		#state = [state_0, state_1, state_2, state_3, state_4]
		action = self.actor.forward(state).detach()
		#print(action)        
		new_action = action.data.numpy() + (self.noise.sample()) # * self.action_lim)  
		#print('\nExploration action', new_action)        
		return new_action
    

	def optimize(self):
		"""
		Samples a random batch from replay memory and performs optimization
		:return:
		"""
		
        
		s1,d1,a1,r1,s2,done = self.ram.sample(self.batch_size)
		s1 = torch.from_numpy(s1).float() 
		d1 = torch.from_numpy(d1).float()
		d1 = d1.unsqueeze(1)
		d1.requires_grad = True
		d1_copy = d1.detach().cpu().numpy()
		self.mse_logic.append(d1_copy[-1][-1])        
		a1 = torch.from_numpy(a1).float()       
		r1 = torch.from_numpy(r1).float()
		r1 = torch.squeeze(r1)
		r1.requires_grad = True
		s2 = torch.from_numpy(s2).float()
		done = torch.from_numpy(done).float()

		# ---------------------- optimize critic ----------------------
		# Use target actor exploitation policy here for loss evaluation    
		a2 = self.target_actor.forward(s2).detach()
		next_val = torch.squeeze(self.target_critic.forward(s2, a2).detach())
		y_expected = r1 + (GAMMA*next_val*(1-done))
		y_predicted = torch.squeeze(self.critic.forward(s1, a1))     
		# compute critic loss, and update the critic
		loss_critic = self.l_loss(y_predicted, y_expected.detach()).unsqueeze(0)
		self.critic_optimizer.zero_grad()
		loss_critic.backward()
		self.critic_optimizer.step()

		# ---------------------- optimize actor ----------------------
		pred_a1 = self.actor.forward(s1)
		pred_a1_copy = pred_a1.detach().cpu().numpy()
		self.actor_logic.append(pred_a1_copy[-1][-1])
		loss_actor =-(torch.sum(self.critic.forward(s1, pred_a1))/self.batch_size)
		entropy = torch.mean((0.001*pred_a1*torch.log(pred_a1)))
		loss_policy = loss_actor-entropy
		self.loss_l1_list.append(loss_policy.item())
		loss_mse = self.mse(pred_a1, d1)
		self.loss_mse_list.append(loss_mse.item())
		loss = sum([(1-self.lambda_mse)*loss_policy, self.lambda_mse*loss_mse])
		self.loss_final_list.append(loss.item())            
		self.actor_optimizer.zero_grad()
		loss.backward()
		self.actor_optimizer.step()
        
		soft_update(self.target_actor, self.actor, TAU)
		soft_update(self.target_critic, self.critic, TAU)
	def save_models(self, episode_count, path_target, path_critic):
		"""
		saves the target actor and critic models
		:param episode_count: the count of episodes iterated
		:return:
		"""
		        
		torch.save(self.target_actor.state_dict(),  path_target )#, 'C:\Users\hp\Desktop\Python DDPG') #'./Models/' + str(episode_count) + '_actor.pt'
		torch.save(self.target_critic.state_dict(), path_critic)#, 'C:\Users\hp\Desktop\Python DDPG') #'./Models/' + str(episode_count) + '_critic.pt'
		#print ('Models saved successfully')
		#retrn
	def load_models(self, episode):
		"""
		loads the target actor and critic models, and copies them onto actor and critic models
		:param episode: the count of episodes iterated (used to find the file name)
		:return:
		"""
		self.actor.load_state_dict(torch.load('./Models/' + str(episode) + '_actor.pt'))
		self.critic.load_state_dict(torch.load('./Models/' + str(episode) + '_critic.pt'))
		hard_update(self.target_actor, self.actor)
		hard_update(self.target_critic, self.critic)
		#print('Models loaded succesfully')
        
        


        
################# Utils #################



def soft_update(target, source, tau):
	"""
	Copies the parameters from source network (x) to target network (y) using the below update
	y = TAU*x + (1 - TAU)*y
	:param target: Target network (PyTorch)
	:param source: Source network (PyTorch)
	:return:
	"""
	for target_param, param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(
			target_param.data * (1.0 - tau) + param.data * tau
		)


def hard_update(target, source):
	"""
	Copies the parameters from source network to target network
	:param target: Target network (PyTorch)
	:param source: Source network (PyTorch)
	:return:
	"""
	for target_param, param in zip(target.parameters(), source.parameters()):
			target_param.data.copy_(param.data)


def save_training_checkpoint(state, is_best, episode_count):
	"""
	Saves the models, with all training parameters intact
	:param state:
	:param is_best:
	:param filename:
	:return:
	"""
	filename = str(episode_count) + 'checkpoint.path.rar'
	torch.save(state, filename)
	if is_best:
		shutil.copyfile(filename, 'model_best.pth.tar')


# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:

	def __init__(self, action_dim, mu = 0, theta = 0.15, sigma = 0.2):
		self.action_dim = action_dim
		self.mu = mu
		self.theta = theta
		self.sigma = sigma
		self.X = np.ones(self.action_dim) * self.mu

	def reset(self):
		self.X = np.ones(self.action_dim) * self.mu

	def sample(self):
		dx = self.theta * (self.mu - self.X)
		dx = dx + self.sigma * np.random.randn(len(self.X))
		self.X = self.X + dx
		return self.X


# use this to plot Ornstein Uhlenbeck random motion
if __name__ == '__main__':
	ou = OrnsteinUhlenbeckActionNoise(1)
	states = []
	for i in range(1000):
		states.append(ou.sample())


	#plt.plot(states)
	#plt.show()
        

#####################################################################################################################

############ Main #############


now = datetime.datetime.now()

#print
print("Simulation Started at:")
print(str(now))

MAX_EPISODES = 500# Madhav kept it as 2500 states for best result
MAX_STEPS = 1000 # 550
MAX_BUFFER = 100000
batch_size = 512
steps = 0
_ep = 0
Energy_Consumed_Per_Episode = []
Episode_counter = []
Terminal_reached = []
Reward_Per_Episode_1 = []
Reward_Per_Episode_2 = []
Reward_Per_Episode_3 = []
Reward_Per_Episode_4 = []
Reward_Per_Episode_5 = []
Reward_Per_Episode_6 = []

logic_m_1 = []
logic_a_1 = []
logic_m_2 = []
logic_a_2 = []
logic_m_3 = []
logic_a_3 = []
logic_m_4 = []
logic_a_4 = []
logic_m_5 = []
logic_a_5 = []

loss_mse_1 = []
loss_actor_1 = []
loss_mse_2 = []
loss_actor_2 = []
loss_mse_3 = []
loss_actor_3 = []
loss_mse_4 = []
loss_actor_4 = []
loss_mse_5 = []
loss_actor_5 = []

ram_1 = MemoryBuffer(MAX_BUFFER)
ram_2 = MemoryBuffer(MAX_BUFFER)
ram_3 = MemoryBuffer(MAX_BUFFER)
ram_4 = MemoryBuffer(MAX_BUFFER)
ram_5 = MemoryBuffer(MAX_BUFFER)
ram_6 = MemoryBuffer(MAX_BUFFER)

trainer_1 = Trainer(2, 1, ram_1,batch_size)
trainer_2 = Trainer(2, 1, ram_2,batch_size) # , A_MAX
trainer_3 = Trainer(2, 1, ram_3,batch_size) # , A_MAX
trainer_4 = Trainer(2, 1, ram_4,batch_size) # , A_MAX
trainer_5 = Trainer(2, 1, ram_5,batch_size) # , A_MAX
trainer_6 = Trainer(2, 1, ram_6,batch_size) # , A_MAX
env = bulk_good(MAX_STEPS,MAX_EPISODES)
for _ep in range(MAX_EPISODES):
    env.state_space_reset()
    x = env.initial_state()
    for steps_in_episode in range(MAX_STEPS):
        
        state_1 = [x[0]/20, x[3]/10]       #(silo-1,hopper-1)
        state_2 = [x[3]/10,x[1]/20]        #(hopper-1,silo-2)
        state_3 = [x[1]/20, x[4]/10]       #(silo-2,hopper-2)
        state_4 = [x[4]/10,x[2]/20]        #(hopper-2,silo-3)
        state_5 = [x[2]/20, x[5]/10]       #(silo-3,hopper-3)
        state_6 = [x[5]/10,x[1]/20]        #(hopper-3,silo-1)
        state_space_1 = torch.tensor(state_1).float()
        state_space_2 = torch.tensor(state_2).float()
        state_space_3 = torch.tensor(state_3).float()
        state_space_4 = torch.tensor(state_4).float()
        state_space_5 = torch.tensor(state_5).float()
        state_space_6 = torch.tensor(state_6).float()
        
        if _ep <1000:
             prob = (-0.089)*_ep + 90
        else:
             prob = 1
                
        randomizer = randrange(0, 101)
        if randomizer < prob:
             #print('\nExplore \n')
             Conveyor_Motor_Speed = ((trainer_1.get_exploration_action(state_space_1)) * 1800) 
             #print('Conveyor_Motor_Speed', Conveyor_Motor_Speed)
            
             VP_Time_2 = (trainer_2.get_exploration_action(state_space_2)) * 10
             #print('VP__Time_2',VP__Time_2)
            
             Vibration_Belt_Start = trainer_3.get_exploration_action(state_space_3)
             if Vibration_Belt_Start < 0.5:
                 Vibration_Belt_Start = 0
             else: Vibration_Belt_Start = 1
             #print('Vibration_Belt_Start',Vibration_Belt_Start)
            
             VP_Time_3 = (trainer_4.get_exploration_action(state_space_4)) * 10
             #print('VP_Time_3',VP_Time_3)
            
             Dosing_Speed = ((trainer_5.get_exploration_action(state_space_5)) * 1500)
             #print('Dosing_Speed',Dosing_Speed)
            
             VP_Time_4 = (trainer_6.get_exploration_action(state_space_6)) * 10
             #print('Demand', Demand)
        
        else:
             #print('\nExploit \n')
             Conveyor_Motor_Speed = ((trainer_1.get_exploitation_action(state_space_1)) * 1800) 
             #print('Conveyor_Motor_Speed', Conveyor_Motor_Speed)
            
             VP_Time_2 = (trainer_2.get_exploitation_action(state_space_2)) * 10
             #print('VP__Time_2',VP__Time_2)
            
             Vibration_Belt_Start = trainer_3.get_exploitation_action(state_space_3)
             if Vibration_Belt_Start < 0.5:
                 Vibration_Belt_Start = 0
             else: Vibration_Belt_Start = 1
             #print('Vibration_Belt_Start',Vibration_Belt_Start)
            
             VP_Time_3 = (trainer_4.get_exploitation_action(state_space_4)) * 10
             #print('VP_Time_3',VP_Time_3)
            
             Dosing_Speed = ((trainer_5.get_exploitation_action(state_space_5)) * 1500)
             #print('Dosing_Speed',Dosing_Speed)
            
             VP_Time_4 = (trainer_6.get_exploitation_action(state_space_6)) * 10
           
        ##############
        Plc = plc()
        
        d_conv = Plc.get_Conveyor_Motor_Speed(x[0],x[3])   #(SILO-1, HOPPER-1)
        d_VP_2 = Plc.get_vp_2(x[3],x[1])            #(HOPPER-1, SILO-2) 
        d_VB = Plc.get_Vibration_Belt_Start(x[1],x[4])     #(SILO-2, HOPPER-2)
        d_VP_3 = Plc.get_vp_3(x[4],x[2])            #(HOPPER-2, SILO-3)
        d_DS = Plc.get_Dosing_Speed(x[2],x[5])             #(SILO-3, HOPPER-3)
        d_VP_4 = Plc.get_vp_4(x[5],x[0])            #(HOPPER-3, SILO-1)
  
        new_state = env.environment_bgs(steps_in_episode,_ep)
        new_Silo_Filllevel_1 = new_state[0]
        new_Silo_Filllevel_2 = new_state[1]
        new_Silo_Filllevel_3 = new_state[2]
        new_Hopper_Filllevel_1 = new_state[3]
        new_Hopper_Filllevel_2 = new_state[4]
        new_Hopper_Filllevel_3 = new_state[5]
        new_MF_VC_2 = new_state[19]
        new_MF_VC_3 = new_state[20]
        new_MF_VC_4 = new_state[21]
        MF_S_H_1 = new_state[22]
        MF_S_H_2 = new_state[23]
        MF_S_H_3 = new_state[24]
      
        new_Demand = new_state[53]
        
        Total_Energy_sim = new_state[12]
        #print("Energy",Total_Energy_sim)
        #time.sleep(2)
        reward_1 = new_state[41]
        #print(reward_1)
        reward_2 = new_state[42]
        reward_3 = new_state[43]
        reward_4 = new_state[44]
        reward_5 = new_state[45]
        reward_6 = new_state[46]
        VP_Time_2 = new_state[16]
        reached = new_state[40]

        Total_Reward_1 = new_state[47]
        #print("Total_Reward_1:",Total_Reward_1,"step:",steps_in_episode,"||Episode:",_ep)
        plt.plot(Total_Reward_1)
        Total_Reward_2 = new_state[48]
        Total_Reward_3 = new_state[49]
        Total_Reward_4 = new_state[50]
        Total_Reward_5 = new_state[51]
        Total_Reward_6 = new_state[52]
        #print('hi')
            
        new_state_1 = [new_Silo_Filllevel_1/20, new_Hopper_Filllevel_1/10]
        new_state_2 = [new_Hopper_Filllevel_1/10, new_Silo_Filllevel_2/20]
        new_state_3 = [new_Silo_Filllevel_2/20, new_Hopper_Filllevel_2/10]
        new_state_4 = [new_Hopper_Filllevel_2/10, new_Silo_Filllevel_3/20]
        new_state_5 = [new_Silo_Filllevel_3/20, new_Hopper_Filllevel_3/10]
        new_state_6 = [new_Hopper_Filllevel_3/10, new_Silo_Filllevel_1/20]
        new_state_space_1 = torch.tensor(new_state_1)
        new_state_space_2 = torch.tensor(new_state_2)
        new_state_space_3 = torch.tensor(new_state_3)
        new_state_space_4 = torch.tensor(new_state_4)
        new_state_space_5 = torch.tensor(new_state_5)
        new_state_space_6 = torch.tensor(new_state_6)
        
            # push this exp in ram
        ram_1.add(state_space_1.tolist(), (d_conv/1800),((Conveyor_Motor_Speed)/(1800)), reward_1/100, new_state_space_1.tolist(),reached)
        ram_2.add(state_space_2.tolist(), (d_VP_2/10), [VP_Time_2/10], reward_2/100, new_state_space_2.tolist(),reached)
        ram_3.add(state_space_3.tolist(), (d_VB),[Vibration_Belt_Start], reward_3/100, new_state_space_3.tolist(),reached)
        ram_4.add(state_space_4.tolist(), (d_VP_3/10),VP_Time_3/10, reward_4/100, new_state_space_4.tolist(),reached)
        ram_5.add(state_space_5.tolist(), (d_DS/1500),(Dosing_Speed/1500), reward_5/100, new_state_space_5.tolist(),reached)
        ram_6.add(state_space_6.tolist(), (d_VP_4/10),VP_Time_4/10, reward_6/100, new_state_space_6.tolist(),reached)
        
        trainer_1.optimize()
      
        
        mse_1 = trainer_1.loss_mse_list
        actor_1 = trainer_1.loss_l1_list
        final_loss_1 = trainer_1.loss_final_list
    
        
      
    
        mse_logic_1 = trainer_1.mse_logic
        actor_logic_1 = trainer_1.actor_logic
    
        
        
        
        # #print("End of Step:",steps_in_episode)

        if reached == 1:
     #new_state[32] == reached
            break
        
############################### End of Episode ###################################
    
    Energy_Consumed_Per_Episode.append(np.mean(Total_Energy_sim))
    #print(Energy_Consumed_Per_Episode,"|x:",_ep)
    Episode_counter.append(steps)
    Terminal_reached.append(reached)

    Reward_Per_Episode_1.append(Total_Reward_1)
    #print(Reward_Per_Episode_1,"|x:",_ep)
    
    logic_m_1.append(np.mean(mse_logic_1))
    #print(logic_m_1,"|x:",_ep)
    
    logic_a_1.append(np.mean(actor_logic_1))
    #print(logic_a_1,"|x:",_ep)
    loss_mse_1.append(np.mean(mse_1))
    #print(loss_mse_1,"|x:",_ep)
    loss_actor_1.append(np.mean(actor_1))
    #print(loss_actor_1,"|x:",_ep)


    # check memory consumption and clear memory
    gc.collect()

    torch.save(Energy_Consumed_Per_Episode, 'Energy_Consumed_Per_Episode.pt')
    torch.save(Terminal_reached, 'Terminal_reached.pt')
    torch.save(Reward_Per_Episode_1, 'Reward_Per_Episode_1.pt')
    
    torch.save(Episode_counter, 'Episode_counter.pt')
    
    torch.save(mse_logic_1,'logic_m_1.pt')
    
    
    torch.save(loss_mse_1,'loss_mse_1.pt')
    torch.save(loss_actor_1,'loss_actor_1.pt')
    
    
 

    if _ep%100 == 0:        
        print(_ep)
#        trainer_1.save_models(_ep, r"/home/at-lab/ownCloud/bulk_good (1)/leela/Non linear exploration/target_actor_param_1.pt",r"/home/at-lab/ownCloud/bulk_good (1)//leela/Non linear exploration/target_critic_param_1.pt")
#        trainer_2.save_models(_ep, r"/home/at-lab/ownCloud/bulk_good (1)/leela/Non linear exploration/target_actor_param_2.pt",r"/home/at-lab/ownCloud/bulk_good (1)//leela/Non linear exploration/target_critic_param_2.pt")
#        trainer_3.save_models(_ep, r"/home/at-lab/ownCloud/bulk_good (1)/leela/Non linear exploration/target_actor_param_3.pt",r"/home/at-lab/ownCloud/bulk_good (1)//leela/Non linear exploration/target_critic_param_3.pt")
#        trainer_4.save_models(_ep, r"/home/at-lab/ownCloud/bulk_good (1)/leela/Non linear exploration/target_actor_param_4.pt",r"/home/at-lab/ownCloud/bulk_good (1)//leela/Non linear exploration/target_critic_param_4.pt")
#        trainer_5.save_models(_ep, r"/home/at-lab/ownCloud/bulk_good (1)/leela/Non linear exploration/target_actor_param_5.pt",r"/home/at-lab/ownCloud/bulk_good (1)//leela/Non linear exploration/target_critic_param_5.pt")
#        #trainer_1.load_models(_ep)
        #trainer_2.load_models(_ep)
        #trainer_3.load_models(_ep)
        #trainer_4.load_models(_ep)
        #trainer_5.load_models(_ep)
        #trainer_6.load_models(_ep)
        
print ('\n \nCompleted episodes')

now = datetime.datetime.now()

#print
print("Simulation Ended at:")
print(str(now))

#fig=plt.figure(figsize=(20,10))

#plt.plot(torch.arange(0,10,1), torch.rand(10,1))
#fig.savefig("")
