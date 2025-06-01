import numpy as np
import gymnasium as gym
import pdb
from typing import Dict
from gymnasium import spaces

class LAND(gym.Env):
    
    def __init__(self):
        super(LAND, self).__init__()
        # 部分全局参数
        self.dt = 0.05
        self.time_step = 0
        self.max_timestep = 5000
        self.state_prev = None
        #self.V_base = 25
        self.RefArea = 0.070331
        self.pho = 1.225

        # 飞机本体参数
        self.m = 2.8
        self.g = 9.8
        self.Tmax = 12.3
        self.CLa = 0.0962*57.3
        self.CL0 = 0.62
        self.A = 0.0538 # CD = A*CL^2 + CD0
        self.CD0 = 0.0388

        # 着陆点
        self.landing_x = 2000
        self.landing_y = 0
        self.landing_z = 0
        self.reward_outbound = False

        # action:[theta, thr, phi, ay]
        self.action_space = spaces.Box(-1, 1, shape=(4,), dtype=np.float32)

        # state:[x, y, gamma]
        low = np.array([
            -1e4*9.8/(340**2), # x
            -1e4*9.8/(340**2), # y
             0*9.8/(340**2), # z
             20*9.8/340, # V
            -np.pi/9, # gamma
            -np.pi, #psi
            -np.pi/60, # alpha

            -1e4*9.8/(340**2), # landing_x
            -1e4*9.8/(340**2), # landing_y
             0*9.8/(340**2), # landing_z       
        ])
        high = np.array([
            1e4*9.8/(340**2), # x
            1e4*9.8/(340**2), # y
            1e3*9.8/(340**2), # z
            30*9.8/340, # V
            np.pi/9, # gamma
            np.pi, #psi
            np.pi/15, # alpha

            1e4*9.8/(340**2), # landing_x
            1e4*9.8/(340**2), # landing_y
            1e3*9.8/(340**2), # landing_z
        ])
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        # 初始化状态
        self.state = self.reset()
    
    def reset(self, seed=None,):
        if seed is not None:
            self.seed(seed)
        self.time_step = 0

        initial_state: Dict[str, float] = {
            'x': 0.0,
            'y': 0.0,
            'z': 150.0,
            'V': 25,
            'gamma': 0/57.3,
            'psi': 0,
            'alpha': 4/57.3,
        }
        self.state = initial_state
        self.reward_outbound = False
        self.has_been_penalized = False
        observation = self.get_observation()
        return observation, {}


    def step(self, action):
        self.time_step += 1
        self.state = self.run(self.state, action)
        done, truncated = self.get_done(self.state)
        reward = self.get_reward(self.state)
        info = self.get_info(self.state)
        observation = self.get_observation()
        return observation, reward, done, truncated, info


    def get_observation(self):
        state = self.state
        observation = np.array([
            state['x']*9.8/(340**2),
            state['y']*9.8/(340**2),
            state['z']*9.8/(340**2),
            state['V']*9.8/340,
            state['gamma'],
            state['psi'],
            state['alpha'],
            self.landing_x*9.8/(340**2), 
            self.landing_y*9.8/(340**2), 
            self.landing_z*9.8/(340**2) 
        ])
        return observation

    def get_done(self, state):

        uav_pos = np.array([state['x'], state['y'], state['z']])
        landing_pos = np.array([self.landing_x, self.landing_y, self.landing_z])
        landing_pos_horizontal = np.array([self.landing_x, self.landing_y])
        dist = np.linalg.norm(uav_pos - landing_pos)

        if  dist < 1:
            return True, False
            self.reward_outbound = True
        if state['z'] <= 1:
           return True, False
        # 超过时间限制
        elif self.time_step >= self.max_timestep:
            return True, True
        #elif state['alpha'] >= 12/57.3 or np.abs(state['gamma']) > np.deg2rad(20) :
        #    self.reward_outbound = True
        #    return True, False
        else:
            return False, False
        

    
    def run(self, state, action):
        # state
        x, y, z = state['x'], state['y'], state['z'] # 反归一化 m
        V = state['V']
        gamma = state['gamma']
        psi = state['psi']
        alpha = state['alpha']
        
        # action
        tht = np.clip(action[0], -1, 1) * np.deg2rad(20) # rad
        thr = (np.clip(action[1], -1, 1) + 1) * 0.5   # m/s
        #phi = np.clip(action[2], -1, 1) * np.deg2rad(30)
        #ay = np.clip(action[3], -1, 1) * 3
        phi = 0
        ay = 0


        # F
        CL = self.CLa * alpha + self.CL0
        CD = self.A * CL ** 2 + self.CD0
        L = 0.5*self.pho*V*V*self.RefArea*CL
        D = 0.5*self.pho*V*V*self.RefArea*CD
        T = thr * self.Tmax

        # run
        dx = V * np.cos(gamma) * np.cos(psi) # m
        dy = V * np.cos(gamma) * np.sin(psi) # m
        dz = V * np.sin(gamma)  # m
        V_dot = (T * np.cos(alpha) - D) / self.m - self.g * np.sin(gamma)
        gamma_dot = (L * np.cos(phi) + T * np.sin(alpha)) / (V * self.m) - self.g * np.cos(gamma) / V
        psi_dot = ay / (V * np.cos(gamma))

        # update
        alpha = tht - gamma # rad
        x += dx * self.dt
        y += dy * self.dt 
        z += dz * self.dt 
        V += V_dot * self.dt 
        gamma += gamma_dot * self.dt
        psi += psi_dot * self.dt

        # return state
        state = {
            'x': x,  
            'y': y,  
            'z': z,  
            'V': V,
            'gamma': gamma,
            'psi': psi,
            'alpha': alpha,
        }
        return state
    
    def get_reward(self, state):
        reward = 0.0
        max_reward = 15

        max_total_reward = 1 * self.max_timestep
        w_reward = max_reward / max_total_reward

        uav_pos = np.array([state['x'], state['y'], state['z']])
        uav_pos_horizontal = np.array([state['x'], state['y']])
        uav_pos_vert = np.array([0, 0, state['z']])
        V = state['V']
        z = state['z']
        gamma = state['gamma']
        psi = state['psi']
        alpha = state['alpha']
        dz = V * np.sin(gamma)  # m
        

        
        landing_pos = np.array([self.landing_x, self.landing_y, self.landing_z])
        landing_pos_horizontal = np.array([self.landing_x, self.landing_y])
        landing_pos_vert = np.array([0, 0, self.landing_z])
        dist = np.linalg.norm(uav_pos - landing_pos)
        dist_horizontal = np.linalg.norm(uav_pos_horizontal - landing_pos_horizontal)
        dist_vert = np.linalg.norm(self.landing_z - state['z'])
        max_dist = np.linalg.norm([0,0,150] - landing_pos)
        dist_horizontal_max =  np.linalg.norm([0,0] - landing_pos_horizontal)

    
    
        if dist <= 0:
            dist = 1e-6
        if dist_horizontal <= 0:
            dist_horizontal = 1e-6           
        if dist_vert <= 0:
            dist_vert = 1e-6



        # 允许偏差归一化范围（用于归一化）
        z_tol = 5.0     # m

        reward +=  1-dist/max_dist
        

        
        # 平滑压缩（使总 reward ∈ [ -1, 0 ]）
        #reward = np.tanh(reward_dist)




        # 惩罚失稳：迎角过大或飞行路径角异常
        if abs(state['alpha']) > np.deg2rad(12):
            reward -= 0.2 * ((abs(state['alpha']) - np.deg2rad(12)) / np.deg2rad(12)) ** 2

        if abs(state['gamma']) > np.deg2rad(20):
            reward -= 0.2 * ((abs(state['gamma']) - np.deg2rad(20)) / np.deg2rad(20)) ** 2

        if state['V'] > 35:
            reward -= 0.2 * (state['V'] - 35) / 5 ** 2
        
        if state['V'] < 20:
            reward -= 0.2 * (20 - state['V']) / 5 ** 2



        if (state['z'] < 2) and (dz < 0) and (dz > -2):
            reward += 50



        # reward += w_reward * (1 - dist/max_dist)
        # reward += 20 * w_reward * (1 - dist_horizontal/dist_horizontal_max)

        #reward +=   w_reward *  ((state['z']-150)/150)




        # if dist_horizontal < 5:
        #   reward += 50

        #if np.deg2rad(10) <= alpha <= np.deg2rad(12):
        #    penalty_factor_aoa = (alpha - np.deg2rad(10)) / np.deg2rad(2)  # 从0增长到1
        #    reward += -penalty_factor_aoa * w_reward

        #if np.deg2rad(10) <= np.abs(gamma) <= np.deg2rad(20):
        #    penalty_factor_gamma = (np.abs(gamma) - np.deg2rad(10)) / np.deg2rad(10)  # 从0增长到1
        #    reward += -5 * penalty_factor_gamma * w_reward
            
        #if 40 <= V :
        #    penalty_factor_V = (V - 40) / 20  # 从0增长到1
        #    reward += -5 * penalty_factor_V * w_reward 

        # if (state['z'] < 2) and dist < 5:
        #    reward += 15


        # if (state['z'] < 2) and (dz < 0) and (dz > -2):
        #    reward += 50


        #if state['alpha'] >= 12/57.3 or np.abs(state['gamma']) > np.deg2rad(20) :
        #    self.reward_outbound = True    

        #if self.reward_outbound == True:
        #    reward += -50
        #    self.reward_outbound = False
        
    
        return reward



    def get_info(self, state):
        return {}
    
 