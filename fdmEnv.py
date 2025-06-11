import numpy as np
import gymnasium as gym
import pdb
from typing import Dict
from gymnasium import spaces

class LAND(gym.Env):
    
    def __init__(self):
        super(LAND, self).__init__()
        # 部分全局参数
        self.dt = 0.005
        self.time_step = 0
        self.max_timestep = 7000
        self.state_prev = None
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

        # 飞机初始位置
        self.uav_x = 0
        self.uav_y = 0
        self.uav_z = 150


        # 着陆点
        self.landing_x = 900
        self.landing_y = 0
        self.landing_z = 0


        #标志位
        self.low_hight = False
        self.landing_hight = False
        self.out_bound_alp = False
        self.out_bound_gamma = False
        self.out_bound_V = False
        self.out_bound_z = False
        self.landing_stage1 = False
        self.landing_stage2 = False
        self.landing_stage3 = False

        self.last_dist_horizontal = self.get_dist_horizontal(self.uav_x, self.uav_y)
        

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
            -10*9.8/340, # hdot

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
            10*9.8/340, # hdot

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
            'x': self.uav_x,
            'y': self.uav_y,
            'z': self.uav_z,
            'V': 25,
            'gamma': 0/57.3,
            'psi': 0,
            'alpha': 8/57.3,
            'hdot': 0,
            'x_landing': self.landing_x,
            'y_landing': self.landing_y,
            'z_landing': self.landing_z
        }
        self.state = initial_state
        self.low_hight = False
        self.landing_hight = False
        self.out_bound_alp = False
        self.out_bound_gamma = False
        self.out_bound_V = False
        self.out_bound_z = False
        self.landing_stage1 = False
        self.landing_stage2 = False
        self.landing_stage3 = False

        self.last_dist_horizontal = self.get_dist_horizontal(self.uav_x, self.uav_y)
        
        observation = self.get_observation()
        return observation, {}


    def step(self, action):
        self.time_step += 1
        self.last_dist_horizontal = self.get_dist_horizontal(self.state['x'], self.state['y'])
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
            state['hdot']*9.8/340,
            state['x_landing']*9.8/(340**2), 
            state['y_landing']*9.8/(340**2), 
            state['z_landing']*9.8/(340**2) 
        ])
        return observation

    def get_done(self, state):

        uav_pos = np.array([state['x'], state['y'], state['z']])
        landing_pos = np.array([state['x_landing'], state['y_landing'], state['z_landing']])
        landing_pos_horizontal = np.array([state['x_landing'], state['y_landing']])
        dist = np.linalg.norm(uav_pos - landing_pos)

        # 到目标点
        if  dist < 1:
            return True, False
        # 掉地上
        if state['z'] <= 0.1 :
           print('落地')
           return True, False
        if state['z'] >= 160 :
           print('升高')
           return True, False
        if abs(state['alpha']) >= np.deg2rad(15) :
           print('alp超界')
           return True, False      
        # 超过时间限制
        elif self.time_step >= self.max_timestep:
            return True, True

        else:
            return False, False
        

    
    def run(self, state, action):
        # state
        x, y, z = state['x'], state['y'], state['z'] # 反归一化 m
        V = state['V']
        gamma = state['gamma']
        psi = state['psi']
        alpha = state['alpha']
        x_landing = state['x_landing']
        y_landing = state['y_landing']
        z_landing = state['z_landing']
        
        # action
        tht = np.clip(action[0], -1, 1) * np.deg2rad(12) # rad
        thr = (np.clip(action[1], -1, 1) + 1) * 0.5 *0   # m/s
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
            'hdot':dz,
            'x_landing':x_landing,
            'y_landing':y_landing,
            'z_landing':z_landing

        }
        return state
    
    def get_reward(self, state):

        ## stage1 :2D纵向平面内的训练，训练目标是到达5m时满足着陆窗口 

        # reward初始化
        reward = 0.0
        reward_alp = 0.0
        reward_alpha = 0.0
        reward_gamma = 0.0
        reward_V = 0.0
        reward_Vmax = 0.0
        reward_Vmin = 0.0
        guide_term  = 0.0
        V_err = 0.0
        alpha_err = 0.0

        # 参数计算
        uav_pos = np.array([state['x'], state['y'], state['z']])
        uav_pos_horizontal = np.array([state['x'], state['y']])
        uav_pos_vert = np.array([0, 0, state['z']])
        V = state['V']
        z = state['z']
        gamma = state['gamma']
        psi = state['psi']
        alpha = state['alpha']
        dz = state['hdot']  # m
        

        
        landing_pos = np.array([state['x_landing'], state['y_landing'], state['z_landing']])
        landing_pos_horizontal = np.array([state['x_landing'], state['y_landing']])
        landing_pos_vert = np.array([0, 0, state['z_landing']])
        dist = np.linalg.norm(uav_pos - landing_pos)
        dist_horizontal = np.linalg.norm(uav_pos_horizontal - landing_pos_horizontal)
        dist_vert = np.linalg.norm(state['z'] -state['z_landing'])
        max_dist = np.linalg.norm([0,0,150] - landing_pos)
        dist_horizontal_max =  np.linalg.norm([0,0] - landing_pos_horizontal)
        dist_vert_max = np.linalg.norm(150 - state['z_landing'])

    
    
        if dist <= 0:
            dist = 1e-6
        if dist_horizontal <= 0:
            dist_horizontal = 1e-6           
        if dist_vert <= 0:
            dist_vert = 1e-6
        
        reward_total_sense = 10 #稠密奖励的总和
        reward_step = reward_total_sense/self.max_timestep


        # 正向引导奖励（高空）

        #reward +=  reward_step * (1-dist_horizontal/dist_horizontal_max)
        
        dist_horizontal_error = self.last_dist_horizontal - dist_horizontal
        reward += (dist_horizontal_error / dist_horizontal_max) *reward_total_sense

        if  10 < state['z'] < 40 :
            if -5 < dz < -1 :
                reward += 1.2 * reward_step
            elif -8 < dz < -5 :
                reward += 1.2 * reward_step * ((dz + 8)/3)
            else:
                reward += -reward_step
        elif 0 < state['z'] < 10 :
            if -3 < dz < -1 :
                reward += 1.2 * reward_step
            elif -5 < dz < -3 :
                reward += 1.2 * reward_step * ((dz + 5)/2)
            else:
                reward += -reward_step

        # 到达中高度
        if state['z'] < 30 and self.landing_stage1 == False:

            # 窗口达标奖励
            if dist_horizontal <= 200 :
                reward += 2
                if -8 < dz < -1 :
                    reward += 2
                elif -5 < dz < -1 :
                    reward += 5
                else: reward -= 5
            else: reward -= 0

            self.landing_stage1 = True

        # 末端窗口奖励
        if state['z'] < 10 and self.landing_stage2 == False:
            if dist_horizontal <= 100 :
                reward += 5
                if -5 < dz < -3 :
                    reward += 5
                elif -3 < dz < 0 :
                    reward += 10
                else:
                    reward -= 5
            else: reward -= 0

            self.landing_stage2 = True

        # 着陆结果
        if (state['z'] < 2) and self.landing_stage3 == False:
            if dist_horizontal < 50:
                reward += 60
                if (dz < 0) and (dz > -2):
                    reward += 100
                elif (dz < -2) and (dz > -5):
                    reward += 50
                else:
                    reward -= 50
            else: reward -= 100

            self.landing_stage3 = True
        
        #if (state['z'] < 8) and (dz < 0) and (dz > -5) and self.landing_hight == False:
             #低空着陆达标奖励
        #    reward += 50

            #低空着陆标志位
        #    self.landing_hight = True

        # if (state['z'] < 2) and (dz < 0) and (dz > -2) :
        #     reward += 500



        # 惩罚失稳：迎角过大或飞行路径角异常(全程 稠密奖励)
        if  np.deg2rad(10) < abs(state['alpha']) <  np.deg2rad(12):
            reward_alp -= reward_step * ((abs(state['alpha']) - np.deg2rad(10)) / np.deg2rad(2)) 

        if  np.deg2rad(-3) < abs(state['alpha']) <  np.deg2rad(0):
            reward_alp -= reward_step * ((abs(state['alpha']) - np.deg2rad(-3)) / np.deg2rad(3)) 

        if np.deg2rad(25) < abs(state['gamma']) < np.deg2rad(30):
            reward_gamma -= reward_step * ((abs(state['gamma']) - np.deg2rad(25)) / np.deg2rad(5)) 

        if 35 < state['V'] < 40:
            reward_Vmax -= reward_step * ((state['V'] - 35) / 5) 
        
        if 20 < state['V'] < 22:
            reward_Vmin -= reward_step * ((22 - state['V']) / 2) 

        reward += 0.35 * reward_alp + 0.35 * reward_gamma + 0.3 * reward_Vmax + 0.3 * reward_Vmin
        # 压缩到[-1 0]
        # if reward_add < 0:
        #    reward += np.tanh(3 * reward_add)

        # 硬约束 不done，给大惩罚
        if abs(state['alpha']) > np.deg2rad(12) and self.out_bound_alp == False:
            reward -= 2
            self.out_bound_alp = True
        if abs(state['gamma']) > np.deg2rad(30) and self.out_bound_gamma == False:
            reward -= 2
            self.out_bound_gamma = True
        if  (state['V'] < 20 or state['V'] > 40) and self.out_bound_V == False:
            reward -= 5
            self.out_bound_V = True
            #print(str(self.out_bound_V))
        if state['z'] > 152 and self.out_bound_z == False:
            reward -= 5
            # self.out_bound_z = True

        return reward



    def get_info(self, state):
        return {}
    

    def get_dist_horizontal(self, x,y):

        dist_horizontal = 0.0
        uav_pos_horizontal = np.array([x, y])
        landing_pos_horizontal = np.array([self.landing_x, self.landing_y])
        dist_horizontal = np.linalg.norm(uav_pos_horizontal - landing_pos_horizontal)

        return dist_horizontal

    
    
 