import numpy as np
import gymnasium as gym
import math
from typing import Dict
from gymnasium import spaces

class BVRAC(gym.Env):
    
    def __init__(self):
        super(BVRAC, self).__init__()
        # 部分全局参数
        self.dt = 0.25
        self.time_step = 0
        self.state_prev = None
        self.n_actions = 3
        # action:dphi
        self.action_space = spaces.Discrete(self.n_actions)
        # state:[x, y, x_t, y_t, phi, phi_t, psi, psi_t]
        low = np.array([
            -1e4*9.8/(340**2), # x
            -1e4*9.8/(340**2), # y
            -1e4*9.8/(340**2), # x_t
            -1e4*9.8/(340**2), # y_t
            -np.pi/6, # phi
            -np.pi/6, # phi_t
            -np.pi, # psi
            -np.pi, # psi_t
            -np.pi, # AA
            -np.pi, # ATA
        ])
        high = np.array([
             1e4*9.8/(340**2), # x
             1e4*9.8/(340**2), # y
             1e4*9.8/(340**2), # x_t
             1e4*9.8/(340**2), # y_t
             np.pi/6, # phi
             np.pi/6, # phi_t
             np.pi, # psi
             np.pi, # psi_t
             np.pi, # AA
             np.pi, # ATA
        ])
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        # 初始化状态
        self.state = self.reset()
    
    def reset(self, seed=None,):
        if seed is not None:
            self.seed(seed)
        self.time_step = 0
        # 给定初值
        initial_state: Dict[str, float] = {
            'x': 0.0,
            'y': 0.0,
            'x_t': 500.0,
            'y_t': 1000.0,
            'phi': 0,
            'phi_t': 0,
            'psi': 0.0,
            'psi_t': np.pi/6
        }
        # 计算 ATA 和 AA
        initial_state['ATA'] = self.compute_AATA(
            initial_state['x'], initial_state['y'],
            initial_state['x_t'], initial_state['y_t'],
            initial_state['psi']
        )
        initial_state['AA'] = self.compute_AATA(
            initial_state['x'], initial_state['y'],
            initial_state['x_t'], initial_state['y_t'],
            initial_state['psi_t']
        )
        self.state = initial_state
        observation = self.get_observation()
        return observation, {}

    def step(self, action):
        self.time_step += 1
        self.state = self.run(self.state, action)
        reward = self.get_reward(self.state)
        done, truncated = self.get_done(self.state)
        info = self.get_info(self.state)
        observation = self.get_observation()
        return observation, reward, done, truncated, info


    def get_observation(self):
        state = self.state
        observation = np.array([
            state['x']*9.8/(340**2),
            state['y']*9.8/(340**2),
            state['x_t']*9.8/(340**2),
            state['y_t']*9.8/(340**2),
            state['phi'],
            state['phi_t'],
            state['psi'],
            state['psi_t'],
            state['ATA'],
            state['AA'],
        ])
        return observation

    def get_done(self, state):
        pursuer_pos = np.array([state['x'], state['y']])
        target_pos = np.array([state['x_t'], state['y_t']])
        distance = np.linalg.norm(pursuer_pos - target_pos)
        # 接近到一定距离内
        if distance < 100:
            return True, False
        # 超过时间限制
        elif self.time_step >= 500:
            return True, True
        else:
            return False, False
        
    
    def get_reward(self, state):
        reward = 0.0
        pursuer_pos = np.array([state['x'], state['y']])
        target_pos = np.array([state['x_t'], state['y_t']])
        dist = np.linalg.norm(pursuer_pos - target_pos)
        if dist <= 0:
            dist = 1e-6
        dist_disired = 100
        aspect_angle = state['AA']
        angle_2_aspect = state['ATA']

        reward += ((1 - aspect_angle/np.pi) + (1 - angle_2_aspect/np.pi) /2) * math.exp((-abs(dist-dist_disired)/dist))
        return reward
    
    def run(self, state, action):
        # 追击无人机:
        x, y = state['x'], state['y']
        phi = state['phi']
        psi = state['psi']
        v = 120
        # update
        if action == 0:
            dphi = -np.deg2rad(40)
        if action == 1:
            dphi = 0
        if action == 2:
            dphi = np.deg2rad(40)
        dx = v * np.sin(psi)
        dy = v * np.cos(psi)
        dpsi = 9.81/v * np.tan(phi)
        phi += dphi * self.dt
        psi += dpsi * self.dt
        x += dx * self.dt
        y += dy * self.dt
        
        # 逃逸无人机:
        x_t, y_t = state['x_t'], state['y_t']
        phi_t = state['phi_t']
        psi_t = state['psi_t']
        v_t = 60
        # update
        dphi_t = 0
        dx_t = v_t * np.sin(psi_t)
        dy_t = v_t * np.cos(psi_t)
        dpsi_t = 9.81/v_t * np.tan(phi_t)
        phi_t += dphi_t * self.dt
        psi_t += dpsi_t * self.dt
        x_t += dx_t * self.dt
        y_t += dy_t * self.dt

        aspect_angle = BVRAC.compute_AATA(x, y, x_t, y_t, psi_t)
        angle2aspect = BVRAC.compute_AATA(x, y, x_t, y_t, psi)
        # 返回state
        state = {
            'x': x,
            'y': y,
            'x_t': x_t,
            'y_t': y_t,
            'phi': np.clip(phi,np.deg2rad(-30),np.deg2rad(30)),
            'phi_t': phi_t,
            'psi': BVRAC.check_heading(psi),
            'psi_t': BVRAC.check_heading(psi_t),
            'ATA':angle2aspect,
            'AA':aspect_angle,
        }
        return state
        
    @staticmethod
    def check_heading(psi):
        if psi > np.pi:
            psi -= 2 * np.pi
        elif psi < -np.pi:
            psi += 2 * np.pi
        return psi
    
    @staticmethod
    def compute_AATA(x, y, x_t, y_t, psi):
        LOS = np.array([x_t - x, y_t - y])
        LOS_norm = np.linalg.norm(LOS)
        if LOS_norm == 0:
            print(f"Warning: LOS vector has zero length! x={x}, y={y}, x_t={x_t}, y_t={y_t}")
            return 0
        LOS = LOS / LOS_norm
        v = np.array([np.sin(psi), np.cos(psi)])
        dot_LOS_v = np.clip(np.dot(LOS, v), -1.0, 1.0)
        AATA = np.arccos(dot_LOS_v)
        return AATA

    
    def get_info(self, state):
        return {}