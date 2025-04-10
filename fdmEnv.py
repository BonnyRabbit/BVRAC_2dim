import numpy as np
import gymnasium as gym
import pdb
from typing import Dict
from gymnasium import spaces

class BVRAC(gym.Env):
    
    def __init__(self):
        super(BVRAC, self).__init__()
        # 部分全局参数
        self.dt = 0.25
        self.time_step = 0
        self.max_timestep = 5000
        self.state_prev = None
        self.reward_dist_500 = False
        self.reward_dist_300 = False
        self.reward_dist_100 = False
        # action:[dphi, dv]
        self.action_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
        # state:[x, y, x_t, y_t, phi, phi_t, psi, psi_t]
        low = np.array([
            -1e4*9.8/(340**2), # x
            -1e4*9.8/(340**2), # y
            -1e4*9.8/(340**2), # x_t
            -1e4*9.8/(340**2), # y_t
            -1.4e4*9.8/(340**2), # rel_dist
             50*9.8/340, # v
             40*9.8/340, # v_t
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
             1.4e4*9.8/(340**2), # rel_dist
             130*9.8/340, # v
             100*9.8/340, # v_t
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
        self.reward_dist_500 = False
        self.reward_dist_300 = False
        self.reward_dist_50 = False

        initial_state: Dict[str, float] = {
            'x': 0.0,
            'y': 0.0,
            'x_t': np.random.uniform(-5000, 5000),
            'y_t': np.random.uniform(-5000, 5000),
            'v': 90,
            'v_t': 60,
            'phi': 0,
            'phi_t': 0,
            'psi': 0,
            'psi_t': np.random.uniform(-np.pi, np.pi)
        }
        # 计算rel_dist ATA 和 AA
        pursuer_pos = np.array([initial_state['x'],initial_state['y']])
        target_pos = np.array([initial_state['x_t'], initial_state['y_t']])
        initial_state['rel_dist'] = np.linalg.norm(pursuer_pos - target_pos)

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
        # if self.time_step > 1500:
        #     pdb.set_trace()
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
            state['rel_dist']*9.8/(340**2),
            state['v']*9.8/340,
            state['v_t']*9.8/340,
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
        if distance <= 100:
            return True, False
        # 超过时间限制
        elif self.time_step >= self.max_timestep:
            return True, True
        else:
            return False, False
        
    
    def get_reward(self, state):
        reward = 0.0
        max_reward = 15
        max_total_reward = 1 * self.max_timestep
        w_reward = max_reward / max_total_reward
        pursuer_pos = np.array([state['x'], state['y']])
        target_pos = np.array([state['x_t'], state['y_t']])
        dist = np.linalg.norm(pursuer_pos - target_pos)
        if dist <= 0:
            dist = 1e-6
        angle_2_aspect = state['ATA']

        reward += w_reward * (1 - angle_2_aspect/(np.pi/6))
        if dist <= 500 and not self.reward_dist_500:
            reward += 20
            self.reward_dist_500 = True
        if dist <= 300 and not self.reward_dist_300:
            reward += 30
            self.reward_dist_300 = True
        if dist <= 100 and not self.reward_dist_100:
            reward += 50
            self.reward_dist_100 = True
            
        return reward
    
    def run(self, state, action):
        # 追击无人机:
        x, y = state['x'], state['y']
        v = state['v']
        phi = state['phi']
        psi = state['psi']
        # update
        dphi = np.clip(action[0], -1, 1) * np.deg2rad(30)
        dv = np.clip(action[1], -1, 1) * 1
        dx = v * np.sin(psi)
        dy = v * np.cos(psi)
        v += dv * self.dt
        v = np.clip(v, 50, 130)
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

        pursuer_pos = np.array([state['x'], state['y']])
        target_pos = np.array([state['x_t'], state['y_t']])
        rel_dist = np.linalg.norm(pursuer_pos - target_pos)

        # 判断是否在尾追优势区
        self.track_inbound = 200 <= rel_dist <=300
        if self.track_inbound:
            self.step_track_inboud += 1
        else:
            self.step_track_inboud = 0
        
        phi = np.clip(phi,np.deg2rad(-30),np.deg2rad(30))
        psi = BVRAC.check_heading(psi)
        psi_t = BVRAC.check_heading(psi_t)

        aspect_angle = BVRAC.compute_AATA(x, y, x_t, y_t, psi_t)
        angle2aspect = BVRAC.compute_AATA(x, y, x_t, y_t, psi)
        # 返回state
        state = {
            'x': x,
            'y': y,
            'x_t': x_t,
            'y_t': y_t,
            'rel_dist':rel_dist,
            'v': v,
            'v_t': v_t,
            'phi': phi,
            'phi_t': phi_t,
            'psi': psi,
            'psi_t': psi_t,
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
    
class SIXCLOCK_TRACK(BVRAC):
    def __init__(self):
        super().__init__()
        self.step_track_inboud = 0
        self.max_reward_angle = 15
        self.max_reward_dist = 10
        self.reward_weights = {
            'angle': self.max_reward_angle / self.max_timestep,
            'dist': self.max_reward_dist / self.max_timestep
        }
    
    def reset(self, seed=None):
        observation, _ = super().reset(seed)
        self.track_inbound = False
        self.step_track_inboud = 0
        return observation, {}
    
    def get_reward(self, state):
        reward = 0.0
        reward_dist = 0.0
        reward_ATA = 0.0
        rel_speed = (state['v'] - state['v_t'])
        rel_dist = state['rel_dist']

        if rel_dist <= 0:
            rel_dist = 1e-6
        angle_2_aspect = state['ATA']
        # 1.稠密奖励，通过方向角引导获得回合奖励
        reward_ATA = 1 - np.abs(angle_2_aspect/np.deg2rad(30))
        # 2.稀疏奖励，通过距离
        if rel_dist <= 500 and not self.reward_dist_500:
            reward += 20
            self.reward_dist_500 = True
        # 3.稠密奖励，通过保持200-300m距离尾追获得引导
        # if self.track_inbound:
        if self.reward_dist_500:
            reward_dist = 1 - np.abs(rel_speed / 40)
        else:
            self.step_track_inboud = 0
        # 4.稀疏奖励，通过保持位置+持续时间获得回合奖励
        if self.track_inbound and self.step_track_inboud >= 500:
            reward += 15

        reward += (
            self.reward_weights['angle'] * reward_ATA +
            self.reward_weights['dist'] * reward_dist
        )
        return reward

    def get_done(self, state):
        rel_dist = state['rel_dist']
        # 200-300m内且尾追一段时间
        if self.track_inbound and self.step_track_inboud > 500:
            return True, False
        # 超过时间限制
        elif self.time_step >= self.max_timestep:
            return True, True
        else:
            return False, False