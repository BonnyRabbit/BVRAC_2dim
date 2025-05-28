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
        self.max_timestep = 3000
        self.state_prev = None
        self.reward_flags = {dist: False for dist in [2000, 1350, 800, 500]}
        # action:[dphi, dv]
        self.action_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
        # state:[x, y, x_t, y_t,rel_state, v, v_t, phi, phi_t, psi, psi_t, AA, ATA]
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
             120*9.8/340, # v
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
        self.reward_flags = {dist: False for dist in [2000, 1350, 800, 500]}
        radius = np.random.uniform(4000, 6000)
        angle = np.random.uniform(0, 2 * np.pi)
        x_t = radius * np.cos(angle)
        y_t = radius * np.sin(angle)
        psi_t = np.arctan2(x_t, y_t)

        initial_state: Dict[str, float] = {
            'x': 0.0,
            'y': 0.0,
            'x_t': x_t,
            'y_t': y_t,
            'v': 90,
            'v_t': 60,
            'phi': 0,
            'phi_t': 0,
            'psi': 0,
            'psi_t': psi_t
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
        if dist <= 2000 and not self.reward_flags[2000]:
            reward += 20
            self.reward_flags[2000] = True
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
        dv = np.clip(action[1], -1, 1) * 4
        dx = v * np.sin(psi)
        dy = v * np.cos(psi)
        v += dv * self.dt
        v = np.clip(v, 60, 120)
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
        
        phi = np.clip(phi,np.deg2rad(-30),np.deg2rad(30))
        psi = BVRAC.check_heading(psi)
        psi_t = BVRAC.check_heading(psi_t)

        aspect_angle = BVRAC.compute_AATA(x, y, x_t, y_t, psi_t)
        angle2aspect = BVRAC.compute_AATA(x, y, x_t, y_t, psi)

        # 判断是否丢失视线
        if angle2aspect <= np.deg2rad(30):
            self.ATA_lost_step = 0
            self.first_ATAgood = True # 第一次进入视线优势区
            self.ATA_lost = False 
        else:
            self.ATA_lost_step += 1
            if self.first_ATAgood:
                self.ATA_lost = self.ATA_lost_step >= 50
        # 判断是否在尾追优势区
        self.track_inbound = 150 <= rel_dist <=450
        if self.track_inbound:
            self.step_track_inboud += 1
        else:
            self.step_track_inboud = 0
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
        self.track_inbound = False
        self.speed_keep_step = 0
        self.ATA_lost = False
        self.ATA_lost_step = 0
        self.first_ATAgood = False
        # 奖励权重
        self.max_reward_angle = 30
        self.max_reward_speed = 20
        self.reward_weights = {
            'angle': self.max_reward_angle / self.max_timestep,
            'speed': self.max_reward_speed / self.max_timestep
        }
    def reset(self, seed=None):
        observation, _ = super().reset(seed)
        self.track_inbound = False
        self.step_track_inboud = 0
        self.speed_keep_step = 0
        self.ATA_lost = False
        self.ATA_lost_step = 0
        self.first_ATAgood = False
        return observation, {}
    
    def get_reward(self, state):
        reward = 0.0
        reward_ATA = 0.0
        reward_speed = 0.0
        rel_speed = (state['v'] - state['v_t'])
        if abs(rel_speed) <= 5:
            self.speed_keep_step += 1
        else:
            self.speed_keep_step = 0

        rel_dist = max(state['rel_dist'], 1e-6)
        angle_2_aspect = state['ATA']
        v_des = 60 + np.sqrt(max(0, 72 * (rel_dist - 250) / 35))
        v_tol = 5

        # 奖励计算
        reward_ATA = 1 - abs(angle_2_aspect / (np.deg2rad(30)))
        # 距离较远时鼓励加速追踪：
        if rel_dist >= 2000:
            reward_speed -= (120 - state['v']) / 60
        # 初次接近2km时，稀疏奖励
        if 300 <= rel_dist <= 2000 and not self.reward_flags[2000]:
            reward += 20.0
            self.reward_flags[2000] = True
        # 距离较近时鼓励减速尾追，多个flag引导任务：
        if 300 <= rel_dist <= 2000:
            reward_speed -= abs(state['v'] - v_des) / v_tol
        reward_dist = [
            (1350, 20.0),
            (800, 20.0),
            (500, 20.0)
        ]
        for dist, r in reward_dist:
            if rel_dist <= dist and abs(state['v'] - v_des) <= v_tol and not self.reward_flags[dist]:
                reward += r
                self.reward_flags[dist] = True
        # 进入zone:
        if self.track_inbound:
            reward_speed -= rel_speed / v_tol
        # 保持奖励
        if self.track_inbound and abs(rel_speed) <= 5 and self.step_track_inboud >= 200:
            reward += 100.0
        # 跟丢惩罚
        if rel_dist >= 2000 and self.reward_flags[2000]:
            reward -= 20
        if self.ATA_lost and self.first_ATAgood:
            reward -= 20

        reward += (
            self.reward_weights['speed'] * reward_speed +
            self.reward_weights['angle'] * reward_ATA
        )
        return reward

    def get_done(self, state):
        rel_dist = state['rel_dist']
        rel_speed = state['v'] - state['v_t']
        angle_2_aspect = state['ATA']
        # 200-300m内且尾追一段时间
        if self.track_inbound and abs(rel_speed) <= 5 and self.step_track_inboud >= 200:
            return True, False
        # if rel_dist <= 1500 and abs(rel_speed) < 5 and self.speed_keep_step >= 200:
        #     return True, False
        if rel_dist >= 2000 and self.reward_flags[2000]:
            return True, False
        if self.ATA_lost and self.first_ATAgood:
            return True, False
        # 超过时间限制
        elif self.time_step >= self.max_timestep:
            return True, True
        else:
            return False, False
    
class WVRAC(SIXCLOCK_TRACK):
    def __init__(self, single_turn_times=1, s_turn_times=2):
        super().__init__()
        self.maneuver_mode = None
        self.s_turn_direction = None
        self.s_turn_stage = None
        self.s_turn_psi_list = None
        self.psi_t_des = None
        self.single_turn_times = single_turn_times
        self.s_turn_times = s_turn_times
        self.single_turn_count = 0
        self.s_turn_count = 0
        self.track_out_step = 0

        self.max_reward_angle = 50
    def maneuver_library(self, mode, psi_t):
        """
        机动库
        mode: 机动模式, 1为单方向转弯, 2为S转弯
        return: phi_t
        """
        if mode == 1:
            if self.single_turn_count >= self.single_turn_times:
                return 0
            if self.psi_t_des is None or self.single_turn_direction is None:
                direction = np.random.choice([-1, 1])
                self.single_turn_direction = direction
                self.psi_t_des = self.check_heading(psi_t + direction * np.pi)
            if abs(self.psi_t_des - psi_t) >= np.deg2rad(20):
                return np.deg2rad(8) * self.single_turn_direction
            else:
                self.single_turn_count += 1
                self.psi_t_des = None
                self.single_turn_direction = None
                return 0
        elif mode == 2:
            if self.s_turn_count >= self.s_turn_times:
                return 0
            if self.s_turn_stage is None:
                self.psi_t_des = psi_t
                direction = np.random.choice([-1, 1])
                self.s_turn_direction = direction
                self.s_turn_stage = 0
                self.s_turn_psi_list = [
                    self.psi_t_des + direction * np.deg2rad(50),   
                    self.psi_t_des - direction * np.deg2rad(50),   
                    self.psi_t_des                                  
                ]
            psi_target = self.s_turn_psi_list[self.s_turn_stage]
            if abs(psi_t - psi_target) >= np.deg2rad(3):
                if self.s_turn_stage == 1:
                    turn_dir = -self.s_turn_direction
                else:
                    turn_dir = self.s_turn_direction
                return np.deg2rad(8) * turn_dir
            else:
                self.s_turn_stage += 1
                if self.s_turn_stage >= len(self.s_turn_psi_list):
                    self.s_turn_stage = None
                    self.s_turn_psi_list = None
                    self.s_turn_direction = None
                    self.psi_t_des = None
                    self.s_turn_count += 1
                    return 0
                return 0
        return 0

    def reset(self, seed=None):
        observation, _ = super().reset(seed)
        self.maneuver_mode = None
        self.s_turn_direction = None
        self.s_turn_stage = None
        self.s_turn_psi_list = None
        self.psi_t_des = None
        self.single_turn_direction = None
        self.single_turn_count = 0
        self.s_turn_count = 0
        self.track_out_step = 0
        return observation, {}
    
    def run(self, state, action):
        # 追击无人机:
        x, y = state['x'], state['y']
        v = state['v']
        phi = state['phi']
        psi = state['psi']
        dphi = np.clip(action[0], -1, 1) * np.deg2rad(30)
        dv = np.clip(action[1], -1, 1) * 4
        dx = v * np.sin(psi)
        dy = v * np.cos(psi)
        v += dv * self.dt
        v = np.clip(v, 60, 120)
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

        pursuer_pos = np.array([x, y])
        target_pos = np.array([x_t, y_t])
        rel_dist = np.linalg.norm(pursuer_pos - target_pos)
        if rel_dist < 4000:
            if self.maneuver_mode is None:
                self.maneuver_mode = np.random.choice([1, 2])
            phi_t = self.maneuver_library(self.maneuver_mode, psi_t)
            if (self.maneuver_mode == 1 and self.single_turn_count >= self.single_turn_times) or \
               (self.maneuver_mode == 2 and self.s_turn_count >= self.s_turn_times):
                self.maneuver_mode = None
        else:
            phi_t = 0
            self.maneuver_mode = None

        dphi_t = phi_t
        dx_t = v_t * np.sin(psi_t)
        dy_t = v_t * np.cos(psi_t)
        dpsi_t = 9.81/v_t * np.tan(phi_t)
        phi_t += dphi_t * self.dt
        psi_t += dpsi_t * self.dt
        x_t += dx_t * self.dt
        y_t += dy_t * self.dt

        phi = np.clip(phi, np.deg2rad(-30), np.deg2rad(30))
        psi = BVRAC.check_heading(psi)
        psi_t = BVRAC.check_heading(psi_t)

        aspect_angle = BVRAC.compute_AATA(x, y, x_t, y_t, psi_t)
        angle2aspect = BVRAC.compute_AATA(x, y, x_t, y_t, psi)

        # 判断是否丢失视线
        if angle2aspect <= np.deg2rad(60):
            self.ATA_lost_step = 0
            self.first_ATAgood = True
            self.ATA_lost = False
        else:
            self.ATA_lost_step += 1
            if self.first_ATAgood:
                self.ATA_lost = self.ATA_lost_step >= 200
        # 判断是否在尾追优势区
        if 150 <= rel_dist <= 450:
            self.track_inbound = True
            self.step_track_inboud += 1
            self.track_out_step = 0  # 新增：重置出区计数
        else:
            if hasattr(self, "track_out_step"):
                self.track_out_step += 1
            else:
                self.track_out_step = 1
            if self.track_out_step > 100:
                self.track_inbound = False
                self.step_track_inboud = 0

        state = {
            'x': x,
            'y': y,
            'x_t': x_t,
            'y_t': y_t,
            'rel_dist': rel_dist,
            'v': v,
            'v_t': v_t,
            'phi': phi,
            'phi_t': phi_t,
            'psi': psi,
            'psi_t': psi_t,
            'ATA': angle2aspect,
            'AA': aspect_angle,
        }
        return state
    
    def get_reward(self, state):
        reward = 0.0
        reward_ATA = 0.0
        reward_speed = 0.0
        rel_speed = (state['v'] - state['v_t'])
        if abs(rel_speed) <= 5:
            self.speed_keep_step += 1
        else:
            self.speed_keep_step = 0

        rel_dist = max(state['rel_dist'], 1e-6)
        angle_2_aspect = state['ATA']
        v_des = 60 + np.sqrt(max(0, 72 * (rel_dist - 250) / 35))
        v_tol = 5

        # 奖励计算
        reward_ATA = 1 - abs(angle_2_aspect / (np.deg2rad(30)))
        # 距离较远时鼓励加速追踪：
        if rel_dist >= 2000:
            reward_speed -= (120 - state['v']) / 60
        # 初次接近2km时，稀疏奖励
        if 300 <= rel_dist <= 2000 and not self.reward_flags[2000]:
            reward += 20.0
            self.reward_flags[2000] = True
        # 距离较近时鼓励减速尾追，多个flag引导任务：
        if 300 <= rel_dist <= 2000:
            reward_speed -= abs(state['v'] - v_des) / v_tol
        reward_dist = [
            (1350, 20.0),
            (800, 20.0),
            (500, 20.0)
        ]
        for dist, r in reward_dist:
            if rel_dist <= dist and abs(state['v'] - v_des) <= v_tol and not self.reward_flags[dist]:
                reward += r
                self.reward_flags[dist] = True
        # 进入zone:
        if self.track_inbound:
            reward_speed -= rel_speed / v_tol
        # 保持奖励
        if self.track_inbound and abs(rel_speed) <= 5 and self.step_track_inboud >= 200:
            reward += 100.0
        # 跟丢惩罚
        if rel_dist >= 2000 and self.reward_flags[2000]:
            reward -= 20

        reward += (
            self.reward_weights['speed'] * reward_speed +
            self.reward_weights['angle'] * reward_ATA
        )
        return reward
    
    def get_done(self, state):
        rel_dist = state['rel_dist']
        rel_speed = state['v'] - state['v_t']
        # 200-300m内且尾追一段时间
        if self.track_inbound and abs(rel_speed) <= 5 and self.step_track_inboud >= 200:
            return True, False
        if rel_dist >= 2000 and self.reward_flags[2000]:
            return True, False
        # 超过时间限制
        elif self.time_step >= self.max_timestep:
            return True, True
        else:
            return False, False