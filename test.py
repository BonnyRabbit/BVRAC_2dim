import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from fdmEnv import BVRAC, SIXCLOCK_TRACK

stage = 1

log_dir = f'logs/stage{stage}_Quadrant_12/'
os.makedirs(log_dir, exist_ok=True)
# model_path = os.path.join(log_dir, 'best_model/best_model.zip')
# model_path = os.path.join(log_dir, 'BVRAC_2dim_3740000_steps.zip')
model_path = os.path.join(log_dir, 'best_model/best_model.zip')


model = PPO.load(model_path, device='cuda' if torch.cuda.is_available() else 'cpu')

env = BVRAC()

obs, _ = env.reset()

done = False
step = 0
# 记录的数据
steps = []
pursuer_traj = []  
target_traj = []
ATA_traj = []   
AA_traj = []

while not done:
    action, _ = model.predict(obs)
    
    obs, reward, done, truncated, info = env.step(action)
    
    scaling_factor = (340**2) / 9.8
    pursuer_x = obs[0] * scaling_factor
    pursuer_y = obs[1] * scaling_factor
    target_x = obs[2] * scaling_factor
    target_y = obs[3] * scaling_factor
    ATA = obs[8] * 180/np.pi
    AA = obs[9] * 180/np.pi
    
    pursuer_traj.append((pursuer_x, pursuer_y))
    target_traj.append((target_x, target_y))
    ATA_traj.append(ATA)
    AA_traj.append(AA)
    steps.append(step)
    step += 1

pursuer_x, pursuer_y = zip(*pursuer_traj)
target_x, target_y = zip(*target_traj)
ATA = ATA_traj
AA = AA_traj    

plt.figure(figsize=(10, 8))
plt.plot(pursuer_x, pursuer_y, 'b-', label='pursuer')
plt.plot(target_x, target_y, 'r-', label='target')
plt.plot(pursuer_x[0], pursuer_y[0], 'bo', label='P_{start point}')
plt.plot(target_x[0], target_y[0], 'ro', label='T_{start point}')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('trajectory')
plt.legend()
plt.grid(True)
plt.gca().set_aspect('equal')
# plt.show()
plt.savefig('trajectory.png')
print("图像已保存为 trajectory.png")

plt.figure(figsize=(10, 8))
plt.plot(np.array(steps) * 0.25, ATA_traj, 'g-', label='ATA (deg)')
plt.plot(np.array(steps) * 0.25, AA_traj, 'm-', label='AA (deg)')
plt.xlabel('Time(s)')
plt.ylabel('Angle (deg)')
plt.title('ATA and AA vs Time')
plt.legend()
plt.grid(True)
plt.savefig('angles_vs_step.png')
print("图像已保存为 angles_vs_time.png")