import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from fdmEnv import LAND  # 确保 fdmEnv.py 中定义了 LAND


def evaluate_model(model_path, n_episodes=5, plot=True):
    # 初始化环境
    env = LAND()

    # 加载模型（不绑定 env 避免结构冲突）
    model = PPO.load(model_path, device='cuda' if torch.cuda.is_available() else 'cpu')

    print(f"✅ 正在评估模型：{model_path}")
    success_count = 0

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        ep_reward = 0
        traj = {
            'x': [], 'z': [], 'V': [], 'alpha': [], 'gamma': []
        }

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            ep_reward += reward

            traj['x'].append(env.state['x'])
            traj['z'].append(env.state['z'])
            traj['V'].append(env.state['V'])
            traj['alpha'].append(env.state['alpha'])
            traj['gamma'].append(env.state['gamma'])

        success = (env.state['z'] < 2 and env.state['V'] > 0 and done and not truncated)
        success_count += int(success)

        print(f"[Episode {ep+1}] Total Reward: {ep_reward:.2f}, Done: {done}, Success: {success}")

        if plot:
            plt.figure(figsize=(6, 4))
            plt.plot(traj['x'], traj['z'], label='Trajectory')
            plt.xlabel('X [m]')
            plt.ylabel('Z [m]')
            plt.title(f"Episode {ep+1} Trajectory")
            plt.grid(True)
            plt.gca().invert_yaxis()
            plt.tight_layout()

            # 保存图像（使用绝对路径）
            save_dir = os.path.join(os.path.dirname(model_path), 'eval_figs')
            os.makedirs(save_dir, exist_ok=True)
            fig_path = os.path.join(save_dir, f"trajectory_ep{ep+1}.png")
            plt.savefig(fig_path)
            print(f"📁 已保存轨迹图: {fig_path}")
            plt.close()

    print(f"\n🎯 成功降落 {success_count}/{n_episodes} 回合")


if __name__ == '__main__':
    # 使用绝对路径，确保无论从哪里执行都不会出错
    root_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(root_dir, 'logs/best_model/best_model.zip')

    assert os.path.isfile(model_path), f"❌ 模型文件不存在: {model_path}"
    evaluate_model(model_path=model_path, n_episodes=5, plot=True)
