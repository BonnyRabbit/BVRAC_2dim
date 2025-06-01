from matplotlib import rcParams
from fdmEnv import LAND
import matplotlib.pyplot as plt
import numpy as np

def check_trim(tht_deg=8.0, thr_ratio=0.5, steps=100):
    """
    检查在指定俯仰角tht（度）和油门比例（0~1）下，飞机能否平飞配平。
    
    参数：
    - tht_deg: 实际俯仰角（单位：度）
    - thr_ratio: 油门推力比例 [0~1]
    - steps: 模拟步数
    """
    rcParams.update({'font.size': 12})
    
    # 初始化环境
    env = LAND()
    obs, _ = env.reset()

    # 归一化动作
    tht_norm = np.clip(tht_deg / 12.0, -1, 1)
    thr_norm = np.clip(thr_ratio * 2 - 1, -1, 1)  # [0,1] -> [-1,1]

    action = np.array([
        tht_norm,
        thr_norm,
        0.0,  # phi
        0.0   # ay
    ])

    # 轨迹记录
    gamma_list, V_list, alpha_list, z_list, time_list = [], [], [], [], []
    for i in range(steps):
        obs, reward, done, truncated, info = env.step(action)
        s = env.state
        gamma_list.append(np.rad2deg(s['gamma']))
        V_list.append(s['V'])
        alpha_list.append(np.rad2deg(s['alpha']))
        z_list.append(s['z'])
        time_list.append(i * env.dt)
        if done:
            print("⚠️ Episode ended early.")
            break

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)
    plt.plot(time_list, gamma_list)
    plt.ylabel('gamma (°)')
    plt.grid()

    plt.subplot(3, 1, 2)
    plt.plot(time_list, V_list)
    plt.ylabel('Velocity (m/s)')
    plt.grid()

    plt.subplot(3, 1, 3)
    plt.plot(time_list, alpha_list)
    plt.ylabel('alpha (°)')
    plt.xlabel('Time (s)')
    plt.grid()
    plt.tight_layout()
    plt.show()

    print(f"\n✅ 最终状态:")
    print(f"  高度 z = {s['z']:.2f} m")
    print(f"  速度 V = {s['V']:.2f} m/s")
    print(f"  路径角 gamma = {np.rad2deg(s['gamma']):.2f} deg")
    print(f"  迎角 alpha = {np.rad2deg(s['alpha']):.2f} deg")
# 自动运行

if __name__ == "__main__":
     check_trim(tht_deg=4.0, thr_ratio=0.22, steps=2000)