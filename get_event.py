import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

base_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(base_dir, 'logs/stage1_action_box/BVRAC_2dim_0')

fig_dir = os.path.join(base_dir, 'figs')
os.makedirs(fig_dir, exist_ok=True)

event_file = None
for fname in sorted(os.listdir(log_dir)):
    if fname.startswith("events.out.tfevents"):
        path = os.path.join(log_dir, fname)
        try:
            ea = event_accumulator.EventAccumulator(path)
            ea.Reload()
            print(f"成功读取: {fname}")
            print("包含的 tags:", ea.Tags()['scalars'])
            event_file = path
            break
        except Exception as e:
            print(f"读取失败: {fname}，原因: {e}")

if event_file:
    reward_events = ea.Scalars('eval/mean_reward')
    steps = [e.step for e in reward_events]
    values = [e.value for e in reward_events]

    plt.figure(figsize=(10, 8))
    plt.plot(np.array(steps), np.array(values), 'orange', label='mean_reward')
    plt.xlabel('Steps', fontsize=20)
    plt.ylabel('Mean Reward', fontsize=20)
    plt.title('Mean Reward vs Steps', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.legend(fontsize=14)
    plt.grid(True)
    save_path = os.path.join(fig_dir, 'reward.png')
    plt.savefig(save_path)
    print(f"图像已保存为 {save_path}")

    ep_len_events = ea.Scalars('eval/mean_ep_length')
    steps = [e.step for e in ep_len_events]
    values = [e.value for e in ep_len_events]

    plt.figure(figsize=(10, 8))
    plt.plot(np.array(steps), np.array(values), 'g-', label='mean_ep_length')
    plt.xlabel('Steps', fontsize=20)
    plt.ylabel('Mean Episode Length', fontsize=20)
    plt.title('Episode Length vs Steps', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.legend(fontsize=14)
    plt.grid(True)
    save_path = os.path.join(fig_dir, 'episode_length.png')
    plt.savefig(save_path)
    print(f"图像已保存为 {save_path}")
else:
    print("未找到有效的 event 文件，无法绘图。")