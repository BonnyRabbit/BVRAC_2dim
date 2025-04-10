import os
import time
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from fdmEnv import BVRAC, SIXCLOCK_TRACK
from callback import TSCallback

device = 'cuda' if torch.cuda.is_available() else 'cpu'
stage = 2

def make_env(stage):
    def _init():
        if stage == 1:
            env = BVRAC()
        elif stage == 2:
            env = SIXCLOCK_TRACK()
        print(f"Environment created: {env}")
        return env
    return _init

def main():
    torch.autograd.set_detect_anomaly(True)
    start_time = time.time()

    log_dir = f'logs/stage{stage}_keep_intrack/'
    os.makedirs(log_dir, exist_ok=True)

    pretrained_path = os.path.join(f'logs/stage{stage}/', 'best_model/best_model.zip')

    n_envs = 4
    batch_size = 64
    n_steps = batch_size // n_envs

    eval_env = SubprocVecEnv([make_env(stage=stage) for _ in range(n_envs)])

    train_env = SubprocVecEnv([make_env(stage=stage) for _ in range(n_envs)])
# 单一环境pdb.set_trace()用于调试
    # eval_env = DummyVecEnv([make_env(stage=stage) for _ in range(n_envs)])
    # train_env = DummyVecEnv([make_env(stage=stage) for _ in range(n_envs)])

    if os.path.exists(pretrained_path):
        print(f"Loading pretrained model from {pretrained_path}")
        model = PPO.load(
            pretrained_path,
            env=train_env,
            verbose=0,
            device=device,
            tensorboard_log=log_dir,
            learning_rate=1e-5,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.25,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(
                net_arch=dict(
                    pi=[512, 256],
                    vf=[512, 256]
                )
            )
        )
        print("Successfully loaded pretrained model!")
    else:
        print("No pretrained model found, initializing new model.")

        model = PPO(
            "MlpPolicy",              # 策略网络类型
            train_env,                # 训练环境
            device=device,            # 训练设备
            verbose=0,                # 显示详细信息的级别
            tensorboard_log=log_dir,  # TensorBoard 日志目录
            learning_rate=1e-5,       # 学习率
            n_steps=n_steps,          # 每次更新前在每个环境中运行的步数
            batch_size=batch_size,    # 小批量大小
            n_epochs=10,              # 优化代理损失时的迭代次数
            gamma=0.99,               # 折扣因子
            gae_lambda=0.95,          # GAE lambda 参数
            clip_range=0.25,          # PPO 剪切参数
            ent_coef=0.001,             # 熵系数
            vf_coef=0.5,              # 值函数系数
            max_grad_norm=0.5,        # 梯度最大范数
            policy_kwargs=dict(
                net_arch=dict(
                    pi=[512, 256],
                    vf=[512, 256]
                    )
                )  # pi、vf 网络结构
        )

    callbacks = [
        CheckpointCallback(
            save_freq=20000 // n_envs,
            save_path=log_dir,
            name_prefix='BVRAC_2dim',
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(log_dir, 'best_model'),
            log_path=log_dir,
            eval_freq=10000 // n_envs,
            deterministic=True,
            render=False,
        ),
        TSCallback(log_dir=log_dir)
    ]

    total_timesteps = 5_000_000

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        tb_log_name='BVRAC_2dim',
        reset_num_timesteps=False,
        progress_bar=True
    )

    model.save(os.path.join(log_dir, 'final_model'))
    print("Model saved before exit.")

    end_time = time.time()
    use_time = end_time - start_time
    hours = int(use_time // 3600)
    minutes = int((use_time % 3600) // 60)
    seconds = int(use_time % 60)

    print(f"训练结束，耗时：{hours}h {minutes}min {seconds}s")

    train_env.close()
    eval_env.close()
    
if __name__ == "__main__":
    main()