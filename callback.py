import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter
from typing import Dict

class TSCallback(BaseCallback):
    def __init__(self,
                 log_dir: str,
                 log_freq: int = 20,
                 verbose: int = 0,  ):
        super().__init__(verbose)
        self.log_dir = os.path.join(log_dir, 'custom_metrics')
        self.log_freq = log_freq
        self.writer = None

        self.episode_buffers: Dict[int, dict] = {}
        self.episode_counter = 0
        self.action_names = ['dphi']
        self.obs_names = [
            'x', 'y', 'x_t', 'y_t', 'phi', 'phi_t', 'psi', 'psi_t',
            'AA', 'ATA'
        ]

    def _init_callback(self) -> None:
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def _convert_value(self, name: str, value: float) -> float:
        if name in ['dphi', 'phi', 'phi_t', 'psi', 'psi_t', 'AA', 'ATA']:
            return value * 180 / np.pi
        elif name in ['x', 'y', 'x_t', 'y_t']:
            return value * (340**2) / 9.8
        else:
            return value

    def _on_step(self) -> bool:
        for env_idx in range(self.training_env.num_envs):
            done = self.locals['dones'][env_idx]

            if env_idx not in self.episode_buffers:
                self.episode_buffers[env_idx] = {
                    'actions': [],
                    'observations': [],
                    'rewards': []
                }

            buffer = self.episode_buffers[env_idx]
            buffer['actions'].append(self.locals['actions'][env_idx])
            buffer['observations'].append(self.locals['new_obs'][env_idx])
            buffer['rewards'].append(self.locals['rewards'][env_idx])

            if done:
                current_episode = self.episode_counter
                self._log_episode(env_idx, current_episode)
                self.episode_counter += 1
                self.episode_buffers[env_idx] = {'actions': [], 'observations': [], 'rewards': []}

        return True

    def _log_episode(self, env_idx: int, episode_num: int) -> None:
        if episode_num % self.log_freq != 0:
            return

        buffer = self.episode_buffers[env_idx]

        obs = np.array(buffer['observations'])

        for step, (act, obs) in enumerate(zip(buffer['actions'], buffer['observations'])):

            if not isinstance(act, (list, tuple, np.ndarray)):
                act = [act]
            for name, value in zip(self.action_names, act):
                self.writer.add_scalar(
                    f"episode_{episode_num}/actions/{name}",
                    self._convert_value(name, value),
                    step
                )

            for name, value in zip(self.obs_names, obs):
                self.writer.add_scalar(
                    f"episode_{episode_num}/observations/{name}",
                    self._convert_value(name, value),
                    step
                )

    def _on_close(self) -> None:
        for env_idx in self.episode_buffers:
            if len(self.episode_buffers[env_idx]['rewards']) > 0:
                self._log_episode(env_idx, self.episode_counter)
                self.episode_counter += 1
        if self.writer:
            self.writer.close()