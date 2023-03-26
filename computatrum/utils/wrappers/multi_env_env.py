from __future__ import annotations
import gymnasium as gym


class MultiEnvEnv(gym.Env):

    envs: dict[str, gym.Env]

    def __init__(self, **envs):
        self.envs = envs

    def reset(self, *a, **kw):
        for name, env in self.envs.items():
            env.reset(*a, **kw)

    def step(self, action, *a, **kw):
        for name, env in self.envs.items():
            env.step(action, *a, **kw)

    @property
    def observation_space(self):
        return gym.spaces.Dict({name: env.observation_space for name, env in self.envs})

    @property
    def action_space(self):
        return gym.spaces.Dict({name: env.action_space for name, env in self.envs})
