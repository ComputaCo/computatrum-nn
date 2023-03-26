import gymnasium as gym
from computatrum.utils.wrappers.bufferred_env_wrapper import BufferredEnvWrapper
from computatrum.utils.wrappers.multi_env_env import MultiEnvEnv


def BufferredMultiEnvEnv(envs: dict[str, gym.Env], wait=False, buffersize=-1):
    envs = {
        n: BufferredEnvWrapper(env, wait=wait, buffersize=buffersize)
        for n, env in envs.items()
    }
    env = MultiEnvEnv(**envs)
    return env
