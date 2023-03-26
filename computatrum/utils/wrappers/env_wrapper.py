import gymnasium as gym


class EnvWrapper(gym.Env):
    def __init__(self, env) -> None:
        self.env = env
        super().__init__()

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.env, name)
