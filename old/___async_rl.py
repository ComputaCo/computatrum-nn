import gymnasium as gym

gym.spaces.
@dataclass
class Modality:
    name: str
class ImageModality(Modality):
    name: str
    size: tuple[int, int]
    channels: int
class AudioModality(Modality):
    name: str
    duration: float = None
    sample_rate: int = None
class TextModality(Modality):
    name: str
    length: int = None
    vocab_size: int = None
class MouseModality(Modality):
    name: str
    loc_bounds: tuple[int, int, int, int] = None
    wheel_max



class AsyncEnv(gym.Env):
    
    env: gym.Env
    obs_queue: list[
    
    def __init__(self, env, *a, **kw):
        super().__init__(*a, **kw)
        self.env = env
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self):
        for q in self.
        self.observation_queue
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self, mode="human"):
        return self.env.render(mode=mode)

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed=seed)

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        return getattr(self.env, name)
