import gym
import gym.spaces


class TaskGuidedBehaviorDistilationEnvWrapper(gym.Env):
    """Used to train a student to imitate a teacher policy.

    During training, the student policy watches the teacher demonstrate.
    The student is rewarded based on its `task_eval` action similarity
    with the teacher policy's `task_eval` action.

    The teacher policy is supplied in the constructor. The student
    policy interacts with the environment using the `step` method.

    Both poilicies should include the environment modalities as well as:
    - a `task` observation space
    - an `task_eval` action space that the `loss_fn` can consume
    - a `task_eval_confidence` action space Box(0, 1, shape=[1])
    They should used `Dict` action / observation spaces.

    The `info` return value of the wrapped environment should be a dict.

    See README.md for the big picture.
    """

    def __init__(self, env, task,
                 teacher_policy, loss_fn,
                 task_space, task_eval_space,
                 c_external_reward=1.0, c_imitation=1.0):
        self.env = env
        self.task = task
        self.teacher_policy = teacher_policy
        self.loss_fn = loss_fn
        self._task_space = task_space
        self._task_eval_space = task_eval_space
        self.c_external_reward = c_external_reward
        self.c_imitation = c_imitation
        self._obs = None

    def reset(self):
        self._obs = self.env.reset()
        self._obs['task'] = self.task
        return self._obs

    def step(self, action):
        student_action = action
        del action
        teacher_action = self.teacher_policy(self.prev_obs)
        self._obs, external_reward, done, info = self.env.step(teacher_action)
        self._obs['task'] = self.task
        reward = self.c_external_reward * external_reward + (
            student_action['task_eval_confidence'] *
            teacher_action['task_eval_confidence'] *
            self.c_imitation *
            self.loss_fn(teacher_action['task_eval'],
                         student_action['task_eval']))
        info.update({'task': self.task})
        return self._obs, reward, done, info

    @property
    def observation_space(self):
        observation_space = self.env.observation_space
        observation_space.space.update({
            'task': self._task_space
        })
        return observation_space

    @property
    def action_space(self):
        action_space = self.env.action_space
        action_space.space.update({
            'task_eval': self._task_eval_space,
            'task_eval_confidence': gym.spaces.Box(0, 1, shape=[1])
        })
        return action_space
