from dataclasses import dataclass
import functools
from typing import Callable
import typing
import ivy
import gymnasium as gym


Tensor = ivy.Tensor


T = typing.TypeVar('T')
class tree(typing.Generic[T]):
    @staticmethod
    def filter(fn: Callable[[T], bool], xs: T) -> T:
        ...

    @staticmethod
    def flatten(xs: T) -> T:
        ...

    @staticmethod
    def map(fn, *xs: T) -> T:
        ...

class Agent:
    def act(self, obs):
        pass


class tc:
    @staticmethod
    def enbode(obj):
        return obj


@dataclass
class Message:
    text: str
    time: float
    sender: str # 'user', 'agent', 'system'
@dataclass
class Observation:
    messages: list[Message]
    video: Tensor # [B, L=?, H, W, C]
    audio: Tensor # [B, L=?, C]
    cursor_loc: Tensor # [B, L=?, 2]
    mouse_wheel: Tensor # [B, L=?, 1]
    mouse_state: Tensor # [B, L=?, 3] # left, right, middle
    keyboard_state: Tensor # [B, L=?, NUM_KEYS=256]
@dataclass
class Action:
    message: Message
    cursor_move: Tensor # [B, L=?, 2]
    mouse_button_changes: Tensor # [B, L=?, 3] # left, right, middle
    keyboard_key_changes: Tensor # [B, L=?, NUM_KEYS=256]

class Computatrum(Agent):
    pass

class Computatrum0(Computatrum):

    lr_online = 1e-3
    lr_offline = 1e-3
    
    message_encoder = tc.Encoder()

    encoders: dict[str, Callable]
    decoders: dict[str, Callable]

    def __init__(self, env, hparams, *a, **kw):
        super().__init__(*a, **kw)
        
        self.env = env
        
        self.hparams = Computatrum0.HPARAM_DEFAULTS.copy()
        self.hparams.update(hparams)
        
        self._started = False
        
    def act(self, obs: Observation) -> Action:

        """NOTES
        
        the tc processes need to happen for both embeddings and text
        """
        
        if not self._started:
            self._started = True
            self.prev_action = Action(
                message=[],
                cursor_move=ivy.zeros([1, 1, 2]),
                mouse_button_changes=ivy.zeros([1, 1, 3]),
                keyboard_key_changes=ivy.zeros([1, 1, 256]),
            )

        ####### ## Early exits ##################################################
        ####### if self.enable_early_exits:
        #######     ### Action-oriented language processing
        #######     if self.fast_gpt.chat('is this urgent?', obs.messages):
        #######         self.enable_perception = False
        #######         self.enable_action = True
        #######     
        #######     ### Action-oriented vision processing
        #######     # This isnt really important atm
        
        ## Perception ##################################################
        if self.enable_perception:
            ### Undirected perception
            inputs = {
                'message': obs.messages,
                'video': obs.video,
                'audio': obs.audio,
                'cursor_loc': obs.cursor_loc,
                'mouse_wheel': obs.mouse_wheel,
                'mouse_state': obs.mouse_state,
                'keyboard_state': obs.keyboard_state,
                'prev_cursor_move': self.prev_action.cursor_move,
                'prev_mouse_button_changes': self.prev_action.mouse_button_changes,
                'prev_keyboard_key_changes': self.prev_action.keyboard_key_changes,
            }
            enc_inputs = {
                'message': self.message_encoder(inputs['message']),
                'video':
                'audio':
                'cursor_loc':
                'mouse_wheel':
                'mouse_state':
                'keyboard_state':
                'prev_cursor_move':
                'prev_mouse_button_changes':
                'prev_keyboard_key_changes':
            }
            
            ### Directed perception
            
            ### Sensorimotor fusion
        
        ## Introception ##################################################
        if self.enable_introception:
            
            ### Undirected associative memory retrieval
            
            ### Directed associative memory retrieval
        
        ## Integration ##################################################
        if self.enable_integration:
            
            ### Surprise
            
            ### Valency
            
            ### Arousal (learned)
            
            ### Fusion
            
            ### Curiosity
        
            ### Add new memories to VectorDB
        
        ## Action ##################################################
        if self.enable_action:
            
            
        ## Online Learning ##################################################
        if self.enable_online_learning:


input[t]
reconstruction[t]
encoding[t]
decoding[t]


signal_forier_recon_loss = HP(name='signal_forier_recon_loss', initial=0, size=())

signal_forier_recon_loss[t] = sum(
    (signal_recon_loss[t] + signal_recon_loss[t-1]) / 2
    for t in range(1, T)
)

signal_forier_pred[t+1] = signal_forier_pred[t] + signal_forier_recon_loss[t]

# meta-signals

## controllable meta-signals slowly adapt via gradient descent
lr[t]: tree[float, 2] # learning rate for each module for each layer
T_FFT[t] = 30 # however many timesteps back you need to get a reasonable FFT
T_IFFT_PREV[t] = 30 # how many timesteps to rollback the meta-signal IFFT
T_IFFT_FORE[t] = 30 # how many timesteps to rollforward the meta-signal IFFT

## uncontrollable meta-signals are hardcoded
surprise[t-1] = treesum(treemap(pred_immediate[t-2], immediate[t-1])) # cumulative surprise is just the root[0] value
activation_statistics[t-1]: Tree[Tensor, 2] = mean, std, skewness, kurtosis, etc for each layer for each module
gradient_statistics[t-1]: Tree[Tensor, 2] = mean, std, skewness, kurtosis, etc for each layer for each module
meta_signal_forier_modes[t-1] = treemap(meta_signals[t-1], lambda x: fft(x, n=3))

controllable_meta_signals = [T_FFT[t], lr[t]]
uncontrollable_meta_signals = {
  surprise[t-1], reward[t-1], 
  activation_statistics[t-1], gradient_statistics[t-1], activation_statistics_forier_modes[t-1], gradient_statistics_forier_modes[t-1],
  meta_signal_forier_modes[t-1]
}
meta_signals = [controllable_meta_signals, uncontrollable_meta_signals]

# internal state

top_down[t] = f_top_down(state[t-1])

immediate[t] = {
  'encodings': encodings[t],
  'meta_signals': meta_signals[t],
  'top_down': top_down[t],
}

stm: Multigraph

semantic_ltm: Multigraph
Episode = list[T_IMMEDIATE]
episodic_ltm: set[Episode]

### Note: Previous LTM states are not saved
ltm = (semantic_ltm, episodic_ltm)

# episodic ltm's are linked by agent, temporal, and semanticly

state[t] = (immediate[t], stm[t], ltm)


# reward module

## discounted signal reconstruction
## Even though immediate prediction, reconstruction, and replay are also performed, this uses a different technique so it may add value to the model
meta_signal_reconstruction_times = range(-T_IFFT_PREV, T_IFFT_FORE+1)
meta_signal_reconstruction = ifft(signal_forier_modes[t], signal_reconstruction_times)
meta_signal_reconstruction_error = meta_signal_forier_modes - meta_signal_reconstruction
meta_signal_reconstruction_discounter = MLP([
  PositionalEncoder(HP('D_META_SIGNAL_RECONSTRUCTOR_PE')),
  Linear(HP('D_META_SIGNAL_RECONSTRUCTOR_FC_1'), activation='relu'),
  Linear(T_IFFT_PREV+T_IFFT_FORE, activation='sigmoid'), # sigmoid to ensure output in (0, 1)
]) # I thought about flipping the gradients to make it want to learn confusing signals, but I'm not sure that's a good idea
meta_signal_reconstruction_discount[t] = meta_signal_reconstruction_discounter(meta_signal_reconstruction_times)
discounted_meta_signal_reconstruction_error = meta_signal_reconstruction_discount[t] * meta_signal_reconstruction_error
total_discounted_meta_signal_reconstruction_error = treesum(discounted_meta_signal_reconstruction_error, weighted=True)


# although fixed/learnable_reward looks like a flat dict, it might actually be hierarchically organized -- perhaps nodes appearing under more than one branch 
fixed_reward[t]: TREE = {
  'discounted_meta_signal_reconstruction_error': -total_discounted_meta_signal_reconstruction_error
  'immediate_prediction': 
}
learnable_reward[t]: Tree = {

}
reward_tree = Tree({ 'fixed_reward': fixed_rewards[t], 
                     'learnable_rewards': learnable_rewards[t] })
w_fixed_reward[t]: TREE.like(fixed_rewards, include_root=True) = operator-supplied parameters
w_learnable_reward[t]: TREE.like(fixed_rewards, include_root=True) = 
reward[t] = treesum(reward, weights={
  'fixed_reward': w_fixed_reward[t],
  'learnable_reward': w_learnable_reward[t],
})
total_fixed_reward = total_reward['fixed_reward']
total_learnable_reward = total_reward['learnable_reward'] 

# TODO: generate different reward signals for different modules, layers, and even neurons

pred_immediate[t] = world_model(state[t])

# TODO: Add Q-function for each reward signal, cumulatively