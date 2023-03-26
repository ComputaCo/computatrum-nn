import functools
from typing import Callable
import ivy


class Obs:
    conversation: Tensor # [B, L]
    video: Tensor # (batch, seq_len, channels, height, width)
    audio: Tensor
    mouse: Tensor


class AutonomousAgent(ivy.Module):

    # types
    # not complete
    # mainly for documentation purposes.
    # but useful to distinguish properties of T_INPUT/ENC/DEC
    # since they are not necesarily the same
    T_INPUT = tree[ivy.Tensor]
    T_INPUT_ENC = tree[ivy.Tensor] # k in T_INPUT_ENC -> k in T_IMMEDIATE
    T_INPUT_DEC = tree[ivy.Tensor] # k in T_INPUT_DEC -> k in T_INPUT
    T_IMMEDIATE = tree[ivy.Tensor]

    # i/o
    modalities[t] = list[Modality] # provides I/O information
    encoders: list[tuple[str, Callable]] # name, function
    decoders = list[tuple[str, Callable]] # name, function
    input[t]: T_INPUT
    reconstruction[t]: T_INPUT
    encoding[t]: T_INPUT_ENC = each encoder's encoding (# encoders <> # inputs)
    decoding[t]: T_INPUT_DEC = each decoder's decoding (# decoders <> # outputs)

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.env = env

    def _forward(self, inputs):
        _inputs = inputs
        _inputs = tree.filter(lambda p: p[1] is not None, _inputs.items())
        _inputs = tree.flatten(_inputs)




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