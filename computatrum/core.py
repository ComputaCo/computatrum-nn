from dataclasses import dataclass
import functools
import inspect
import itertools
import math
from random import randint
from threading import Thread
from typing import Callable
import typing
import ivy
import gymnasium as gym
from PIL import Image
from tesserocr import PyTessBaseAPI, RIL
import numpy as np
import multiprocessing as mp
import torch
import torchaudio
import transformers

from hoists import declare, hoists

Tensor = ivy.Tensor


T = typing.TypeVar("T")


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

    def train(self, traj):
        pass


class _Buffered:

    _input_queue: mp.Queue
    _output_queue: mp.Queue

    _input_lock: mp.Lock
    _output_lock: mp.Lock

    _process: mp.Process

    def __init__(self, fn, wait=False):
        self._input_queue = mp.Queue()
        self._output_queue = mp.Queue()
        self._input_lock = mp.Lock()
        self._output_lock = mp.Lock()

        self._process = mp.Process(target=self._loop, args=(self,), daemon=True)
        self.fn = fn
        self.wait = wait

    def __call__(self, *args, **kwargs):
        with self._inputs_lock:
            self._input_queue.put((args, kwargs))
        with self._output_lock:  # required to prevent sampling after .empty() but before .put()
            if self.wait:
                return self._output_queue.get()
            else:
                if self._output_queue.qsize() > 0:
                    return self._output_queue.get()
                else:
                    return None

    def __del__(self):
        self.stop()
        super().__del__()

    def start(self):
        if not self._process.is_alive():
            self._process.start()

    def stop(self):
        if not self._process.is_alive():
            return
        self._process.join()

    def _loop(self):
        while not self._process.is_alive():
            self._step()

    def _step(self):
        with self._inputs_lock:
            args, kwargs = self.input_queue.get()
            self.input_queue.empty()
        output = self.fn(*args, **kwargs)
        with self._output_lock:
            self.outputs.empty()
            self.outputs.put(output)


def buffered(fn, wait=False):
    return declare(
        _Buffered(fn, wait=wait),
        frames_above=2,
        name=None if fn.__name__ == "<lambda>" else fn.__name__,
    )


def skipnone(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if args[0] is None:
            return None
        return fn(*args, **kwargs)

    return wrapper


class tc:
    @staticmethod
    def enbode(obj):
        return obj


@dataclass
class Observation:
    input_text: list[str]  # [B]
    audio: Tensor  # [B, L=?, C]
    screen: Tensor  # [B, L=?, H, W, C]
    screen_text: list[OCRProcessor.Result]  # [B]
    cursor_loc: Tensor  # [B, L=?, 2]
    mouse_wheel: Tensor  # [B, L=?, 1]
    mouse_state: Tensor  # [B, L=?, 3] # left, right, middle
    keyboard_state: Tensor  # [B, L=?, NUM_KEYS=256]


@dataclass
class Action:
    output_text: list[str]  # [B]
    cursor_movement: Tensor  # [B, L=?, 2]
    mouse_button_changes: Tensor  # [B, L=?, NUM_MOUSE_BUTTONS=3] # left, right, middle
    mouse_wheel_movement: Tensor  # [B, L=?, 1]
    keyboard_key_changes: Tensor  # [B, L=?, NUM_KEYS=256]


class Computatrum(Agent):

    NUM_MOUSE_BUTTONS = 3
    NUM_KEYS = 256

    enable_perception = True
    enable_introspection = True
    enable_integration = True
    enable_action = True
    enable_online_learning = True
    enable_offline_learning = False

    user_conversation: list[Message]

    def __init__(self) -> None:
        self._setup = False

    def setup(self, batch_size):
        self._setup = True

    @hoists
    def act(self, obs: Observation) -> Action:
        B = obs.text.shape[0]  # batch size

        if not self._setup:
            self.setup(B)

        prev_action = declare(
            "prev_action",
            lambda: Action(
                output_text=["" for _ in range(B)],
                cursor_movement=ivy.zeros((B, 1, 2)),
                mouse_button_changes=ivy.zeros((B, 1, Computatrum.NUM_MOUSE_BUTTONS)),
                mouse_wheel_movement=ivy.zeros((B, 1, 1)),
                keyboard_key_changes=ivy.zeros((B, 1, Computatrum.NUM_KEYS)),
            ),
        )

        ## Perception ##################################################
        if self.enable_perception:

            class undirected_perception:
                ### Undirected perception
                inputs = {
                    "input_text": obs.input_text,
                    "audio": obs.audio,
                    "screen": obs.screen,
                    "screen_text": obs.screen_text,
                    "cursor_loc": obs.cursor_loc,
                    "mouse_wheel": obs.mouse_wheel,
                    "mouse_state": obs.mouse_state,
                    "keyboard_state": obs.keyboard_state,
                    "output_text": prev_action.output_text,
                    "cursor_movement": prev_action.cursor_movement,
                    "mouse_button_changes": prev_action.mouse_button_changes,
                    "mouse_wheel_movement": prev_action.mouse_wheel_movement,
                    "keyboard_key_changes": prev_action.keyboard_key_changes,
                }

                def build_encoders():
                    ## all text
                    text_encoder = tc.Encoder.for_inputs("")

                    ## audio
                    # https://huggingface.co/facebook/wav2vec2-base-960h
                    wav2vec2_processor = Wav2Vec2Processor.from_pretrained(
                        "facebook/wav2vec2-base-960h"
                    )
                    wav2vec2 = Wav2Vec2ForCTC.from_pretrained(
                        "facebook/wav2vec2-base-960h"
                    )

                    ## screen
                    screen_encoder = TODO()

                    ## cursor
                    cursor_loc_enc = tc.Encoder.for_inputs(
                        {
                            "cursor_loc",
                            "mouse_wheel",
                            "mouse_state",
                            "cursor_movement",
                            "mouse_button_changes",
                            "mouse_wheel_movement",
                        }
                    )

                    # keyboard
                    keyboard_state_enc = tc.Encoder.for_inputs(
                        {"keyboard_state" "keyboard_key_changes"}
                    )

                    return {
                        "input_text": text_encoder,
                        "audio": audio_encoder,
                        "screen": screen_encoder,
                        "screen_text": text_encoder,
                        "mouse": mouse_encoder,
                        "keyboard": keyboard_encoder,
                        "output_text": text_encoder,
                    }

                encoders = declare(
                    build_encoders, name="unsupervised_perception_encoders"
                )

                encoded_inputs = {
                    "input_text": buffered(skipnone(encoders["input_text"]))(
                        input["input_text"]
                    ),
                    "audio": None,
                    "screen": None,
                    "screen_text": None,
                    "mouse": None,
                    "keyboard": None,
                    "output_text": None,
                }

                if self.enable_online_learning:
                    decoders = {}
                    reconstructed_inputs = {
                        name: buffered(skipnone(lambda val: decoder(val)))(
                            encoded_inputs[name]
                        )
                        for name, decoder in decoders.items()
                    }

            ### Directed perception
            ## TODO: visual chat gpt

            ### Sensorimotor fusion

        ## Introception ##################################################
        if self.enable_introception:
            pass

            ### Undirected associative memory retrieval

            ### Directed associative memory retrieval

        ## Integration ##################################################
        if self.enable_integration:

            ### The big loop
            def build_integration_core_cell():
                swift_former_attention_layer = SwiftFormer.from_pretrained().layers[
                    "1:6"
                ]  # TODO: use swift former because it is fast and efficient
                lm_attention_layer = (
                    TODO()
                )  # TODO: find a good language model pretrained attention layer
                hopfield_network = (
                    TODO()
                )  # TODO: find a good floating point valued hopfield network
                mlp = (
                    TODO()
                )  # TODO: freshly inited mlp (input) -> (output, x1a, x1b, x2a)

                def integration_core_cell(input, recurrent):

                    x1a = swift_former_attention_layer(
                        query=recurrent["q1a"], key=input, value=input
                    )
                    x1b = lm_attention_layer(
                        query=recurrent["q1b"], key=input, value=input
                    )
                    x1 = concat(x1a, x1b)
                    x2a = hopfield_network(x1, recurrent=recurrent["x2a"])
                    x2 = x2a + x1
                    output, x1a, x1b, x2a = mlp(x2)

                    return output, {"q1a": x1a, "q1b": x1b, "x2a": x2a}

            integration_core_cell = declare(
                build_integration_core_cell, name="integration_core_cell"
            )

            ### Surprise

            ### Valency

            ### Arousal (learned)

            ### Fusion

            ### Curiosity

            ### Add new memories to VectorDB

        ## Action ##################################################

        if not self.enable_action:
            action = Action(
                text=[],
                cursor_movement=ivy.zeros((B, 1, 2)),
                mouse_button_changes=ivy.zeros((B, 1, Computatrum.NUM_MOUSE_BUTTONS)),
                mouse_wheel_movement=ivy.zeros((B, 1, 1)),
                keyboard_key_changes=ivy.zeros((B, 1, Computatrum.NUM_KEYS)),
            )
        if self.enable_action:
            pass

        ## Online Learning ##################################################
        if self.enable_online_learning:
            pass

        ## Offline Learning ##################################################
        if self.enable_offline_learning:
            pass

        self.prev_action = action
        return action


'''
    encoders: dict[str, Callable]
    decoders: dict[str, Callable]
    
    enable_observation
    enable_introception

    def __init__(self, env, *a, **kw):
        super().__init__(*a, **kw)
        
        self.env = env
        
        self._started = False
        
    def act(self, obs: Observation) -> Action:

        """NOTES
        
        the tc processes need to happen for both embeddings and text
        """
        


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
'''
