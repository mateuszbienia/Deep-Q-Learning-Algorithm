import tensorflow as tf
import numpy as np
from .ReplayMemory import ReplayMemory

from typing import List


class DQN():
    def __init__(self, model, output_size: int, gamma: float = 0.99, lr: float = 1e-4, batch_size: int = 32, min_memory: int = 100, max_memory: int = 10_000, preprocess_function=None):
        self.model = model
        self.gamma = gamma
        self.optimizer = tf.optimizers.Adam(lr)
        self.memory = ReplayMemory(min_memory, max_memory, batch_size)
        self.num_actions = output_size

        if preprocess_function is None:
            self.preprocess_fun = lambda state: state
        else:
            self.preprocess_fun = preprocess_function

    def train(self, TargetNet) -> None:
        samples = self.memory.sample()
        if samples == None:
            return

        samples = np.asarray(samples, dtype=object)
        states = np.asarray([self.preprocess_state(sample)
                            for sample in samples[:, 0]], dtype=object)
        actions = np.asarray(samples[:, 1], dtype=np.int32)
        rewards = np.asarray(samples[:, 2], dtype=np.float32)
        next_states = np.asarray([self.preprocess_state(sample)
                                 for sample in samples[:, 3]], dtype=object)
        dones = np.asarray(samples[:, 4], dtype=bool)

        next_states = np.asarray(
            [next_states[i] for i in range(len(next_states)) if not dones[i]])
        q_val = np.max(TargetNet.predict(next_states), axis=1)
        real_rewards = np.zeros(len(samples))
        j = 0
        for i in range(len(samples)):
            if dones[i]:
                real_rewards[i] = rewards[i]
            else:
                real_rewards[i] = rewards[i] + self.gamma*q_val[j]
                j += 1
        with tf.GradientTape() as tape:
            selected_action_values = tf.math.reduce_sum(
                self.predict(states) * tf.one_hot(actions, self.num_actions), axis=1)
            loss = tf.math.reduce_sum(
                tf.square(real_rewards - selected_action_values))

        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

    def predict(self, input_state):
        return self.model(np.atleast_2d(input_state.astype('float32')))

    def get_action(self, state, moves: List[int], epsilon: float = 0) -> int:
        if np.random.random() < epsilon:
            return np.random.choice(moves)
        else:
            return self._get_best_move(state, moves)

    def preprocess_state(self, state):
        return self.preprocess_fun(state)

    def copy_weights(self, TrainNet):
        variables1 = self.model.trainable_variables
        variables2 = TrainNet.model.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())

    def save_weights(self, path: str):
        self.model.save_weights(path)

    def _get_best_move(self, state, moves: List[int]) -> int:
        prediction = self.predict(self.preprocess_state(state)).numpy()[0]
        for i in range(len(prediction)):
            if i not in moves:
                prediction[i] = -np.inf
        return np.argmax(prediction)
