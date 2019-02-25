import random
import numpy as np
import math

class Memory:
    def __init__(self, max_memory):
        self._max_memory = max_memory
        self._samples = []

    def add_sample(self, sample):
        self._samples.append(sample)
        if len(self._samples) > self._max_memory:
            self._samples.pop(0)

    def sample(self, no_samples):
        if no_samples > len(self._samples):
            return random.sample(self._samples, len(self._samples))
        else:
            return random.sample(self._samples, no_samples)

class GameRunner:
    def __init__(self, sess, model, env, memory, max_eps, min_eps,
                 decay, gamma, render=True):
        self._sess = sess
        self._env = env
        self._model = model
        self._memory = memory
        self._render = render
        self._max_eps = max_eps
        self._min_eps = min_eps
        self._decay = decay
        self._gamma = gamma
        self._eps = self._max_eps
        self._steps = 0
        self._reward_store = []
    
    def run(self):
        state = self._env.reset()
        tot_reward = 0
        count = 0
        while True:
            if self._render:
                self._env.render()

            action = self._choose_action(state)
            next_state, reward, done, info = self._env.step(action)
            # is the game complete? If so, set the next state to
            # None for storage sake
            if done:
                next_state = None

            self._memory.add_sample((state, action, reward, next_state))
            self._replay()

            # exponentially decay the eps value
            self._steps += 1
            self._eps = self._min_eps + (self._max_eps - self._min_eps) \
                                    * math.exp(-self._decay * self._steps)

            # move the agent to the next state and accumulate the reward
            state = next_state
            tot_reward += reward

            # if the game is done, break the loop
            count += 1
            if done or count > 200:
                self._reward_store.append(tot_reward)
                break
            
        print("Step {}, Total reward: {}, Eps: {}, took {} loops".format(self._steps, tot_reward, self._eps, count))

    def _choose_action(self, state):
        if random.random() < self._eps:
            random_action =  self._env.get_correct_action()
            return random_action
        else:
            chosen_action = np.argmax(self._model.predict_one(state, self._sess))
            print("chosen_action", chosen_action)
            return chosen_action

    def _replay(self):
        batch = self._memory.sample(self._model.batch_size)
        states = np.array([val[0] for val in batch])
        next_states = []
        for val in batch:
            if val[3] is None:
                next_state = self._env.env.get_empty()
            else:
                next_state  = val[3]
            next_states.append(next_state)
            
        next_states = np.array(next_states)
        # predict Q(s,a) given the batch of states
        q_s_a = self._model.predict_batch(states, self._sess)
        # predict Q(s',a') - so that we can do gamma * max(Q(s'a')) below
        q_s_a_d = self._model.predict_batch(next_states, self._sess)
        # setup training arrays
        x = np.zeros((len(batch), self._model.num_states))
        y = np.zeros((len(batch), self._model.num_actions))
        for i, b in enumerate(batch):
            state, action, reward, next_state = b[0], b[1], b[2], b[3]
            # get the current q values for all actions in state
            current_q = q_s_a[i]
            # update the q value for action
            if next_state is None:
                # in this case, the game completed after action, so there is no max Q(s',a')
                # prediction possible
                current_q[action] = reward
            else:
                current_q[action] = reward + self._gamma * np.amax(q_s_a_d[i])
            x[i] = state.reshape(self._env.get_num_states())
            y[i] = current_q
        self._model.train_batch(self._sess, x, y)