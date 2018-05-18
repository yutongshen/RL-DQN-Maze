import pandas as pd
import numpy as np
import keras
from keras.models import Sequential, Model, clone_model
from keras.layers.core import Dense, Activation, Lambda
from keras.layers import Input, LSTM, TimeDistributed, RepeatVector, Reshape, Dropout, Bidirectional, Concatenate
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop

class RL:
    def __init__(self, Feature_size, Actions, Epsilon = .5, Gamma = .9, lr = .01, Memory_size = 200, replace_target_iter = 200):
        self.Epsilon             = Epsilon
        self.Gamma               = Gamma
        self.lr                  = lr
        self.Actions             = Actions
        self.Feature_size        = Feature_size
        self.Memory_size         = Memory_size
        self.replace_target_iter = replace_target_iter
        self.Memory              = None
        self.q_evaluation_model  = None
        self.q_target_model      = None
        self.mem_cnt             = 0
        self.learning_cnt        = 0
        self.history             = []
        
        self._init_memory()
        self._build_model()

    def _init_memory(self):
        self.Memory = np.zeros((self.Memory_size, 2 * self.Feature_size + 2))

    def _build_model(self):
        # Q-Evaluation Model
        HIDDEN_SIZE = 50
        inputs = Input(shape=(self.Feature_size,))
        fc_1   = Dense(HIDDEN_SIZE, activation='relu')(inputs)
        fc_2   = Dense(len(self.Actions))(fc_1)
        self.q_evaluation_model = Model(inputs, fc_2)
        rmsp = RMSprop(lr=self.lr)
        self.q_evaluation_model.compile(loss='mse', optimizer=rmsp)

    def _copy_model(self):
        self.q_target_model = keras.models.clone_model(self.q_evaluation_model)
        self.q_target_model.set_weights(self.q_evaluation_model.get_weights())

    def store_observation(self, s, a, r, s_):
        self.Memory[self.mem_cnt % self.Memory_size, :] = np.hstack((list(s), [self.Actions.index(a), r], list(s_)))
        self.mem_cnt += 1

    def actor(self, observation):
        if np.random.uniform(0, 1) > self.Epsilon:
            action = np.random.choice(self.Actions)
        else:
            observation = np.array(observation)
            observation = observation[np.newaxis, :]
            q_value = self.q_evaluation_model.predict(observation)
            action = self.Actions[q_value.argmax()]

        return action

    def learn(self):
        if self.learning_cnt % self.replace_target_iter == 0:
            self._copy_model()

        if self.mem_cnt < self.Memory_size:
            s_batch  = self.Memory[:self.mem_cnt, :self.Feature_size]
            s_batch_ = self.Memory[:self.mem_cnt, -self.Feature_size:]
        else:
            s_batch  = self.Memory[:, :self.Feature_size]
            s_batch_ = self.Memory[:, -self.Feature_size:]

        reward    = self.Memory[:, self.Feature_size + 1]
        max_q     = self.q_target_model.predict(s_batch_).max(axis=1)
        q_predict = self.q_evaluation_model.predict(s_batch)
        q_target = np.copy(q_predict)
        q_target[np.arange(len(q_target)), self.Memory[:, self.Feature_size].astype(np.int32)] = reward + self.Gamma * max_q
        mask = np.random.permutation(len(s_batch))

        report = self.q_evaluation_model.fit(s_batch[mask], q_target[mask], batch_size=len(s_batch), verbose=0)

        if self.Epsilon < .9:
            self.Epsilon += .001

        self.history.append(report.history['loss'])

        self.learning_cnt += 1
