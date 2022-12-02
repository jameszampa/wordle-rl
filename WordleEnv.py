import os
import gym
import numpy as np
from gym import spaces
from tensorboardX import SummaryWriter

VALID_WORDLE_WORDS_FILE = 'valid-wordle-words.txt'

def getWords():
    words = []
    with open(VALID_WORDLE_WORDS_FILE, 'r') as f:
        for line in f.readlines():
            words.append(line.replace('\n', ''))
    return words

WORD_LIST = getWords()
WORD_LENGTH = len(WORD_LIST[0])
NUM_LETTERS_IN_ALPHABET = 26
MAX_TRIES = 6

class WordleEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, logdir):
        super(WordleEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(len(WORD_LIST))
        # Example for using image as input:
        self.observation_space = spaces.Dict({ 'letters': spaces.Box(low=-1, high=NUM_LETTERS_IN_ALPHABET, shape=(MAX_TRIES, WORD_LENGTH), dtype=np.uint8),
                                               'letter_feedback': spaces.Box(low=-1, high=2, shape=(MAX_TRIES, WORD_LENGTH,), dtype=np.uint8)
        })
        self.tries = 0
        self.logger = SummaryWriter(os.path.join(logdir, 'PPO_0'))
        self.games = 0

    def _new_answer(self):
        self.answer_idx = np.random.randint(0, len(WORD_LIST))
        self.answer = WORD_LIST[self.answer_idx]

    def _action_num_to_letter(self, number):
        return chr(97 + number)

    def _action_letter_to_number(self, letter):
        return ord(letter) - 97

    def _update_observation(self, action):
        for idx_g, number_g in enumerate(action):
            self.observation['letters'][self.tries][idx_g] = number_g

        for idx_g, number_g in enumerate(action):
            if self._action_num_to_letter(number_g) == self.answer[idx_g]:
                self.observation['letter_feedback'][self.tries][idx_g] = 2
        
        for idx_g, number_g in enumerate(action):
            if not self.observation['letter_feedback'][self.tries][idx_g] is None:
                continue
            if self._action_num_to_letter(number_g) in self.answer:
                num_letter_answer = 0
                for char in self.answer:
                    if char == self._action_num_to_letter(number_g):
                        num_letter_answer += 1
                #print(num_letter_answer)
                num_letter_guess = 0
                for num2 in action[:idx_g + 1]:
                    if self._action_num_to_letter(num2) == self._action_num_to_letter(number_g):
                        num_letter_guess += 1
                #print(num_letter_guess)
                if num_letter_guess <= num_letter_answer:
                    self.observation['letter_feedback'][self.tries][idx_g] = 1
                else:
                    self.observation['letter_feedback'][self.tries][idx_g] = 0
            else:
                self.observation['letter_feedback'][self.tries][idx_g] = 0

    def _give_mid_game_reward(self):
        reward = 0
        for val in self.observation['letter_feedback'][self.tries]:
            if val != -1:
                reward += val / 10 * (MAX_TRIES - self.tries)
        return reward

    def step(self, action):
        reward = 0
        info = {}
        action = [self._action_letter_to_number(letter) for letter in WORD_LIST[int(action)]]
        guess = ""
        for number in action:
            guess += self._action_num_to_letter(number)
        if not guess in WORD_LIST:
            return self.observation, reward, self.done, info
        self._update_observation(action)
        self.isCorrect = True
        for val in self.observation['letter_feedback'][self.tries]:
            if val != 2:
                self.isCorrect = False
        if self.tries >= (MAX_TRIES - 1):
            self.done = True
            if self.isCorrect:
                reward = 1
            else:
                reward = -1 + self._give_mid_game_reward()
            self.games += 1
        elif self.isCorrect:
            self.done = True
            reward = MAX_TRIES - self.tries
            self.games += 1
        else:
            reward = 0
        self.tries += 1
        #print(self.observation)
        return self.observation, reward, self.done, info

    def reset(self):
        self._new_answer()
        self.observation = { 'letters': np.ones(shape=(MAX_TRIES, WORD_LENGTH)) * -1,
                             'letter_feedback': np.ones(shape=(MAX_TRIES, WORD_LENGTH)) * -1
        }
        self.done = False
        self.logger.add_scalar('tries', self.tries, self.games)
        self.tries = 0
        return self.observation

    def render(self, mode='human'):
        pass

    def close (self):
        pass