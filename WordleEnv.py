import os
import gym
import cv2
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

    def __init__(self, logdir, harsh=False):
        super(WordleEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.MultiDiscrete([NUM_LETTERS_IN_ALPHABET for _ in range(WORD_LENGTH)])
        # Example for using image as input:
        self.observation_space = spaces.Dict({ 'letters': spaces.Box(low=-1, high=NUM_LETTERS_IN_ALPHABET, shape=(MAX_TRIES, WORD_LENGTH), dtype=np.uint8),
                                               'letter_feedback': spaces.Box(low=-1, high=2, shape=(MAX_TRIES, WORD_LENGTH,), dtype=np.uint8)
        })
        self.tries = 0
        self.logger = SummaryWriter(os.path.join(logdir, 'PPO_0'))
        self.games = 0
        self.answer = None
        self.guesses = []
        self.guess_rewards = []
        self.games_won = 0
        self.harsh = harsh
        self.possible_answers = WORD_LIST

    def _new_answer(self):
        self.answer_idx = np.random.randint(0, len(WORD_LIST))
        self.answer = WORD_LIST[self.answer_idx]
        self.possible_answers = WORD_LIST.copy()

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
            if self.observation['letter_feedback'][self.tries][idx_g] == 2:
                continue
            #print(self._action_num_to_letter(number_g))
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
        # reward = 0
        # for val in self.observation['letter_feedback'][self.tries]:
        #     assert val != -1
        #     reward += val / 10
        return self._calculate_infomation(self.tries)

    def _calculate_infomation(self, attempt_idx):
        for word_idx, (number, feedback) in enumerate(zip(self.observation['letters'][attempt_idx], self.observation['letter_feedback'][attempt_idx])):
            assert feedback != -1
            assert number != -1
            if feedback == 0:
                for word in self.possible_answers:
                    if self._action_num_to_letter(int(number)) in word:
                        self.possible_answers.remove(word)
            elif feedback == 1:
                for word in self.possible_answers:
                    if not self._action_num_to_letter(int(number)) in word:
                        self.possible_answers.remove(word)
            elif feedback == 2:
                for word in self.possible_answers:
                    if self._action_num_to_letter(int(number)) != word[word_idx]:
                        self.possible_answers.remove(word)
        p = len(self.possible_answers) / len(WORD_LIST)
        i = np.log2(1 / p)
        return i

    def step(self, action):
        self.reward = 0
        info = {}
        #action = [self._action_letter_to_number(letter) for letter in WORD_LIST[int(action)]]
        guess = ""
        for number in action:
            guess += self._action_num_to_letter(number)
        if not guess in WORD_LIST:
            self.reward -= 0.1
            if self.harsh:
                return self.observation, self.reward, self.done, info
        if guess in self.guesses:
            self.reward -= 0.1
            if self.harsh:
                return self.observation, self.reward, self.done, info
        self._update_observation(action)
        self.isCorrect = True
        for val in self.observation['letter_feedback'][self.tries]:
            if val != 2:
                self.isCorrect = False
        if self.tries >= (MAX_TRIES - 1):
            self.done = True
            if self.isCorrect:
                self.reward = 100
                self.games_won += 1
            else:
                if guess in WORD_LIST and not guess in self.guesses:
                    self.reward = self._give_mid_game_reward()
            self.games += 1
        elif self.isCorrect:
            self.done = True
            self.reward = (MAX_TRIES - self.tries) * 100 + 100
            self.games += 1
            self.games_won += 1
        else:
            if guess in WORD_LIST and not guess in self.guesses:
                self.reward = self._give_mid_game_reward()
        self.guesses.append(guess)
        self.tries += 1
        self.guess_rewards.append(self.reward)
        #print(self.observation)
        return self.observation, self.reward, self.done, info

    def reset(self):
        self._new_answer()
        self.observation = { 'letters': np.ones(shape=(MAX_TRIES, WORD_LENGTH)) * -1,
                             'letter_feedback': np.ones(shape=(MAX_TRIES, WORD_LENGTH)) * -1
        }
        self.done = False
        self.logger.add_scalar('tries', self.tries, self.games)
        self.tries = 0
        self.guesses = []
        self.guess_rewards = []
        return self.observation

    def render(self, mode='human'):
        height = 500
        width = 500
        channels = 3
        img = np.zeros((height,width,channels), dtype=np.uint8)
        img[:] = (int(255 * 0.07), int(255 * 0.07), int(255 * 0.07))
        box_size = 60
        x_offset = 40
        y_offset = 50
        box_gap_width = 10
        for row in range(MAX_TRIES):
            for col in range(WORD_LENGTH):
                if self.observation['letters'][row][col] == -1:
                    img = cv2.rectangle(img,
                                        (x_offset + (col * (box_size + box_gap_width)), y_offset + (row * (box_size + box_gap_width))),
                                        (x_offset + (col * (box_size + box_gap_width)) + box_size, y_offset + (row * (box_size + box_gap_width)) + box_size),
                                        (int(255 * 0.22), int(255 * 0.22), int(255 * 0.22)),
                                        2)
                    continue
                if self.observation['letter_feedback'][row][col] == 2:
                    box_color = (int(255 * 0.32), int(255 * 0.55), int(255 * 0.30))
                elif self.observation['letter_feedback'][row][col] == 1:
                    box_color = (int(255 * 0.23), int(255 * 0.62), int(255 * 0.71))
                else:
                    box_color = (int(255 * 0.22), int(255 * 0.22), int(255 * 0.22))
                img = cv2.rectangle(img,
                                    (x_offset + (col * (box_size + box_gap_width)), y_offset + (row * (box_size + box_gap_width))),
                                    (x_offset + (col * (box_size + box_gap_width)) + box_size, y_offset + (row * (box_size + box_gap_width)) + box_size),
                                    box_color,
                                    -1)
                img = cv2.putText(img,
                                  self._action_num_to_letter(int(self.observation['letters'][row][col])).upper(),
                                  (int(x_offset + (col * (box_size + box_gap_width)) + (box_size / 2) - 12), int(y_offset + (row * (box_size + box_gap_width)) + (box_size / 2)) + 10),
                                  cv2.FONT_HERSHEY_SIMPLEX,
                                  1,
                                  (255, 255, 255),
                                  2)
            if len(self.guess_rewards) > row:
                img = cv2.putText(img,
                                "{:.2f}".format(self.guess_rewards[row]),
                                (int(x_offset + (WORD_LENGTH * (box_size + box_gap_width))), int(y_offset + (row * (box_size + box_gap_width)) + (box_size / 2) + 10)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (int(255 * 0.29), int(255 * 0.12), int(255 * 0.76)),
                                2)
        img = cv2.putText(img,
                          f"Answer: {self.answer}",
                          (int(width / 2) - 100, 35),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          1,
                          (255, 255, 255),
                          2)
        cv2.imshow(f"WordleRL", img)

    def close (self):
        pass