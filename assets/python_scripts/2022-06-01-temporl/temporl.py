import copy
from collections import defaultdict

import numpy as np
from typing import Tuple
import time

from js import (console, document, devicePixelRatio, ImageData, Uint8ClampedArray,
                CanvasRenderingContext2D as Context2d, setTimeout, clearTimeout)
from pyodide.ffi import create_once_callable, create_proxy

LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3

reward_queue = []

ID = None
animation_starte=False

from typing import Optional


def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.random()).argmax()


class Discrete:
    def __init__(self, nActions):
        self.nActions = nActions
        self.n = nActions


class DiscreteEnv:

    """
    Based on the original OpenAI Gym implementation
    """

    def __init__(self, nS, nA, P, isd):
        self.P = P
        self.isd = isd
        self.lastaction = None  # for rendering
        self.nS = nS
        self.nA = nA
        self.np_random = np.random.RandomState()
        self.s = None
        self.prev_s = None

        self.action_space = Discrete(self.nA)
        self.observation_space = Discrete(self.nS)

    def reset(self, seed: Optional[int] = None):
        self.prev_s = None
        self.s = categorical_sample(self.isd, self.np_random)
        self.lastaction = None
        return int(self.s)

    def step(self, a):
        self.prev_s = int(self.s)
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d = transitions[i]
        self.s = s
        self.lastaction = a
        return (int(s), r, d, {"prob": p})


class GridCore(DiscreteEnv):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, shape: Tuple[int] = (5, 10), start: Tuple[int] = (0, 0),
                 goal: Tuple[int] = (0, 9), max_steps: int = 1000,
                 percentage_reward: bool = False, no_goal_rew: bool = False):
        try:
            self.shape = self._shape
        except AttributeError:
            self.shape = shape
        self.nS = np.prod(self.shape, dtype=int)  # type: int
        self.nA = 4
        self.start = start
        self.goal = goal
        self.max_steps = max_steps
        self._steps = 0
        self._pr = percentage_reward
        self._no_goal_rew = no_goal_rew
        self.total_steps = 0

        P = self._init_transition_probability()

        # We always start in state (3, 0)
        isd = np.zeros(self.nS)
        isd[np.ravel_multi_index(start, self.shape)] = 1.0

        super(GridCore, self).__init__(self.nS, self.nA, P, isd)

    def step(self, a):
        self._steps += 1
        s, r, d, i = super(GridCore, self).step(a)
        if self._steps >= self.max_steps:
            d = True
            i['early'] = True
        self.total_steps += 1
        return s, r, d, i

    def reset(self):
        self._steps = 0
        return super(GridCore, self).reset()

    def _init_transition_probability(self):
        raise NotImplementedError

    def _check_bounds(self, coord):
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def map_output(self, s, pos):
        if self.s == s:
            output = " x "
        elif pos == self.goal:
            output = " T "
        else:
            output = " o "
        return output

    def map_control_output(self, s, pos):
        return self.map_output(s, pos)

    def map_with_inbetween_goal(self, s, pos, in_between_goal):
        return self.map_output(s, pos)

    def render(self, mode='human', close=False, in_control=None, in_between_goal=None):
        self._render(mode, close, in_control, in_between_goal)


class FallEnv(GridCore):
    _pits = []

    def __init__(self, act_fail_prob: float = 1.0, **kwargs):
        self.afp = act_fail_prob
        self._pits = copy.deepcopy(self._raw_pits)
        super(FallEnv, self).__init__(**kwargs)

    def _calculate_transition_prob(self, current, delta, prob):
        transitions = []
        for d, p in zip(delta, prob):
            new_position = np.array(current) + np.array(d)
            new_position = self._check_bounds(new_position).astype(int)
            new_state = np.ravel_multi_index(tuple(new_position), self.shape)
            reward = 0
            is_done = False
            if tuple(new_position) == self.goal:
                reward = 1.0
                is_done = True
            elif new_state in self._pits:
                reward = -1.
                is_done = True
            transitions.append((p, new_state, reward, is_done))
        return transitions

    def _init_transition_probability(self):
        for idx, p in enumerate(self._pits):
            self._pits[idx] = np.ravel_multi_index(p, self.shape)
        # Calculate transition probabilities
        P = {}
        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            P[s] = {a: [] for a in range(self.nA)}
            other_prob = self.afp / 3.
            tmp = [[UP, DOWN, LEFT, RIGHT],
                   [DOWN, LEFT, RIGHT, UP],
                   [LEFT, RIGHT, UP, DOWN],
                   [RIGHT, UP, DOWN, LEFT]]
            tmp_dirs = [[[-1, 0], [1, 0], [0, -1], [0, 1]],
                        [[1, 0], [0, -1], [0, 1], [-1, 0]],
                        [[0, -1], [0, 1], [-1, 0], [1, 0]],
                        [[0, 1], [-1, 0], [1, 0], [0, -1]]]
            tmp_pros = [[1 - self.afp, other_prob, other_prob, other_prob],
                        [1 - self.afp, other_prob, other_prob, other_prob],
                        [1 - self.afp, other_prob, other_prob, other_prob],
                        [1 - self.afp, other_prob, other_prob, other_prob], ]
            for acts, dirs, probs in zip(tmp, tmp_dirs, tmp_pros):
                P[s][acts[0]] = self._calculate_transition_prob(position, dirs, probs)
        return P

    def map_output(self, s, pos):
        if self.s == s:
            output = " * "
        elif pos == self.goal:
            output = " X "
        elif s in self._pits:
            output = " . "
        else:
            output = " o "
        return output

    def map_control_output(self, s, pos):
        if self.s == s:
            return " * "
        else:
            return self.map_output(s, pos)

    def map_with_inbetween_goal(self, s, pos, in_between_goal):
        if s == in_between_goal:
            return " x "
        else:
            return self.map_output(s, pos)


class Bridge6x10Env(FallEnv):
    _pits = [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7],
             [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7],
             [4, 2], [4, 3], [4, 4], [4, 5], [4, 6], [4, 7],
             [5, 2], [5, 3], [5, 4], [5, 5], [5, 6], [5, 7]]
    _raw_pits = [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7],
             [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7],
             [4, 2], [4, 3], [4, 4], [4, 5], [4, 6], [4, 7],
             [5, 2], [5, 3], [5, 4], [5, 5], [5, 6], [5, 7]]
    _shape = (6, 10)


class LargeField(FallEnv):
    _pits = []
    _raw_pits = []
    _shape = (15, 30)


class SmallField(FallEnv):
    _pits = []
    _raw_pits = []
    _shape = (10, 20)


class Pit6x10Env(FallEnv):
    _pits = [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7],
             [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7],
             [2, 2], [2, 3], [2, 4], [2, 5], [2, 6], [2, 7]]
    _raw_pits = [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7],
             [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7],
             [2, 2], [2, 3], [2, 4], [2, 5], [2, 6], [2, 7]]
    _shape = (6, 10)


class ZigZag6x10(FallEnv):
    _pits = [[0, 2], [0, 3],
             [1, 2], [1, 3],
             [2, 2], [2, 3],
             [3, 2], [3, 3],
             [5, 7], [5, 6],
             [4, 7], [4, 6],
             [3, 7], [3, 6],
             [2, 7], [2, 6],
             ]
    _raw_pits = [[0, 2], [0, 3],
             [1, 2], [1, 3],
             [2, 2], [2, 3],
             [3, 2], [3, 3],
             [5, 7], [5, 6],
             [4, 7], [4, 6],
             [3, 7], [3, 6],
             [2, 7], [2, 6],
             ]
    _shape = (6, 10)


def make_epsilon_greedy_policy(Q: defaultdict, epsilon: float, nA: int) -> callable:
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    I.e. create weight vector from which actions get sampled.
    :param Q: tabular state-action lookup function
    :param epsilon: exploration factor
    :param nA: size of action space to consider for this policy
    """

    def policy_fn(observation):
        policy = np.ones(nA) * epsilon / nA
        best_action = np.random.choice(np.flatnonzero(  # random choice for tie-breaking only
            Q[observation] == Q[observation].max()
        ))
        policy[best_action] += (1 - epsilon)
        return policy

    return policy_fn


def get_decay_schedule(start_val: float, decay_start: int, num_steps: int, type_: str):
    """
    Create epsilon decay schedule
    :param start_val: Start decay from this value (i.e. 1)
    :param decay_start: number of iterations to start epsilon decay after
    :param num_steps: Total number of steps to decay over
    :param type_: Which strategy to use. Implemented choices: 'const', 'log', 'linear'
    :return:
    """
    if np.isinf(num_steps):
        return np.array([start_val])
    if type_ == 'const':
        return np.array([start_val for _ in range(num_steps)])
    elif type_ == 'log':
        return np.hstack([[start_val for _ in range(decay_start)],
                          np.logspace(np.log10(start_val), np.log10(0.000001), (num_steps - decay_start))])
    elif type_ == 'linear':
        return np.hstack([[start_val for _ in range(decay_start)],
                          np.linspace(start_val, 0, (num_steps - decay_start), endpoint=True)])
    else:
        raise NotImplementedError


def td_update(q: defaultdict, state: int, action: int, reward: float, next_state: int, gamma: float, alpha: float,
              done: bool):
    """ Simple TD update rule """
    # TD update
    best_next_action = int(np.random.choice(np.flatnonzero(q[next_state] == q[next_state].max())))  # greedy best next
    if done:
        td_target = reward
    else:
        td_target = reward + gamma * q[next_state][best_next_action]
    td_delta = td_target - q[state][action]
    return q[state][action] + alpha * td_delta


def q_learning(
        environment: GridCore,
        num_episodes: int,
        discount_factor: float = 1.0,
        alpha: float = 0.5,
        epsilon: float = 0.1,
        epsilon_decay: str = 'const',
        decay_starts: int = 0,
        eval_every: int = 10):
    """
    Vanilla tabular Q-learning algorithm
    :param environment: which environment to use
    :param num_episodes: number of episodes to train
    :param discount_factor: discount factor used in TD updates
    :param alpha: learning rate used in TD updates
    :param epsilon: exploration fraction (either constant or starting value for schedule)
    :param epsilon_decay: determine type of exploration (constant, linear/exponential decay schedule)
    :param decay_starts: After how many episodes epsilon decay starts
    :param eval_every: Number of episodes between evaluations
    :param render_eval: Flag to activate/deactivate rendering of evaluation runs
    :return: training and evaluation statistics (i.e. rewards and episode lengths)
    """
    global ID
    assert 0 <= discount_factor <= 1, 'Lambda should be in [0, 1]'
    assert 0 <= epsilon <= 1, 'epsilon has to be in [0, 1]'
    assert alpha > 0, 'Learning rate has to be positive'
    # The action-value function.
    # Nested dict that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(environment.action_space.n))

    epsilon_schedule = get_decay_schedule(epsilon, decay_starts, num_episodes, epsilon_decay)
    ID = setTimeout(create_once_callable(rollout_one_episode), 1/1000,
                    create_proxy(epsilon_schedule), 0, num_episodes,
                    create_proxy(Q), create_proxy(environment), discount_factor, alpha, int(episode_slider.element.value))


def rollout_one_episode(epsilon_schedule, i_episode, num_episodes,
                        Q, environment, discount_factor, alpha, eval_every):
    global ID, reward_queue
    clearTimeout(ID)
    if eval_every > 0:
        out_element.element.innerText = f"{i_episode:>5d}"
        if i_episode > num_episodes:
            return
        # pyscript.write("envcanvas", '#' * 100)
        epsilon = epsilon_schedule[min(len(epsilon_schedule) - 1, min(i_episode, num_episodes - 1))]
        # The policy we're following
        policy = make_epsilon_greedy_policy(Q, epsilon, environment.action_space.n)
        policy_state = environment.reset()
        episode_length, cummulative_reward = 0, 0
        while True:  # roll out episode
            policy_action = int(np.random.choice(list(range(environment.action_space.n)), p=policy(policy_state)))
            s_, policy_reward, policy_done, _ = environment.step(policy_action)
            cummulative_reward += policy_reward
            episode_length += 1

            Q[policy_state][policy_action] = td_update(Q, policy_state, policy_action,
                                                       policy_reward, s_, discount_factor, alpha, policy_done)

            if policy_done:
                break
            policy_state = s_
        reward_queue.append(cummulative_reward)
        if len(reward_queue) >= 100:
            reward_queue.pop(0)
        reward_element.element.innerText = f"{np.mean(reward_queue):>+1.2f}"
    if (eval_every == 0) or (i_episode % eval_every == 0):
        ID = setTimeout(create_once_callable(do_step_and_plot), 1000/int(speed_slider.element.value),
                   create_proxy(Q), None, create_proxy(environment), create_proxy(epsilon_schedule),
                   i_episode, num_episodes, discount_factor, alpha, int(episode_slider.element.value), True)
    else:
        ID = setTimeout(create_once_callable(rollout_one_episode), 1/1000,
                   create_proxy(epsilon_schedule), i_episode+1, num_episodes,
                   create_proxy(Q), create_proxy(environment), discount_factor, alpha, int(episode_slider.element.value))


def do_step_and_plot(Q, policy_state, environment, epsilon_schedule,
                     i_episode, num_episodes, discount_factor, alpha, eval_every, start=False):
    global ID
    clearTimeout(ID)
    tempo_element.element.innerText = "-"
    if start:
        s_ = environment.reset()
    else:
        policy_action = int(np.random.choice(np.flatnonzero(Q[policy_state] == Q[policy_state].max())))
        environment.total_steps -= 1  # don't count evaluation steps
        s_, policy_reward, policy_done, _ = environment.step(policy_action)
        if policy_done:
            ID = setTimeout(create_once_callable(rollout_one_episode), 1/1000,
                            create_proxy(epsilon_schedule), i_episode+1, num_episodes,
                            create_proxy(Q), create_proxy(environment), discount_factor, alpha, int(episode_slider.element.value))
    policy_state = s_
    render_on_canvas(environment, ctx, is_decision=True, Q=Q)
    ID = setTimeout(create_once_callable(do_step_and_plot), 1000/int(speed_slider.element.value),
                    create_proxy(Q), policy_state, create_proxy(environment), create_proxy(epsilon_schedule),
                    i_episode, num_episodes, discount_factor, alpha, int(episode_slider.element.value))



class SkipTransition:
    """
    Simple helper class to keep track of all transitions observed when skipping through an MDP
    """

    def __init__(self, skips, df):
        self.state_mat = np.full((skips, skips), -1, dtype=int)  # might need to change type for other envs
        self.reward_mat = np.full((skips, skips), np.nan, dtype=float)
        self.idx = 0
        self.df = df

    def add(self, reward, next_state):
        """
        Add reward and next_state to triangular matrix
        :param reward: received reward
        :param next_state: state reached
        """
        self.idx += 1
        for i in range(self.idx):
            self.state_mat[self.idx - i - 1, i] = next_state
            # Automatically discount rewards when adding to corresponding skip
            self.reward_mat[self.idx - i - 1, i] = reward * self.df ** i + np.nansum(self.reward_mat[self.idx - i - 1])


def temporl_q_learning(
        environment: GridCore,
        num_episodes: int,
        discount_factor: float = 1.0,
        alpha: float = 0.5,
        epsilon: float = 0.1,
        epsilon_decay: str = 'const',
        decay_starts: int = 0,
        decay_stops: int = None,
        eval_every: int = 10,
        render_eval: bool = True,
        max_skip: int = 7):
    """
    Implementation of tabular TempoRL
    :param environment: which environment to use
    :param num_episodes: number of episodes to train
    :param discount_factor: discount factor used in TD updates
    :param alpha: learning rate used in TD updates
    :param epsilon: exploration fraction (either constant or starting value for schedule)
    :param epsilon_decay: determine type of exploration (constant, linear/exponential decay schedule)
    :param decay_starts: After how many episodes epsilon decay starts
    :param decay_stops: Episode after which to stop epsilon decay
    :param eval_every: Number of episodes between evaluations
    :param render_eval: Flag to activate/deactivate rendering of evaluation runs
    :param max_skip: Maximum skip size to use.
    :return: training and evaluation statistics (i.e. rewards and episode lengths)
    """
    global ID
    temporal_actions = max_skip
    action_Q = defaultdict(lambda: np.zeros(environment.action_space.n).astype(float))
    temporal_Q = defaultdict(lambda: np.zeros(temporal_actions).astype(float))
    if not decay_stops:
        decay_stops = num_episodes

    epsilon_schedule_action = get_decay_schedule(epsilon, decay_starts, decay_stops, epsilon_decay)
    epsilon_schedule_temporal = get_decay_schedule(epsilon, decay_starts, decay_stops, epsilon_decay)

    ID = setTimeout(create_once_callable(tempoRL_rollout_one_episode), 1/1000, 0, num_episodes,
                    create_proxy(epsilon_schedule_action), create_proxy(epsilon_schedule_temporal),
                    create_proxy(action_Q), create_proxy(temporal_Q), create_proxy(environment), temporal_actions,
                    discount_factor, alpha, int(episode_slider.element.value))



def tempoRL_do_step_and_plot(action_Q, temporal_Q, policy_state, environment,
                             epsilon_schedule_action, epsilon_schedule_temporal, temporal_actions,
                             i_episode, num_episodes, discount_factor, alpha, eval_every, start=False,
                             temporal_action=0, policy_action=None):
    global ID
    clearTimeout(ID)
    decision = False
    tempo_element.element.innerText = f"{max(0,temporal_action):>2d}"
    if start:
        s_ = environment.reset()
        decision = True
    else:
        if temporal_action <= 0:
            policy_action = int(np.random.choice(np.flatnonzero(action_Q[policy_state] == action_Q[policy_state].max())))
            temporal_state = (policy_state, policy_action)
            temporal_action = int(np.max(  # if there are ties use the larger action
                    np.flatnonzero(temporal_Q[temporal_state] == temporal_Q[temporal_state].max())))
            decision = True
        environment.total_steps -= 1  # don't count evaluation steps
        s_, policy_reward, policy_done, _ = environment.step(policy_action)
        if policy_done:
            tempo_element.element.innerText = '-'
            ID = setTimeout(create_once_callable(tempoRL_rollout_one_episode), 1/1000, i_episode + 1, num_episodes,
                            create_proxy(epsilon_schedule_action), create_proxy(epsilon_schedule_temporal),
                            create_proxy(action_Q), create_proxy(temporal_Q), create_proxy(environment),
                            temporal_actions, discount_factor, alpha, int(episode_slider.element.value))
    policy_state = s_
    render_on_canvas(environment, ctx, is_decision=decision, Q=action_Q)
    ID = setTimeout(create_once_callable(tempoRL_do_step_and_plot), 1000/int(speed_slider.element.value),
                    create_proxy(action_Q), create_proxy(temporal_Q),
                    create_proxy(policy_state), create_proxy(environment),
                    create_proxy(epsilon_schedule_action), create_proxy(epsilon_schedule_temporal), temporal_actions,
                    i_episode, num_episodes, discount_factor, alpha, int(episode_slider.element.value), False,
                    temporal_action - 1, policy_action)


def tempoRL_rollout_one_episode(i_episode, num_episodes, epsilon_schedule_action, epsilon_schedule_temporal,
                                action_Q, temporal_Q, environment, temporal_actions,
                                discount_factor, alpha, eval_every):
    global ID, reward_queue
    clearTimeout(ID)

    if eval_every > 0:
        out_element.element.innerText = f"{i_episode:>5d}"
        epsilon_action = epsilon_schedule_action[min(len(epsilon_schedule_action) - 1, min(i_episode, num_episodes - 1))]
        epsilon_temporal = epsilon_schedule_temporal[min(len(epsilon_schedule_temporal) - 1,
                                                         min(i_episode, num_episodes - 1))]
        action_policy = make_epsilon_greedy_policy(action_Q, epsilon_action, environment.action_space.n)
        temporal_policy = make_epsilon_greedy_policy(temporal_Q, epsilon_temporal, temporal_actions)

        episode_r = 0
        state = environment.reset()  # type: list
        action_pol_len = 0
        while True:  # roll out episode
            action = int(np.random.choice(list(range(environment.action_space.n)), p=action_policy(state)))
            temporal_state = (state, action)
            action_pol_len += 1
            temporal_action = int(np.random.choice(list(range(temporal_actions)), p=temporal_policy(temporal_state)))

            s_ = None
            done = False
            tmp_state = state
            skip_transition = SkipTransition(temporal_action + 1, discount_factor)
            reward = 0
            for tmp_temporal_action in range(temporal_action + 1):
                if not done:
                    # only perform action if we are not done. If we are not done "skipping" though we have to
                    # still add reward and same state to the skip_transition.
                    s_, reward, done, _ = environment.step(action)
                    episode_r += reward
                    # 1-step update of action Q (like in vanilla Q)
                    action_Q[tmp_state][action] = td_update(action_Q, tmp_state, action,
                                                            reward, s_, discount_factor, alpha, done)
                skip_transition.add(reward, tmp_state)

                count = 0
                # For all sofar observed transitions compute all forward skip updates
                for skip_num in range(skip_transition.idx):
                    skip = skip_transition.state_mat[skip_num]
                    rew = skip_transition.reward_mat[skip_num]
                    skip_start_state = (skip[0], action)

                    # Temporal TD update
                    best_next_action = int(np.random.choice(
                        np.flatnonzero(action_Q[s_] == action_Q[s_].max())))  # greedy best next
                    td_target = rew[skip_transition.idx - 1 - count] + (
                            discount_factor ** (skip_transition.idx - 1)) * action_Q[s_][best_next_action]
                    td_delta = td_target - temporal_Q[skip_start_state][skip_transition.idx - count - 1]
                    temporal_Q[skip_start_state][skip_transition.idx - count - 1] += alpha * td_delta
                    count += 1

                tmp_state = s_
            state = s_
            if done:
                break
        reward_queue.append(episode_r)
        if len(reward_queue) >= 100:
            reward_queue.pop(0)
        reward_element.element.innerText = f"{np.mean(reward_queue):>+1.2f}"
    if (eval_every == 0) or (i_episode % eval_every == 0):
        ID = setTimeout(create_once_callable(tempoRL_do_step_and_plot), 1000/int(speed_slider.element.value),
                        create_proxy(action_Q), create_proxy(temporal_Q),
                        None, create_proxy(environment),
                        create_proxy(epsilon_schedule_action), create_proxy(epsilon_schedule_temporal),
                        temporal_actions, i_episode, num_episodes, discount_factor,
                        alpha, int(episode_slider.element.value), True, -1, -1)
    else:
        ID = setTimeout(create_once_callable(tempoRL_rollout_one_episode), 1/1000, i_episode + 1, num_episodes,
                        create_proxy(epsilon_schedule_action), create_proxy(epsilon_schedule_temporal),
                        create_proxy(action_Q), create_proxy(temporal_Q), create_proxy(environment), temporal_actions,
                        discount_factor, alpha, int(episode_slider.element.value))



def main(*args):
    global ID, reward_queue
    reward_queue = []
    try:
        del d
    except:
        pass
    if ID is not None:
        clearTimeout(ID)
    if env_selector.element.value == 'pit':
        d = Pit6x10Env(max_steps=100, percentage_reward=False, no_goal_rew=False,
                       act_fail_prob=0)
    elif env_selector.element.value == 'bridge':
        d = Bridge6x10Env(max_steps=100, percentage_reward=False, no_goal_rew=False,
                          act_fail_prob=0)
    elif env_selector.element.value == 'zigzag':
        d = ZigZag6x10(max_steps=100, percentage_reward=False, no_goal_rew=False,
                       act_fail_prob=0)
    elif env_selector.element.value == 'smallfield':
        d = SmallField(max_steps=100, percentage_reward=False, no_goal_rew=False,
                       act_fail_prob=0, goal=(4,15))
    elif env_selector.element.value == 'largefield':
        d = LargeField(max_steps=100, percentage_reward=False, no_goal_rew=False,
                       act_fail_prob=0, goal=(9,25))
    agent = 'sq' if not tempo_radio_element.element.checked else 'q'
    max_skip = int(skip_element.element.value)
    episodes = np.inf
    agent_eps_d = 'const'
    agent_eps = 0.1
    eval_eps = 2000
    out_element.element.innerText = "{:>5d}".format(0)
    reward_element.element.innerText = "0"
    tempo_element.element.innerText = "-"

    # setup agent
    if agent == 'sq':
        temporl_q_learning(d, episodes,
                           epsilon_decay=agent_eps_d, epsilon=agent_eps,
                           discount_factor=.99, alpha=.5, eval_every=eval_eps,
                           max_skip=max_skip)
    elif agent == 'q':
        q_learning(d, episodes,
                   epsilon_decay=agent_eps_d,
                   epsilon=agent_eps,
                   discount_factor=.99,
                   alpha=.5, eval_every=eval_eps)


def render_on_canvas(env, canvas, is_decision=False, Q=None):
    # canvas.clearRect(0, 0, 600, 300) # clear canvas
    border_width = 2
    border_height = 2
    width = env_element.element.width
    height = env_element.element.height
    environment_shape = env.shape
    cell_width = (width/environment_shape[1]) - border_width
    cell_height = (height/environment_shape[0]) - border_height
    min_cell_width = cell_width/3 - 1/3
    min_cell_height = cell_height/3 - 1/3
    mini_coords = np.array([[1,0], [0, 1], [1, 2], [2, 1]])
    font_size_in_px = int(20 * cell_height/23) # 13 is min cell width for smallest env
    base_vals = [50, 50, 50]

    if env.prev_s is None:  # only draw full grid once
        canvas.fillstyle = "black"
        canvas.clearRect(0, 0, width, height)  # clear canvas
        for s in range(env.nS):
            base_vals = [50, 50, 50]
            position = np.unravel_index(s, env.shape)
            output = env.map_output(s, position)
            if s == 0:
                canvas.fillStyle = 'blue'
            else:
                if output == ' . ' or s in env._pits:
                    canvas.fillStyle = 'black'
                elif output == ' X ':
                    canvas.fillStyle = 'orange'
                else:
                    if Q is not None:
                        v = Q[s].max()
                        if v >= 0: base_vals[1] += (256-base_vals[1]) **(v)**3
                        if v < 0: base_vals[0] += (256-base_vals[0]) **(v * -1)**3
                    cstr = f'rgb({base_vals[0]}, {base_vals[1]}, {base_vals[2]})'
                    canvas.fillStyle = cstr
            canvas.fillRect((border_width*.5)+position[1]*(cell_width + border_width),
                            (border_height*.5)+position[0]*(cell_height + border_height),
                            cell_width, cell_height)
            # if output not in [' X ', ' . '] and s!=0:
            #     if Q is not None:
            #         pos_x = (border_width*.5)+position[1]*(cell_width + border_width)
            #         pos_y = (border_height*.5)+position[0]*(cell_height + border_height)
            #         v = Q[s]  # left up right down
            #         # m_idxs = np.flatnonzero(v == v.max())
            #         for tmp_v, coord in zip(v, mini_coords):
            #             base_vals = [50, 50, 50]
            #             if tmp_v >= 0: base_vals[1] += (256-base_vals[1]) **(tmp_v)**2
            #             if tmp_v < 0: base_vals[0] += (256-base_vals[0]) **(tmp_v * -1)**2
            #             cstr = f'rgb({base_vals[0]}, {base_vals[1]}, {base_vals[2]})'
            #             canvas.fillStyle = cstr
            #             canvas.fillRect(pos_x + (1/3)*.5 + coord[1] * (min_cell_width + 1/3),
            #                             pos_y + (1/3)*.5 + coord[0] * (min_cell_height + 1/3),
            #                             min_cell_width, min_cell_height)
            canvas.fillStyle = 'black'
            if s == 0:
                canvas.font = f'{font_size_in_px}px bold serif'
                canvas.fillText('S',
                                (border_width*.5)+position[1] * (cell_width + border_width) + cell_width / 2 - font_size_in_px*.33,
                                (border_height*.5)+position[0] * (cell_height + border_height) + cell_height / 2 + font_size_in_px*.33)
            if output == ' X ':
                canvas.font = f'{font_size_in_px}px bold serif'
                canvas.fillText('G',
                                (border_width*.5)+position[1] * (cell_width + border_width) + cell_width / 2 - font_size_in_px*.33,
                                (border_height*.5)+position[0] * (cell_height + border_height) + cell_height / 2 + font_size_in_px*.33)
    else:
        if env.prev_s == env.s:
            return
        position = np.unravel_index(env.prev_s, env.shape)
        output = env.map_output(env.prev_s, position)
        if position[0] == 0 and position[1] == 0:
            canvas.fillStyle = 'blue'
        elif output == ' . ' or env.prev_s in env._pits:
            canvas.fillStyle = 'black'
        elif output == ' X ':
            canvas.fillStyle = 'orange'
        else:
            if Q is not None:
                v = Q[env.prev_s].max()
                if v >= 0: base_vals[1] += (256-base_vals[1]) **(v)**3
                if v < 0: base_vals[0] += (256-base_vals[0]) **(v * -1)**3
            cstr = f'rgb({base_vals[0]}, {base_vals[1]}, {base_vals[2]})'
            canvas.fillStyle = cstr
        canvas.fillRect((border_width*.5)+position[1]*(cell_width + border_width),
                        (border_height*.5)+position[0]*(cell_height + border_height),
                        cell_width, cell_height)
        # if output not in [' . ', ' X '] and env.prev_s != 0:
        #     if Q is not None:
        #         pos_x = (border_width*.5)+position[1]*(cell_width + border_width)
        #         pos_y = (border_height*.5)+position[0]*(cell_height + border_height)
        #         v = Q[env.prev_s]  # left up right down
        #         # m_idxs = np.flatnonzero(v == v.max())
        #         for tmp_v, coord in zip(v, mini_coords):
        #             base_vals = [50, 50, 50]
        #             if tmp_v >= 0: base_vals[1] += (256-base_vals[1]) **(tmp_v)**2
        #             if tmp_v < 0: base_vals[0] += (256-base_vals[0]) **(tmp_v * -1)**2
        #             cstr = f'rgb({base_vals[0]}, {base_vals[1]}, {base_vals[2]})'
        #             canvas.fillStyle = cstr
        #             canvas.fillRect(pos_x + (1/3)*.5 + coord[1] * (min_cell_width + 1/3),
        #                             pos_y + (1/3)*.5 + coord[0] * (min_cell_height + 1/3),
        #                             min_cell_width, min_cell_height)
        if position[0] == 0 and position[1] == 0 or output == ' X ':
            canvas.font = f'{font_size_in_px}px bold serif'
            canvas.fillStyle = 'rgb(0, 0, 0)'
            text = 'S'
            if output == ' X ':
                text = 'G'
            canvas.fillText(text,
                            (border_width*.5)+position[1] * (cell_width + border_width) + cell_width / 2 - font_size_in_px*.33,
                            (border_height*.5)+position[0] * (cell_height + border_height) + cell_height / 2 + font_size_in_px*.33)
    position = np.unravel_index(env.s, env.shape)
    if is_decision:
        canvas.fillStyle = 'rgb(150, 0, 150)'
    else:
        canvas.fillStyle = 'rgb(100, 200, 50)'
    canvas.beginPath()
    canvas.arc((border_width*.5)+position[1] * (cell_width + border_width) + cell_width / 2,
               (border_height*.5)+position[0] * (cell_height + border_height) + cell_height / 2,
               cell_width*.25, 0, 2 * np.pi)
    canvas.stroke()
    canvas.fill()
    canvas.fillStyle = 'black'


def clear_interval(*args):
    """
    Stop animation
    """
    global ID
    clearTimeout(ID)

start_button = Element('start-btn')
stop_button = Element('stop-btn')
env_element = Element("envcanvas")
env_selector = Element("envs")

out_element = Element("out-episodes")
reward_element = Element("out-t-rew")
tempo_element = Element("out-tempo")

skip_element = Element("quantity")
tempo_radio_element = Element("tempo-radio")
vanilla_radio_element = Element("vanilla-radio")

speed_slider = Element("SpeedSlider")
episode_slider = Element("LambdaSlider")
# env_chooser = Element("ddEnvChooser")
start_button.element.onclick = main
stop_button.element.onclick = clear_interval

ctx = env_element.element.getContext("2d")
ctx.fillStyle = 'rgb(200, 0, 0)'
ctx.fillRect(10, 10, 50, 50)

ctx.fillStyle = 'rgba(0, 0, 200, 0.5)'
ctx.fillRect(30, 30, 50, 50)

d = Pit6x10Env(max_steps=100, percentage_reward=False, no_goal_rew=False,
               act_fail_prob=0)
d.reset()
render_on_canvas(d, ctx, is_decision=True)
