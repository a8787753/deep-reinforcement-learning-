from collections import defaultdict, deque
from itertools import chain
import os
import time

import imageio
import numpy as np
import torch
import wandb

# from onpolicy.utils.util import update_linear_schedule
# from onpolicy.runner.shared.base_runner import Runner
from util import update_linear_schedule
from base_runner import Runner

def _t2n(x):
    return x.detach().cpu().numpy()


class FootballRunner(Runner):
    def __init__(self, config):
        super(FootballRunner, self).__init__()
        self.env_infos = defaultdict(list)

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                #Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)

                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)

                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            # save model
            if (total_num_steps % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if total_num_steps % self.log_interval == 0:
                end = time.time()
                print('\n Env {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n'
                      .format(self.env_name,
                              self.algorithm_name,
                              self.experiment_name,
                              episode,
                              episodes,
                              total_num_steps,
                              self.num_env_steps,
                              int(total_num_steps / (end - start))))

                train_infos['average_episode_rewards'] = np.mean(self.buffer.rewards) * self.episode_length
                print('average episode rewards is {}'.format(train_infos['average_episode_rewards']))
                self.log_train(train_infos, total_num_steps)
                self.log_env(self.env_infos, total_num_steps)
                self.env_infos = defaultdict(list)

            # eval
            if total_num_steps % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs = self.envs.reset()

        # insert obs to buffer
        self.buffer.share_obs[0] = obs.copy()
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()

        # [n_envs, n_agents, ...] -> [n_envs*n_agents, ...]
        values, actions, action_log_probs, rnn_states, rnn_states_critic = self.trainer.policy.get_actions(
            np.concatenate(self.buffer.share_obs[step]),
            np.concatenate(self.buffer.obs[step]),
            np.concatenate(self.buffer.rnn_states[step]),
            np.concatenate(self.buffer.rnn_states_critic[step]),
            np.concatenate(self.buffer.masks[step])
        )

        # [n_envs*n_agents, ...] -> [n_envs, n_agents, ...]
        values = np.array(np.split(_t2n(values), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(actions), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_probs), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))

        actions_env = [actions[idx, :, 0] for idx in range(self.n_rollout_threads)]

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        #update env_infos if done
        dones_env = np.all(dones, axis=-1)
        if np.any(dones_env):
            for done, info in zip(dones_env, infos):
                if done:
                    self.env_infos['goal'].append(info['score_reward'])
                    if info['score_reward'] > 0:
                        self.env_infos['win_rate'].append(1)
                    else:
                        self.env_infos['win_rate'].append(0)
                    self.env_infos['steps'].append(info['max_steps'] - info['steps_left'])

        # reset rnn and mask args for done envs
        rnn_states[dones_env==True]
