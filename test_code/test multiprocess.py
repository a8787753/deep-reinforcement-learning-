import numpy as np
import gym
import multiprocessing as mp
from multiprocessing import Process, Queue, Pipe
import collections
import threading as td
import time
import copy
import torch
import torch.nn as nn

from test_MAPPO.util import get_shape_from_obs_space, get_shape_from_act_space
from test_MAPPO.Layers import ACTLayer, MLPLayer


n_process = mp.cpu_count()

env = gym.make('CartPole-v0')

obs_space = env.observation_space
act_space = env.action_space
hidden_size = 256
layer_N = 1


class A2C(nn.Module):
    def __init__(self, obs_space, act_space, hidden_size, layer_N):
        super(A2C, self).__init__()

        self.obs_shape = get_shape_from_obs_space(obs_space)
        self.act_shape = get_shape_from_act_space(act_space)
        # print(self.obs_shape, self.act_shape)
        self.hidden_size = hidden_size
        self.layer_N = layer_N

        self.mlp = MLPLayer(self.obs_shape, self.hidden_size, self.layer_N)

        self.act = ACTLayer(act_space, self.hidden_size)

    def action(self, obs):
        if obs.__class__.__name__ == 'tuple':
            obs = obs[0]
        obs = torch.Tensor(obs)
        return self.act(self.mlp(obs))

    def train(self, data):
        pass


a2c = A2C(obs_space, act_space, hidden_size, layer_N)

overall_data = n_process * 5 // n_process


# print(a2c)


class AgentProcess(Process):
    def __init__(self, conn, _id):
        super(AgentProcess, self).__init__()
        self.conn = conn
        self.id = _id
        self.a2c = a2c

        # self.a2c_params = params[:6]
        #
        # self.agent_params = params[6:]
        # self.n_games = self.agent_params[0]
        # self.overall_data = self.agent_params[1]

        self.msg_queue = []
        np.random.seed(self.id * 100)

    def run(self):
        self.agent = copy.deepcopy(self.a2c)

        # def treatQueue():
        #     msg = self.conn.recv()
        #     if msg == 'load':
        #         self.agent.load_model()
        #         print('Process {} loaded the master model'.format(self.id))
        #
        #     if msg[0] == 'train_with_batchs':
        #         print('Master process is training ...')
        #         t0 = time.time()
        #         self.agent.train_with_batchs(msg[1])
        #         self.agent.save_model()
        #         print('Master process finished training. Time cost: {}\n'.format(time.time()-t0))
        #         self.conn.send('saved')
            # if self.id != 0:

        batch_s = []
        batch_a = []
        batch_r = []

        # env = gym.make('CartPole-v1')

        ep_reward = 0

        n_games = 0

        t = 0

        print('Process {} starts playing the {}th game'.format(self.id, n_games))
        state = env.reset()
        while t < overall_data:
            action = self.agent.action(state)
            action = int(action[0])
            newState, reward, done, truncated, info = env.step(action)

            ep_reward += reward

            batch_s.append(state)
            batch_a.append(action)
            batch_r.append(reward)

            if done:
                n_games += 1
                print('Process {} starts playing the {}th game'.format(self.id, n_games))
                state = env.reset()
            else:
                state = newState
            # elif t >= overall_data:
            #     info = newState

            t += 1

        print("Process "+str(self.id)+" finished playing.")
        batch = (batch_s, batch_a, batch_r, info)
        self.conn.send((ep_reward, batch))
        # treatQueue()


class MasterProcess:
    def __init__(self):
        self.processes = {}

    def train_agents(self):
        pipes = {}
        for i in range(n_process):
            parent_conn, child_conn = Pipe()
            pipes[i] = parent_conn
            p = AgentProcess(conn=child_conn, _id=i)
            p.start()
            self.processes[i] = p

            ep_rewards = {}
            batchs = {}
            t0 = time.time()

            def listenToAgent(_id, ep_rewards):
                while True:
                    msg = pipes[_id].recv()
                    # if msg == 'saved':
                    #     print('Master process saved the weights')
                    #     for j in pipes:
                    #         if j != 0:
                    #             pipes[j].send('load')
                    # else:
                    ep_reward = msg[0]
                    ep_rewards[_id] = ep_reward
                    batchs[_id] = msg[1]
                    print("Process "+str(id)+" returns ep_reward "+str(ep_reward))

            threads_listen = []
            print('Threads to start')
            for _id in pipes:
                t = td.Thread(target=listenToAgent, args=(_id, ep_rewards))
                t.start()
                threads_listen.append(t)
            print('Threads started')

            _iter = 1
            mean_rewards = []
            # file = open('log_rewards', 'w')
            while True:
                if len(ep_rewards) == n_process:
                    # id_best = max(ep_rewards, key=ep_rewards.get)
                    # mean_rewards.append(np.mean(list(ep_rewards.values())))
                    print("End of iteration "+str(iter)+". Mean reward sor far : "+str(np.mean(mean_rewards[-10:])))
                    print("End of iteration " + str(iter))
                    _iter += 1
                    # file.write(str(np.mean(mean_rewards[-10:]))+'\n')
                    # file.flush
                    print('Time: '+str(time.time()-t0))
                    print('\n')
                    # pipes[0].send(('train_with_batchs', list(batchs.values())))
                    t0 = time.time()
                    # ep_rewards.clear()
                    # batchs.clear()
                    



if __name__ == '__main__':
    master = MasterProcess()
    master.train_agents()

