import torch
import numpy as np
from algs.ppo_discrite import PPO
from multiprocessing import Process, Pipe

class CloudpickleWrapper(object):
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if 'bool' in done.__class__.__name__:
                if done:
                    ob = env.reset()
            else:
                if np.all(done):
                    ob = env.reset()

            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send((ob))
        elif cmd == 'render':
            if data == "rgb_array":
                fr = env.render(mode=data)
                remote.send(fr)
            elif data == "human":
                env.render(mode=data)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            env.close()
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.share_observation_space, env.action_space))
        else:
            raise NotImplementedError
       
class SubproVecEnv():
    def __init__(self, env_fns):
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)

        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])


        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                     for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True
            p.start()
        for remote in self.work_remotes:
            remote.close()
        self.remotes[0].send(('get_spaces', None))
        observation_space,share_observation_space, action_space = self.remotes[0].recv()
        self.observation_space = observation_space
        self.share_observation_space = share_observation_space
        self.action_space = action_space

    def step(self, actions):
        """
        Step the environments synchronously.
        This is available for backwards compatibility.
        """
        self.step_async(actions)
        return self.step_wait()

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        obs = [remote.recv() for remote in self.remotes]
        return np.stack(obs)


    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    def render(self, mode="rgb_array"):
        for remote in self.remotes:
            remote.send(('render', mode))
        if mode == "rgb_array":   
            frame = [remote.recv() for remote in self.remotes]
            return np.stack(frame) 

class IPPORunner:
    def __init__(self, env, config):
        self.env = env
        self.num_episodes = config.num_episodes
        self.config = config

        # 初始化PPO智能体
        self.agent = PPO(
            n_states=env.observation_space[0].shape[0],
            n_hiddens=config.n_hiddens,
            n_actions=env.action_space[0].n,
            actor_lr=config.actor_lr,
            critic_lr=config.critic_lr,
            lmbda=config.lmbda,
            eps=config.eps,
            gamma=config.gamma,
            device=torch.device(config.device),
        )
    def run(self):
        step = 0
        for i in range(self.num_episodes):
            # 初始化每个回合的过渡数据集
            transition_dicts = [ {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []} for _ in range(self.config.n_spacecraft) ]

            s, _ = self.env.reset()  # 初始化环境状态
            episode_reward = 0  # 记录回合总奖励

            while True:
                # 为每颗卫星选择动作
                actions = [self.agent.take_action(sat_state) for sat_state in s]

                # 更新环境
                next_s, r, terminated, truncated, info = self.env.step(actions)
                done = terminated or truncated
                episode_reward += r

                # 记录过渡数据
                for idx in range(self.config.n_spacecraft):
                    # print(f'Step: {step}')

                    step += 1
                    transition_dicts[idx]['states'].append(s[idx])
                    transition_dicts[idx]['actions'].append(actions[idx])
                    transition_dicts[idx]['next_states'].append(next_s[idx])
                    transition_dicts[idx]['rewards'].append(r)
                    transition_dicts[idx]['dones'].append(done)

                s = next_s  # 更新状态

                # 若回合结束，打印总奖励并跳出循环
                if done:
                    
                    break

                # 更新智能体

            if step >= self.config.batch_size:
                print(f'Updating agent...')
                print(f'Episode: {i}, Reward: {episode_reward}')
                for transition_dict in transition_dicts:
                    self.agent.update(transition_dict)
                transition_dicts = [ {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []} for _ in range(self.config.n_spacecraft) ]
                step = 0


        # if i % 10 == 0 and i > 500:
        #     save_data = {'net': self.agent.actor.state_dict(), 'opt': self.agent.actor.optimizer.state_dict(), 'i': i}
        #     torch.save(save_data, "./checkpoints/model.pth")

        # c=PPO()
        # checkpoint = torch.load("./checkpoints/model_PG.pth")
        # c.agent.load_state_dict(checkpoint['net'])



# TEST
# print("测试PG中...")
# c=PG()
# checkpoint = torch.load("D:\PyCharm 2019.3\mytorch_spacework\demo\model_PG.pth")
# c.policy.load_state_dict(checkpoint['net'])
# for j in range(10):
#     state = env.reset()
#     total_rewards = 0
#     while True:
#         env.render()
#         state = torch.FloatTensor(state)
#         action=c.choose(state)
#         new_state, reward, done, info = env.step(action)  # 执行动作
#         total_rewards += reward
#         if done:
#             print("Score", total_rewards)
#             break
#         state = new_state
# env.close()
