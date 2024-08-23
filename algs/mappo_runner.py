import torch
import numpy as np
from algs.ppo_discrite import PPO

class MAPPORunner:
    def __init__(self, env, config):
        self.env = env
        self.num_episodes = config.num_episodes

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
        for i in range(self.num_episodes):
            # 初始化每个回合的过渡数据集
            transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}

            s, _ = self.env.reset()
            episode_reward = 0
            
            while True:
                # 选择动作
                a = self.agent.take_action(np.concatenate(s))

                # 更新环境
                next_s, r, done, _ = self.env.step(a)
                episode_reward += r

                # 记录过渡数据
                transition_dict['states'].append(s)
                transition_dict['actions'].append(a)
                transition_dict['next_states'].append(next_s)
                transition_dict['rewards'].append(r)
                transition_dict['dones'].append(done)

                s = next_s