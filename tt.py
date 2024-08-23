import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import numpy as np
import gym

# 相当于rllib中的vf_share_layers = True
class ActorCritic(nn.Module):
    def __init__(self, obs_dim,hidden_dim, act_dim):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.actor = nn.Linear(hidden_dim, act_dim)
        self.critic = nn.Linear(hidden_dim, 1)    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.actor(x), self.critic(x)
    
    def act(self, obs):
        logits, _ = self.forward(obs)
        return torch.relu(logits)
    
    def evaluate(self, obs):
        logits, value = self.forward(obs)
        return torch.relu(logits), value

class SharedBuffer:
    def __init__(self, buffer_size, obs_dim, act_dim):
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        
        self.obs = mp.Array('d', np.zeros((buffer_size, obs_dim)).flatten(), lock=False)
        self.actions = mp.Array('d', np.zeros((buffer_size, act_dim)).flatten(), lock=False)
        self.rewards = mp.Array('d', np.zeros(buffer_size), lock=False)
        self.dones = mp.Array('d', np.zeros(buffer_size), lock=False)
        self.values = mp.Array('d', np.zeros(buffer_size), lock=False)
        self.log_probs = mp.Array('d', np.zeros(buffer_size), lock=False)
        self.lock = mp.Lock()
        self.pointer = mp.Value('i', 0)

    def add(self, obs, action, reward, done, value, log_prob):
        with self.lock:
            idx = self.pointer.value
            self.obs[idx * self.obs_dim:(idx + 1) * self.obs_dim] = obs
            self.actions[idx * self.act_dim:(idx + 1) * self.act_dim] = action
            self.rewards[idx] = reward
            self.dones[idx] = done
            self.values[idx] = value
            self.log_probs[idx] = log_prob
            self.pointer.value = (self.pointer.value + 1) % self.buffer_size

    def get(self):
        with self.lock:
            return (np.array(self.obs).reshape(self.buffer_size, self.obs_dim),
                    np.array(self.actions).reshape(self.buffer_size, self.act_dim),
                    np.array(self.rewards),
                    np.array(self.dones),
                    np.array(self.values),
                    np.array(self.log_probs))

    def reset(self):
        with self.lock:
            self.pointer.value = 0

def worker(shared_buffer, env_name, policy, seed):
    env = gym.make(env_name)


    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    obs = env.reset(seed=seed)
    done = False

    while True:
        obs = env.reset(seed=seed)

        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            action = policy.act(obs_tensor)
            value = policy.evaluate(obs_tensor)[1].item()
            log_prob = torch.log(action).item()
        
        next_obs, reward, done, _ = env.step(action.numpy())
        shared_buffer.add(obs, action.numpy(), reward, done, value, log_prob)
        obs = next_obs
        
        if done:
            obs = env.reset(seed=seed)
            done = False


def ppo_update(policy, optimizer, buffer, clip_param=0.2, c1=0.5, c2=0.01, epochs=10, batch_size=64):
    obs, actions, rewards, dones, values, log_probs = buffer.get()

    obs = torch.tensor(obs, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.float32)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)
    values = torch.tensor(values, dtype=torch.float32)
    log_probs = torch.tensor(log_probs, dtype=torch.float32)

    returns = []
    discounted_return = 0
    for reward, done in zip(reversed(rewards), reversed(dones)):
        if done:
            discounted_return = 0
        discounted_return = reward + 0.99 * discounted_return
        returns.insert(0, discounted_return)
    returns = torch.tensor(returns, dtype=torch.float32)
    
    advantages = returns - values

    for _ in range(epochs):
        for idx in range(0, len(obs), batch_size):
            sampled_indices = np.random.choice(len(obs), batch_size)
            sampled_obs = obs[sampled_indices]
            sampled_actions = actions[sampled_indices].long()  # Assuming discrete actions
            sampled_log_probs = log_probs[sampled_indices]
            sampled_advantages = advantages[sampled_indices]
            sampled_returns = returns[sampled_indices]

            logits, values = policy.evaluate(sampled_obs)
            dist = torch.distributions.Categorical(logits=logits)
            log_probs_new = dist.log_prob(sampled_actions)
            
            ratio = torch.exp(log_probs_new - sampled_log_probs)
            surr1 = ratio * sampled_advantages
            surr2 = torch.clamp(ratio, 1 - clip_param, 1 + clip_param) * sampled_advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = c1 * (sampled_returns - values).pow(2).mean()
            entropy_loss = -c2 * dist.entropy().mean()

            loss = actor_loss + critic_loss + entropy_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    buffer.reset()

if __name__ == '__main__':
    env_name = "CartPole-v1"
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    hidden_dim = 512
    act_dim = 1
    buffer_size = 2048

    shared_buffer = SharedBuffer(buffer_size, obs_dim, act_dim)
    policy = ActorCritic(obs_dim,hidden_dim, act_dim)
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)

    processes = []
    for rank in range(mp.cpu_count()):
        p = mp.Process(target=worker, args=(shared_buffer, env_name, policy, rank))
        p.start()
        processes.append(p)

    try:
        for update in range(1000):
            ppo_update(policy, optimizer, shared_buffer)
            print(f"Update {update} completed.")
    finally:
        for p in processes:
            p.terminate()
            p.join()
