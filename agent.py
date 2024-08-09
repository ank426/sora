import numpy as np
import torch
from tensordict import TensorDict
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer

from net import Net


class Agent:
    def __init__(self, state_dim, action_dim, save_dir):
        # assert state_dim == (4, 72, 128)
        # assert action_dim == 16

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

        self.device = "cuda"
        self.memory = TensorDictReplayBuffer(
            storage=LazyMemmapStorage(20000, device=torch.device("cpu"))
        )

        self.net = Net(self.state_dim, self.action_dim).float()
        self.net = self.net.to(device=self.device)

        self.burnin = 1e4  # min exps before training
        self.learn_every = 3  # no of exps between updates to Q_online
        self.sync_every = 1e4  # no of exps between Q_target & Q_online sync

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

        self.batch_size = 32
        self.gamma = 0.9

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0

        # self.save_every = 5e5
        self.save_every = 1e4

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)
        else:
            state = (state[0] if isinstance(state, tuple) else state).__array__()
            # assert state.shape == (4, 72, 128)
            state = torch.tensor(state, device=self.device).unsqueeze(0)
            # assert state.shape == (1, 4, 72, 128)
            action_values = self.net(state, model="online")
            # assert action_values.shape == (1, 16), print(
            #     action_values, action_values.shape
            # )
            action_idx = torch.argmax(action_values, axis=1).item()

        # assert 0 <= action_idx < 16

        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        def first_if_tuple(x):
            return x[0] if isinstance(x, tuple) else x

        state = first_if_tuple(state).__array__()
        next_state = first_if_tuple(next_state).__array__()

        state = torch.tensor(state)
        next_state = torch.tensor(next_state)
        # assert state.shape == next_state.shape == (4, 72, 128)
        # action = torch.tensor([action])
        # reward = torch.tensor([reward])
        # done = torch.tensor([done])
        action = torch.tensor(action).unsqueeze(0)
        reward = torch.tensor(reward).unsqueeze(0)
        done = torch.tensor(done).unsqueeze(0)
        # assert action.shape == reward.shape == done.shape == (1,)

        # start = process.memory_info().rss
        self.memory.add(
            TensorDict(
                {
                    "state": state,
                    "next_state": next_state,
                    "action": action,
                    "reward": reward,
                    "done": done,
                },
                batch_size=[],
            )
        )
        # end = process.memory_info().rss
        # print(start, end, end - start)

    def recall(self):
        batch = self.memory.sample(self.batch_size).to(self.device)
        state, next_state, action, reward, done = (
            batch.get(key)
            for key in ("state", "next_state", "action", "reward", "done")
        )
        # assert state.shape == next_state.shape == (self.batch_size, 4, 72, 128)
        # assert action.shape == reward.shape == done.shape == (self.batch_size, 1)

        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def td_estimate(self, state, action):
        # assert state.shape == (self.batch_size, 4, 72, 128)
        # assert action.shape == (self.batch_size, 4)

        current_Q = self.net(state, model="online")[
            np.arange(0, self.batch_size), action
        ]

        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        # assert next_state.shape == (self.batch_size, 4, 72, 128)
        # assert reward.shape == done.shape == (self.batch_size,)

        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def save(self):
        save_path = (
            self.save_dir / f"sora_net_{int(self.curr_step // self.save_every)}.chkpt"
        )
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )
        print(f"SoraNet saved to {save_path} at step {self.curr_step}")

    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        state, next_state, action, reward, done = self.recall()

        td_est = self.td_estimate(state, action)
        td_tgt = self.td_target(reward, next_state, done)

        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)
