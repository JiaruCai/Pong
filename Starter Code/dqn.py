from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.autograd as autograd
import math, random
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

class QLearner(nn.Module):
    def __init__(self, env, num_frames, batch_size, gamma, replay_buffer):
        super(QLearner, self).__init__()

        self.batch_size = batch_size
        self.gamma = gamma
        self.num_frames = num_frames
        self.replay_buffer = replay_buffer
        self.env = env
        self.input_shape = self.env.observation_space.shape
        self.num_actions = self.env.action_space.n

        self.features = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def feature_size(self):
            return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)
    
    def act(self, state, epsilon):
        if random.random() > epsilon:
            state = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0), requires_grad=True)
            # TODO: Given state, you should write code to get the Q value and chosen action
        

            result = self.forward(state)[0]
#             print(result)
#             print(result.max(0))
            _, action = result.max(0)


        else:
            action = random.randrange(self.env.action_space.n)
        return action

    def copy_from(self, target):
        self.load_state_dict(target.state_dict())

        
def compute_td_loss(model, target_model, batch_size, gamma, replay_buffer):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = Variable(torch.FloatTensor(np.float32(state)))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)).squeeze(1), requires_grad=True)
    action = Variable(torch.LongTensor(action))
    reward = Variable(torch.FloatTensor(reward))
#     done = Variable(torch.floatTensor(done))
    done = Variable(torch.LongTensor(done))
    # implement the loss function here
    
    state = state.squeeze(1)
    model_q_values = model.forward(state)
    target_q_values = target_model.forward(next_state).detach()
#     target_q_values[done] = 0
    
#     print(model_q_values)
    
    target_output, target_actions = target_q_values.max(1)
#     print(target_actions)
#     _, model_actions = model_q_values.max(1)
#     model_output = model_q_values.gather(1, target_actions.unsqueeze(-1)).squeeze(1)
    model_output = model_q_values.gather(1, action.unsqueeze(-1)).squeeze(1)
#     model_actions = model_actions.type(torch.cuda.FloatTensor)
#     target_actions = target_actions.type(torch.cuda.FloatTensor)
#     print(model_actions)

    target_output[done] = 0
#     model_output[done] = 0
    
    update = reward + gamma * target_output
#     print(update)

    lossf = nn.MSELoss()
    loss = lossf(model_output, update)
#     print(model_output)
#     print(update)
#     print(loss)
    
    return loss
    


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # TODO: Randomly sampling data with specific batch size from the buffer
        
        Is = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        samples = []
        
        for i in Is:
            samples.append(self.buffer[i])
        
        samples_T = list(map(list, zip(*samples)))
        
        state, action, reward, next_state, done = samples_T

        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
