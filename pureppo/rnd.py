import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3.common.callbacks import BaseCallback


class RNDNetwork(nn.Module):
    def __init__(self, input_channels=1, output_dim=512, device=torch.device('auto')):
        super(RNDNetwork, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(7 * 7 * 64, output_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)
        return x


def initialize_rnd(input_channels=1, output_dim=512):
    target_network = RNDNetwork(input_channels, output_dim)
    predictor_network = RNDNetwork(input_channels, output_dim)

    # Freeze the target network parameters
    for param in target_network.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(predictor_network.parameters(), lr=0.001)

    return target_network, predictor_network, optimizer


class RNDCustomCallback(BaseCallback):
    def __init__(self, target_network, predictor_network, optimizer, verbose=0):
        super(RNDCustomCallback, self).__init__(verbose)
        self.target_network = target_network
        self.predictor_network = predictor_network
        self.optimizer = optimizer

    def _on_step(self) -> bool:
        obs = self.locals["new_obs"]
        obs_tensor = torch.tensor(obs, dtype=torch.float32)

        target_output = self.target_network(obs_tensor)
        predictor_output = self.predictor_network(obs_tensor)

        intrinsic_reward = torch.mean((target_output - predictor_output) ** 2).item()

        # Update the predictor network
        self.optimizer.zero_grad()
        loss = torch.mean((target_output - predictor_output) ** 2)
        loss.backward()
        self.optimizer.step()

        # Add the intrinsic reward
        self.locals["rewards"] += intrinsic_reward

        return True
