import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN


class LidarImageExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)

        # Extract dimensions
        lidar_dim = observation_space.spaces['lidar'].shape[0]
        image_space = observation_space.spaces['image']

        # CNN for image processing
        self.cnn = NatureCNN(
            image_space,
            normalized_image=False
        )

        # Linear for lidar processing
        self.lidar_net = nn.Sequential(
            nn.Linear(lidar_dim, 256),
            nn.ReLU(),
        )

        # Combined network
        self.linear = nn.Sequential(
            nn.Linear(256 + 512, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        # Process image
        image_obs = observations['image']
        image_features = self.cnn(image_obs)

        # Process lidar
        lidar_features = self.lidar_net(observations['lidar'])

        # Combine features
        combined = torch.cat([image_features, lidar_features], dim=1)
        return self.linear(combined)
