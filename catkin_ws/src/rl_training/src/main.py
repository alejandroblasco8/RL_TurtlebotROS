#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist

import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env

class TrainingEnv(gym.Env):
    def __init__(self):
        # Action space: Linear velocity and angular velocity
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

        # Observation space: LIDAR sensor
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(360,),
            dtype=np.float32
        )

        # Initial state: Empty LIDAR data
        self.state = np.zeros(360)

        # ROS specific configuration
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.rate = rospy.Rate(10)

    def _get_obs(self):
        return {}

    def _get_info(self):
        return {}

    def step(self, action):
        # Get velocity from the action
        linear_velocity, angular_velocity = action

        # Send control ROS message
        msg = Twist()
        msg.linear.x = linear_velocity
        msg.angular.z = angular_velocity
        self.pub.publish(msg)

        # Mock reward
        reward = np.random.rand()

        # Check if terminated or truncated
        terminated = False
        truncated = False

        # Get observations and extra information
        obvs = self._get_obs()
        info = self._get_info()

        # Return step
        return obvs, reward, terminated, truncated, info

    def reset(self, seed):
        super().reset()
        self.state = np.zeros(360)

        obvs = self._get_obs()
        info = self._get_info()

        return obvs, info

if __name__ == "__main__":
    rospy.loginfo("Starting RL Training...")
    rospy.init_node("rl_training")

    env = TrainingEnv()
    check_env(env, warn=True)

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        gamma=0.99,
        buffer_size=100000,  # Size of the replay buffer.
        learning_starts=1000  # Number of steps before training starts.
    )
    
    model.learn(total_timesteps=10000)