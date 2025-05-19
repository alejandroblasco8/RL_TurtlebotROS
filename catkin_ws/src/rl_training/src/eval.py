#!/usr/bin/env python3
import rospy
import numpy as np
from stable_baselines3 import SAC
from geometry_msgs.msg import Twist
from main import TrainingEnv

if __name__ == "__main__":
    rospy.init_node("rl_evaluate")

    env = TrainingEnv()
    model = SAC.load("ModeloAEntregar.zip", env=env)

    num_episodes = 5
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0

        while not done and not rospy.is_shutdown():
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            ep_reward += reward

        rospy.loginfo(f"[Eval] Episode {ep+1} reward: {ep_reward:.2f}")

    rospy.loginfo("Evaluación completada")
