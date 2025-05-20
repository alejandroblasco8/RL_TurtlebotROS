#!/usr/bin/env python3
import rospy
import zipfile
import io
import torch
from stable_baselines3 import SAC
from geometry_msgs.msg import Twist
from main import TrainingEnv

def load_policy_from_zip(zip_path, model, map_location='cpu'):
    with zipfile.ZipFile(zip_path, 'r') as z:
        buf = z.read('policy.pth')
    state_dict = torch.load(io.BytesIO(buf), map_location=map_location)
    model.policy.load_state_dict(state_dict)

if __name__ == "__main__":
    rospy.init_node("rl_evaluate")

    env = TrainingEnv()

    model = SAC(
        "MultiInputPolicy",
        env,
        buffer_size=700_000,
        verbose=1,
    )

    load_policy_from_zip("rl_training_model.zip", model)

    num_episodes = 1
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_rew = 0.0

        while not done and not rospy.is_shutdown():
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            total_rew += reward

        rospy.loginfo(f"[Eval] Episodio {ep+1}: recompensa = {total_rew:.2f}")

    rospy.loginfo("=== Evaluación completada ===")
