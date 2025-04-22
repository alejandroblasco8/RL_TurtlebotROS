#!/usr/bin/env python3
import rospy
import time
import math
import gymnasium as gym
from gymnasium import spaces
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from gazebo_msgs.srv import SetModelState, GetModelState
from gazebo_msgs.msg import ModelState
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from std_msgs.msg import Float32, Int32

class WaypointEnv(gym.Env):
    metadata = {"render.modes": []}

    def __init__(self):
        super().__init__()
        # single waypoint
        self.waypoint = np.array([8.419438, 8.847774], dtype=np.float32)
        self.goal_threshold = 0.5  # meters
        # collision threshold and temporal confirmation
        self.collision_threshold = 0.12
        self.collision_confirm_steps = 3
        self._collision_counter = 0
        # sectorization of LiDAR
        self.num_sectors = 16

        # get first scan to determine raw size
        first_scan = rospy.wait_for_message('/scan', LaserScan)
        self.raw_size = len(first_scan.ranges)

        # action & observation spaces
        low_action = np.array([0.0, -1.0], dtype=np.float32)
        high_action = np.array([1.0,  1.0], dtype=np.float32)
        self.action_space = spaces.Box(low=low_action, high=high_action, dtype=np.float32)
        obs_dim = 3 + self.num_sectors
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # ROS pubs/subs
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        rospy.Subscriber('/odom', Odometry, self._odom_cb)
        rospy.Subscriber('/scan', LaserScan, self._scan_cb)
        # Publishers for real-time monitoring
        self.step_reward_pub = rospy.Publisher('/rl/step_reward', Float32, queue_size=1)
        self.episode_reward_pub = rospy.Publisher('/rl/episode_reward', Float32, queue_size=1)
        self.episode_length_pub = rospy.Publisher('/rl/episode_length', Int32, queue_size=1)

        # internal state
        self.odom = None
        self.raw_scan = np.zeros(self.raw_size, dtype=np.float32)
        self.scan = np.zeros(self.num_sectors, dtype=np.float32)
        self.last_odom = None
        self._last_v = 0.0
        self._last_a = 0.0
        self.episode_reward = 0.0
        self.episode_length = 0

        # Gazebo reset
        rospy.wait_for_service('/gazebo/get_model_state')
        get_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        ms = get_state('turtlebot3', 'world')
        self.initial_state = ModelState(model_name='turtlebot3', pose=ms.pose, twist=ms.twist, reference_frame='world')
        rospy.wait_for_service('/gazebo/set_model_state')
        self.set_state_srv = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

    def _odom_cb(self, msg):
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        siny = 2*(o.w*o.z + o.x*o.y)
        cosy = 1 - 2*(o.y*o.y + o.z*o.z)
        yaw = math.atan2(siny, cosy)
        self.odom = np.array([p.x, p.y, yaw], dtype=np.float32)

    def _scan_cb(self, msg):
        # filter invalid
        raw = np.array([r if (not math.isinf(r) and not math.isnan(r) and r<3.5) else 10.0 for r in msg.ranges], dtype=np.float32)
        self.raw_scan = raw
        # sectorize
        sectors = np.array_split(raw, self.num_sectors)
        self.scan = np.array([np.min(s) for s in sectors], dtype=np.float32)

    def _get_obs(self):
        return np.concatenate([self.odom, self.scan])

    def reset(self, seed=42):
        super().reset()
        # publish episode summary
        self.episode_reward_pub.publish(self.episode_reward)
        self.episode_length_pub.publish(self.episode_length)
        rospy.loginfo("Reset env: total_reward=%.2f, length=%d", self.episode_reward, self.episode_length)
        # reset gazebo
        self.set_state_srv(self.initial_state)
        time.sleep(0.2)
        # reset counters
        self.last_odom = None
        self._last_v = self._last_a = 0.0
        self._collision_counter = 0
        self.episode_reward = 0.0
        self.episode_length = 0
        # wait for valid
        while self.odom is None:
            rospy.sleep(0.01)
        return self._get_obs(), {}

    def step(self, action):
        # send command
        cmd = Twist(); cmd.linear.x, cmd.angular.z = action
        self.cmd_pub.publish(cmd)
        rospy.sleep(0.1)

        obs = self._get_obs()
        rospy.loginfo("LiDAR sectors@step %d: %s",
                self.episode_length,
                np.round(self.scan, 3).tolist())
        # compute reward with boosting
        # progress reward scaled
        if self.last_odom is None:
            r_prog = 0.0
        else:
            prev_d = np.linalg.norm(self.last_odom[:2] - self.waypoint)
            curr_d = np.linalg.norm(obs[:2] - self.waypoint)
            r_prog = (prev_d - curr_d) * 20.0
        # speed bonus
        v = action[0]
        r_speed = 0.05 * v
        # angular penalty
        r_ang = -0.1 * abs(action[1])
        # centering
        left = np.mean(self.scan[:self.num_sectors//2])
        right = np.mean(self.scan[self.num_sectors//2:])
        r_center = -0.5 * abs(left - right)
        # obstacle proximity penalty
        r_obs = -1.0 * max(0.0, 0.2 - np.min(self.scan))
        reward = r_prog + r_speed + r_ang + r_center + r_obs

        # update episode metrics
        self.episode_reward += reward
        self.episode_length += 1
        self.step_reward_pub.publish(reward)

        # check done
        done = False
        if np.min(self.scan) <= self.collision_threshold:
            self._collision_counter += 1
        else:
            self._collision_counter = 0
        if self._collision_counter >= self.collision_confirm_steps:
            rospy.loginfo("Collision confirmed at step %d", self.episode_length)
            reward -= 20.0; done = True
        elif np.linalg.norm(obs[:2] - self.waypoint) < self.goal_threshold:
            rospy.loginfo("Waypoint reached at step %d", self.episode_length)
            done = True

        self.last_odom = self.odom.copy()
        return obs, reward, done, False, {}

if __name__ == '__main__':
    rospy.init_node('sac_single_waypoint')
    env = Monitor(WaypointEnv(), filename=None, allow_early_resets=True)
    model = SAC(
        'MlpPolicy', env,
        buffer_size=500_000,
        learning_starts=5_000,
        batch_size=256,
        train_freq=1,
        gradient_steps=8,
        tau=0.005,
        ent_coef=0.01,
        learning_rate=5e-4,
        verbose=1,
        tensorboard_log='./sac_logs/'
    )
    model.learn(total_timesteps=500_000)
    model.save('sac_waypoint_model')
