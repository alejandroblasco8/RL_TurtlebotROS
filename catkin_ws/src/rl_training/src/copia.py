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

class WaypointEnv(gym.Env):
    metadata = {"render.modes": []}

    def __init__(self):
        super().__init__()
        # single waypoint
        self.waypoint = np.array([8.419438, 8.847774], dtype=np.float32)
        self.goal_threshold = 0.5  # meters
        self.collision_threshold = 0.12  # m
        self.collision_confirm_steps = 3  # number of consecutive readings
        self._collision_counter = 0

        # wait for first scan to determine raw LiDAR size
        first_scan = rospy.wait_for_message('/scan', LaserScan)
        self.scan_size = len(first_scan.ranges)

        # --- Action & observation spaces ---
        low_action = np.array([0.0, -1.0], dtype=np.float32)
        high_action = np.array([1.0,  1.0], dtype=np.float32)
        self.action_space = spaces.Box(low=low_action, high=high_action, dtype=np.float32)

        obs_dim = 3 + self.scan_size  # odom x,y,yaw + raw scan
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # ROS publishers & subscribers
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        rospy.Subscriber('/odom', Odometry, self._odom_cb)
        rospy.Subscriber('/scan', LaserScan, self._scan_cb)

        # state
        self.odom = None
        self.scan = np.zeros(self.scan_size, dtype=np.float32)
        self.last_odom = None
        self._last_v = 0.0
        self._last_a = 0.0

        # prepare SetModelState service for custom reset
        rospy.wait_for_service('/gazebo/set_model_state')
        get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        model_state = get_model_state("turtlebot3", "world")

        # define initial ModelState
        self.initial_state = ModelState()
        self.initial_state.model_name = "turtlebot3"
        self.initial_state.pose = model_state.pose
        self.initial_state.twist.linear.x = 0.0
        self.initial_state.twist.linear.y = 0.0
        self.initial_state.twist.linear.z = 0.0
        self.initial_state.twist.angular.x = 0.0
        self.initial_state.twist.angular.y = 0.0
        self.initial_state.twist.angular.z = 0.0
        self.initial_state.reference_frame = "world"
        # episode timer
        self.episode_start = None

    def _odom_cb(self, msg):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        siny = 2.0 * (ori.w * ori.z + ori.x * ori.y)
        cosy = 1.0 - 2.0 * (ori.y**2 + ori.z**2)
        yaw = np.arctan2(siny, cosy)
        self.odom = np.array([pos.x, pos.y, yaw], dtype=np.float32)

    def _scan_cb(self, msg):
        # apply 'valid' filter to raw ranges
        valid = lambda x: x if (not math.isinf(x)) and (not math.isnan(x)) and (x < 3.5) else 10.0
        raw = np.array([valid(r) for r in msg.ranges], dtype=np.float32)
        # store raw LiDAR
        self.scan = raw

    def _get_obs(self):
        return np.concatenate([self.odom, self.scan])

    def reset(self, seed=42):
        super().reset()
        rospy.loginfo("Resetting environment: moving robot to initial state.")

        # Reset to initial position
        rospy.wait_for_service('/gazebo/set_model_state')
        set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        set_model_state(self.initial_state)

        self.last_odom = None
        self._last_v = 0.0
        self._last_a = 0.0
        self._collision_counter = 0

        # Get observations and info
        obvs = self._get_obs()

        return obvs, {}

    def step(self, action):
        # command robot
        cmd = Twist()
        cmd.linear.x, cmd.angular.z = [float(a) for a in action]
        self.cmd_pub.publish(cmd)
        rospy.sleep(0.1)

        obs = self._get_obs()
        reward = self._compute_reward(obs)
        done = False

        # temporal collision validation
        if np.min(self.scan) <= self.collision_threshold:
            self._collision_counter += 1
        else:
            self._collision_counter = 0

        if self._collision_counter >= self.collision_confirm_steps:
            rospy.loginfo(
                "Episode done: Collision confirmed after %d consecutive steps. scan_min=%.3f <= threshold=%.3f",
                self._collision_counter, np.min(self.scan), self.collision_threshold
            )
            reward -= 5.0
            self.cmd_pub.publish(Twist())
            done = True

        # goal reached
        elif np.linalg.norm(obs[:2] - self.waypoint) < self.goal_threshold:
            rospy.loginfo("Episode done: Waypoint reached. distance=%.3f <= threshold=%.3f",
                          np.linalg.norm(obs[:2] - self.waypoint), self.goal_threshold)
            done = True

        self.last_odom = self.odom.copy()
        return obs, reward, done, False, {}

    def _compute_reward(self, obs):
        # progress towards waypoint
        if self.last_odom is None:
            r_prog = 0.0
        else:
            prev = np.linalg.norm(self.last_odom[:2] - self.waypoint)
            curr = np.linalg.norm(obs[:2] - self.waypoint)
            r_prog = prev - curr
        # centering using raw scan halves
        left = np.mean(obs[3:3+self.scan_size//2])
        right = np.mean(obs[3+self.scan_size//2:])
        r_center = -0.5 * abs(left - right)
        # speed & jerk
        v = np.linalg.norm(obs[:2] - (self.last_odom[:2] if self.last_odom is not None else obs[:2]))/0.1
        a = (v - self._last_v)/0.1
        jerk = abs(a - self._last_a)
        r_speed = 0.5 * min(v,1.0)
        r_jerk = -0.2 * jerk
        self._last_v, self._last_a = v, a
        # proximity penalty
        min_d = np.min(obs[3:])
        r_obs = -1.0 * max(0.0, 0.2 - min_d)
        return float(r_prog + r_center + r_speed + r_jerk + r_obs)

if __name__ == '__main__':
    rospy.init_node('sac_single_waypoint')
    env = Monitor(WaypointEnv(), filename=None, allow_early_resets=True)
    model = SAC(
        'MlpPolicy',
        env,
        buffer_size=500000,
        learning_starts=5000,
        batch_size=256,
        train_freq=1,
        gradient_steps=4,
        tau=0.005,
        ent_coef='auto',
        learning_rate=3e-4,
        verbose=1
    )
    model.learn(total_timesteps=500000)
    model.save('sac_waypoint_model')
