#!/usr/bin/env python3

import math
import time

import numpy as np

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env

from cv_bridge import CvBridge
import cv2

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, Image
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetModelState, SetModelState

from std_msgs.msg import Float32, Int32

from extractor import LidarImageExtractor


class TrainingEnv(gym.Env):
    def __init__(self):
        # Action space: Linear velocity and angular velocity
        self.action_space = gym.spaces.Box(
            low=np.array([0.0, -1.82]),
            high=np.array([0.26, 1.82]),
            dtype=np.float32
        )

        #LiDAR num sectors
        self.num_sectors = 16

        # Observation space: LIDAR + IMAGE
        self.observation_space = gym.spaces.Dict({
            "lidar": gym.spaces.Box(
                low=0.0,
                high=10.0,
                shape=(self.num_sectors,),
                dtype=np.float32
            ),
            "image": gym.spaces.Box(
                low=0,
                high=255,
                shape=(1, 64, 64),
                dtype=np.uint8
            ),
            "odom": gym.spaces.Box(
                low=np.array([-np.inf, -np.inf, -math.pi], dtype=np.float32),
                high=np.array([ np.inf,  np.inf,  math.pi], dtype=np.float32),
                shape=(3,),
                dtype=np.float32
            )
        })

        self.image = np.zeros((1, 64, 64), dtype=np.uint8)
        self.lidar = np.zeros(360, dtype=np.float32)

        # Initial state: Position
        rospy.wait_for_service('/gazebo/get_model_state')
        get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        model_state = get_model_state("turtlebot3", "world")

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

        # Collision LIDAR threshold
        self.collision_threshold = 0.12

        # Collision counter
        self._collision_counter = 0

        # Confirm collision number
        self.collision_confirm_steps = 3

        #Waypoint threshold
        self.goal_threshold = 0.5

        #Waypoints
        self.waypoint = [[8.419438, 8.847774]]

        # ROS Movement Publisher
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.rate = rospy.Rate(10)

        # ROS LIDAR Subscriber
        rospy.Subscriber("/scan", LaserScan, self._laser_callback)

        # ROS Odometry Subscriber
        rospy.Subscriber('/odom', Odometry, self._odom_callback)

        # ROS Image Subscriber
        rospy.Subscriber("/camera/rgb/image_raw", Image, self._image_callback)

        # Reward publisher
        self.episode_reward_pub = rospy.Publisher('/rl/episode_reward', Float32, queue_size=1)
        self.episode_reward = 0.0

        # CV2 bridge
        self.bridge = CvBridge()

        # Distance travelled threshold to get reward
        self.distance_travelled_threshold = 0.5

        # Last distance reward time
        self.last_distance_reward_time = time.time()

        # Odom
        self.odom = None
        self.last_odom = [model_state.pose.position.x, model_state.pose.position.y]

        # Sum of angular velocity
        self.angular_velocity_sum = 0.0

    def _image_callback(self, img):
        cv_image = self.bridge.imgmsg_to_cv2(img, desired_encoding="bgr8")
        resized_image = cv2.resize(cv_image, (64, 64))
        gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

        self.image = np.expand_dims(gray_image, axis=0).astype(np.uint8)

    def _laser_callback(self, msg):
        # filter invalid
        raw = np.array([r if (not math.isinf(r) and not math.isnan(r) and r<3.5) else 10.0 for r in msg.ranges], dtype=np.float32)
        self.raw_scan = raw
        # sectorize
        sectors = np.array_split(raw, self.num_sectors)
        self.lidar = np.array([np.min(s) for s in sectors], dtype=np.float32)
    
    def _odom_callback(self, msg):
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        siny = 2*(o.w*o.z + o.x*o.y)
        cosy = 1 - 2*(o.y*o.y + o.z*o.z)
        yaw = math.atan2(siny, cosy)
        self.odom = np.array([p.x, p.y, yaw], dtype=np.float32)

    def _has_collisioned(self):
        if np.min(self.lidar) <= self.collision_threshold:
            self._collision_counter += 1
        else:
            self._collision_counter = 0
        if self._collision_counter >= self.collision_confirm_steps:
            return True
        return False

    def _get_obs(self):
        return {
            "lidar": self.lidar.copy(),
            "image": self.image.copy(),
            "odom" : self.odom.copy()
        }

    def _get_info(self):
        return {}

    def step(self, action):
        # Get observations and extra information
        obvs = self._get_obs()
        info = self._get_info()

        if self._has_collisioned():
            # Stop the robot before resetting
            self.pub.publish(Twist())

            reward = -20
            # rospy.loginfo(f"[Reward] Collision: {-100}")

            return obvs, reward, True, False, info

        # Get velocity from the action
        linear_velocity, angular_velocity = action

        # Positive reward for moving
        linear_reward = 2 * linear_velocity
        reward = linear_reward
        rospy.loginfo(f"[Reward] Linear reward: {linear_reward:.2f}")

        # Penalize angular velocity
        angular_penalty = -2 * abs(angular_velocity)
        reward += angular_penalty
        rospy.loginfo(f"[Reward] Angular penalty: {angular_penalty:.2f}")

        # Calculate distance traveled since last reward
        if self.odom is not None:
            euclidean = math.sqrt(
                (self.odom[0] - self.last_odom[0]) ** 2
                + (self.odom[1] - self.last_odom[1]) ** 2
            )

            # Give reward for travelled distance
            travelled_reward = euclidean * 10
            reward += travelled_reward
            rospy.loginfo(f"[Reward] Step distance travelled: {travelled_reward:.2f}")

            # Give reward for getting closer to the waypoint
            waypoint_distance = np.linalg.norm(self.odom[:2] - self.waypoint[0])
            if waypoint_distance < self.goal_threshold:
                waypoint_reward = 100
                wp = self.waypoint.pop(0)
                self.waypoint.append(wp)
            else:
                waypoint_reward = waypoint_distance * 10
            
            reward += waypoint_reward

            # Reset step pose
            self.last_odom = self.odom.copy()

        # Reset step time
        self.last_distance_reward_time = time.time()

        # Add angular velocity to the total sum
        self.angular_velocity_sum += angular_velocity
        rospy.loginfo(f"[Info] Angular-sum: {self.angular_velocity_sum:.2f}")

        # Penalize long angular movements
        reward -= abs(self.angular_velocity_sum) * 10
        rospy.loginfo(f"[Reward] Angular-sum penalty: {-abs(self.angular_velocity_sum) * 5:.2f}")

        # Add reward to the total episode reward
        self.episode_reward += reward

        # Send control ROS message
        msg = Twist()
        msg.linear.x = linear_velocity
        msg.angular.z = angular_velocity
        self.pub.publish(msg)

        # Terminated & Truncated
        terminated = False
        truncated = False

        rospy.loginfo(f"[Info] Step reward: {reward}")

        # Return step
        return obvs, reward, terminated, truncated, info

    def reset(self, seed=42):
        super().reset()

        # Reset to initial position
        rospy.wait_for_service('/gazebo/set_model_state')
        set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        set_model_state(self.initial_state)

        # zero lidar and image
        self.lidar = np.zeros(self.num_sectors, dtype=np.float32)
        self.image = np.zeros((1,64,64), dtype=np.uint8)

        # Reset odom info
        self.last_odom = [self.initial_state.pose.position.x, self.initial_state.pose.position.y]

        # Reset collision counter
        self._collision_counter = 0

        # Publish reward and reset the episode reward value
        self.episode_reward_pub.publish(self.episode_reward)
        self.episode_reward = 0.0

        # Reset sum of angular velocity and reward times
        self.angular_velocity_sum = 0.0

        # wait for valid odom
        while self.odom is None:
            rospy.sleep(0.01)

        return self._get_obs(), {}


if __name__ == "__main__":
    rospy.loginfo("Starting RL Training...")
    rospy.init_node("rl_training")

    env = TrainingEnv()
    check_env(env, warn=True)

    model = SAC(
        "MultiInputPolicy",
        env,
        buffer_size=50000,
        policy_kwargs=dict(
            features_extractor_class=LidarImageExtractor,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=[256, 256]
        ),
        verbose=1,
    )

    while True:
        rospy.loginfo("Iteration starts")

        model.learn(
            total_timesteps=10_000,
            reset_num_timesteps=False,
            tb_log_name="rl_training"
        )

        model.save(f"rl_training_model-{int(time.time())}")
