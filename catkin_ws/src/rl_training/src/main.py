#!/usr/bin/env python3

import math
import time

import numpy as np

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetModelState, SetModelState
from std_srvs.srv import Empty

from std_msgs.msg import Float32

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
            "odom": gym.spaces.Box(
                low=np.array([-np.inf, -np.inf, -math.pi], dtype=np.float32),
                high=np.array([ np.inf,  np.inf,  math.pi], dtype=np.float32),
                shape=(3,),
                dtype=np.float32
            )
        })

        self.lidar = np.zeros(360, dtype=np.float32)

        # Initial state: Position
        rospy.wait_for_service('/gazebo/get_model_state')
        get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        model_state = get_model_state("turtlebot3", "world")

        # Reset Gazebo and odometry
        rospy.wait_for_service('/gazebo/reset_world')
        self.reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)

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

        # LiDAR threshold
        self.lidar_threshold = 0.7

        # Dist left and right
        self.lidar_left = 10
        self.lidar_right= 10

        # Waypoint threshold
        self.goal_threshold = 1.5

        # Dist waypoint threshold
        self.dist_epsilon = 0.5

        # Last waypoint reward
        self.last_waypoint_reward = 0

        # Waypoints
        self.waypoint = [[-2.737936, 8.929284], [4.558541, 9.021430],[7.752216, 7.972296], [7.027048, 7.027048], [7.703240, 4.045793], [7.167124, 2.227963], [9.039780, -5.957927], [1.315660, -8.974951], [-6.588877, -9.004139], [-8.894380, -5.594486], [-6.242726, -1.863769], [-2.014763, -3.952699], [2.206598, -0.400605], [-0.313865, 1.972752], [-6.928076, 2.018310], [-8.195169, 8.364141], [-2.737936, 8.929284]]

        # Waypoint counter
        self.waypoint_counter = 0

        # Waypoint last dist
        self.last_dist_first = 99999

        # ROS Movement Publisher
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.rate = rospy.Rate(10)

        # ROS LIDAR Subscriber
        rospy.Subscriber("/scan", LaserScan, self._laser_callback)

        # ROS Odometry Subscriber
        rospy.Subscriber('/odom', Odometry, self._odom_callback)

        # Reward publisher
        self.episode_reward_pub = rospy.Publisher('/rl/episode_reward', Float32, queue_size=1)
        self.episode_reward = 0.0

        # Distance travelled threshold to get reward
        self.distance_travelled_threshold = 0.5

        # Last distance reward time
        self.last_distance_reward_time = time.time()

        # Odom
        self.odom = None
        self.last_odom = [model_state.pose.position.x, model_state.pose.position.y]

        # Sum of angular velocity
        self.angular_velocity_sum = 0.0

        # Collision timer
        self.last_moved_time = time.time()
        self.last_moved_position = None

    def _laser_callback(self, msg):
        # Filter invalid values
        raw = np.array([r if (not math.isinf(r) and not math.isnan(r) and r<3.5) else 10.0 for r in msg.ranges], dtype=np.float32)
        self.raw_scan = raw
        
        # Get only data from [250º, 360º] U [0º, 110º]
        front_data = np.zeros(220, dtype=np.float32)
        front_data[:110] = raw[250:]
        front_data[110:] = raw[:110]
        
        sectors = np.array_split(front_data, self.num_sectors)
        
        self.lidar = np.array([np.min(s) for s in sectors], dtype=np.float32)
        
        self.lidar_left, self.lidar_right = self.get_dist(raw)

    def get_dist(self, raw):
        N = len(raw)
        deg2idx = lambda deg: int(deg * N / 360.0)

        li, lf = deg2idx(70),  deg2idx(110)
        ri, rf = deg2idx(250), deg2idx(290)

        left_vals  = raw[li: lf+1]
        right_vals = raw[ri: rf+1]

        lidar_left  = float(np.max(left_vals))   if left_vals.size  > 0 else float('inf')
        lidar_right = float(np.max(right_vals))  if right_vals.size > 0 else float('inf')

        return lidar_left, lidar_right
        
    
    def _odom_callback(self, msg):
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        siny = 2*(o.w*o.z + o.x*o.y)
        cosy = 1 - 2*(o.y*o.y + o.z*o.z)
        yaw = math.atan2(siny, cosy)
        self.odom = np.array([p.x, p.y, yaw], dtype=np.float32)

    def _has_collisioned(self):
        if self.odom is not None and self.last_moved_position is not None:
            dist = np.linalg.norm(self.odom[:2] - self.last_moved_position)
            if dist > 0.05:
                self.last_moved_time = time.time()
                self.last_moved_position = self.odom[:2]
            else:
                if time.time() - self.last_moved_time > 3.0:
                    rospy.loginfo("Last_moved_pos = " + str(self.last_moved_position) + ", Actual pos = " + str(self.odom[:2]))
                    return True
        return False

    def _get_obs(self):
        return {
            "lidar": self.lidar.copy(),
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

            reward = -300

            return obvs, reward, True, False, info

        if self.waypoint_counter == len(self.waypoint) + 1:
            self.pub.publish(Twist())
            reward = 5000
            return obvs, reward, True, False, info
        

        # Get velocity from the action
        linear_velocity, angular_velocity = action

        
        if(self.lidar_left == 10.0 and self.lidar_right == 10.0):
            reward = -50
        else:
            dist_diff = abs(self.lidar_left - self.lidar_right)
            reward = -dist_diff*2.5

        rospy.loginfo("Dist_reward = " + str(reward))
        rospy.loginfo("Distancias: " + str(self.lidar_left) + "," + str(self.lidar_right))

        # Linear reward
        linear_reward = 30 * linear_velocity
        reward += linear_reward

        # Penalize angular velocity
        angular_penalty = -10 * abs(angular_velocity)
        reward += angular_penalty

        rospy.loginfo("Angular penalty = " + str(angular_penalty))

        # Calculate distance traveled since last reward
        if self.odom is not None:
            first_wp = np.array(self.waypoint[0], dtype=np.float32)
            dist_first = np.linalg.norm(self.odom[:2] - first_wp)
            # Give reward for getting closer to the waypoint
            waypoint_reward = 1/(dist_first + self.dist_epsilon) * 10 + self.last_waypoint_reward
            if dist_first < self.goal_threshold:
                wp = self.waypoint.pop(0)
                self.waypoint.append(wp)
                self.last_waypoint_reward = waypoint_reward
                self.waypoint_counter += 1
                self.last_dist_first = 99999

            elif dist_first > self.last_dist_first:
                waypoint_reward = 0

            reward += waypoint_reward
            self.last_dist_first = dist_first
            
            rospy.loginfo("Distancia waypoint = " + str(dist_first))
            rospy.loginfo("Waypoint actual = " + str(self.waypoint_counter))
            rospy.loginfo("waypoint_reward = " + str(waypoint_reward))

            # Reset step pose
            self.last_odom = self.odom.copy()

        # Reset step time
        self.last_distance_reward_time = time.time()

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


        # Return step
        return obvs, reward, terminated, truncated, info

    def reset(self, seed=42):
        super().reset()

        rospy.loginfo("Reset env: total_reward=%.2f", self.episode_reward)

        # Reset Gazebo
        try:
            self.reset_world()
            rospy.loginfo("Se ha reseteado Gazebo")
        except rospy.ServiceException as e:
            rospy.logwarn(f"Error al resetear Gazebo: {e}")
        rospy.sleep(0.1)

        # Reset to initial position
        rospy.wait_for_service('/gazebo/set_model_state')
        set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        set_model_state(self.initial_state)

        # zero lidar and image
        self.lidar = np.zeros(self.num_sectors, dtype=np.float32)

        # Reset waypoints
        self.waypoint = [[-2.737936, 8.929284], [4.558541, 9.021430],[8.009294, 7.972296], [6.511956, 6.144635], [8.442575, 4.045553], [6.313862, 2.227839], [9.039780, -5.957927], [1.315660, -8.974951], [-6.588877, -9.004139], [-8.894380, -5.594486], [-6.242726, -1.863769], [-2.014763, -3.952699], [2.206598, -0.400605], [-0.313865, 1.972752], [-6.928076, 2.018310], [-8.195169, 8.364141]]

        # Reset waypoint last reward
        self.last_waypoint_reward = 0

        # Reset waypoint counter
        self.waypoint_counter = 0

        # Reset waypoint last dist
        self.last_dist_first = 99999

        # Reset odom info
        self.last_odom = [self.initial_state.pose.position.x, self.initial_state.pose.position.y]
        self.last_moved_position = self.last_odom

        # Reset collision counter
        self._collision_counter = 0
        self.last_moved_time = time.time()

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
        buffer_size=700_000,
        verbose=1,
    )

    while True:
        rospy.loginfo("Iteration starts")

        model.learn(
            total_timesteps=100_000,
            reset_num_timesteps=False,
            tb_log_name="rl_training"
        )

        model.save("rl_training_model")