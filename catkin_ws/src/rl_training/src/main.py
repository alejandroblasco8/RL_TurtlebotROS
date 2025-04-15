#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import numpy as np
import gymnasium as gym
import math
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env

class TrainingEnv(gym.Env):
    def __init__(self):
        # Action space
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )
        # Observation space: se espera un array de 360 elementos en float32
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(360,),
            dtype=np.float32
        )
        # Inicializa el estado con float32
        self.state = np.zeros(360, dtype=np.float32)

        self.filtered_ranges = []
        self.filtered_angle_min = None
        self.filtered_angle_increment = None
        self.front_window = math.radians(15)
        self.side_window  = math.radians(10)
        
        # Configuración de ROS
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.rate = rospy.Rate(10)
        rospy.Subscriber("/scan", LaserScan, self.laser_callback)
    
    def laser_callback(self, msg):
        """
        Callback que se invoca cada vez que llega un mensaje del láser.
        Se procesa el mensaje para filtrar y almacenar las lecturas válidas.
        """
        self._get_info(msg)

    def _get_obs(self):
        """
        En este ejemplo, puedes actualizar el estado (self.state) con información relevante.
        Por ejemplo, podrías combinar datos del láser procesados (distancias frontales, laterales, etc.).
        Aquí se muestra cómo se podrían utilizar las distancias:
        """
        # Ejemplo: el estado podría ser un vector con tres elementos
        front = self.get_front_distance()
        left = self.get_left_distance()
        right = self.get_right_distance()
        
        # Para efectos de esta demostración se crea un vector de 360 elementos 
        # en el que se podrían insertar los valores medidos. En un caso real,
        # deberías definir cómo representarás la información.
        self.state[:3] = np.array([front, left, right], dtype=np.float32)
        return self.state
    

    def valid_value(self, val):
        """
            Se añade un filtro para los valores obtenidos del láser.
            Se comprueba que no sea "inf".
            Se comprueba que no sea "NaN"
            Se comprueba que el valor no sea mayor a 3.5 (valor seleccionado para eliminar mediciones no confiables)
        """
        return (not math.isinf(val)) and (not math.isnan(val)) and (val < 3.5)

    def angle_of_index(self, msg, i):
        angle_min_raw = msg.angle_min
        angle_inc_raw = msg.angle_increment
        return angle_min_raw + i * angle_inc_raw

    def _get_info(self, msg):
        # Return any additional information

        #Se obtienen los datos del láser
        raw_ranges = msg.ranges
        n = len(raw_ranges)
        #Se almacena el incremento que hay de ángulo entre cada medición
        angle_inc_raw = msg.angle_increment
        
        #Se limita la obtención a +-150
        limit_rad = math.radians(150)

        #Se comprueba cuál es el subconjunto de los datos del láser a obtener
        i_min = 0
        while i_min < n and self.angle_of_index(msg, i_min) < -limit_rad:
            i_min += 1
        i_max = n - 1
        while i_max >= 0 and self.angle_of_index(msg, i_max) > limit_rad:
            i_max -= 1
        if i_min >= i_max:
            self.filtered_ranges = []
            return
    
        #Se guarda el subconjunto extraido.
        self.filtered_ranges = list(raw_ranges[i_min:i_max+1])
        self.filtered_angle_min = self.angle_of_index(msg, i_min)
        self.filtered_angle_increment = angle_inc_raw

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

        # En un entorno real, se podría esperar a que se actualicen los datos del láser
        # antes de recoger la observación. Aquí se simula el avance con _get_obs()
        obvs = self._get_obs()
        info = {}
        return obvs, reward, terminated, truncated, info

    
    def reset(self, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)
        else:
            super().reset()
            
        # Reinicia el estado con float32
        self.state = np.zeros(360, dtype=np.float32)
        obvs = self._get_obs()
        info = {}
        return obvs, info
    
    def get_range_average(self, center_ang, amplitude):
        if not self.filtered_ranges or self.filtered_angle_min is None:
            return 999.0
        
        #Se recupera el incremento angular de las mediciones
        inc = self.filtered_angle_increment

        #Indice correpondiente al ángulo central del subconjunto
        center_idx = int((center_ang - self.filtered_angle_min) / inc)

        #Comprobación de que el índice esté dentro del rango válido
        center_idx = max(0, min(center_idx, len(self.filtered_ranges)-1))

        #Se obtiene el número de mediciones.
        amplitude = int(amplitude / inc)

        #Indice inicial
        s = max(center_idx - amplitude, 0)

        #Indice final
        e = min(center_idx + amplitude, len(self.filtered_ranges)-1)

        #Lista que almacena las mediciones dentro del rango definido por [s, e] que cumple con el filtro de los valores
        vals = [i for i in self.filtered_ranges[s:e+1] if self.valid_value(i)]

        #Si hay mediciones válidas, se devuelve su promedio
        if vals:
            avg = sum(vals)/len(vals)
            return avg
        
        return 999.0

    #Se obtiene la distancia con el primer obstáculo al frente del robot
    def get_front_distance(self):
        return self.get_range_average(0.0, self.front_window)

    #Se obtiene la distancia con el primer obstáculo a la izquierda del robot
    def get_left_distance(self):
        left_center = math.pi/2
        avg = self.get_range_average(left_center, self.side_window)
        return avg

    #Se obtiene la distancia con el primer obstáculo a la derecha del robot
    def get_right_distance(self):
        right_center = -math.pi/2
        return self.get_range_average(right_center, self.side_window)



if __name__ == "__main__":
    try:
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
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Error in RL training: {e}")