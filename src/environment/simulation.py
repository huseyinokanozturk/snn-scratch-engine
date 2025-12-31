import pygame
import numpy as np
import math

# Colors (RGB)
WHITE = (255, 255, 255)
BLACK = (20, 20, 20)
RED = (200, 50, 50)     # Poison / Wall
GREEN = (50, 200, 50)   # Food
BLUE = (50, 100, 200)   # Agent
YELLOW = (200, 200, 50) # Sensor Rays

class Environment:
    def __init__(self, width=800, height=600):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("RHEO SNN Simulation")
        self.clock = pygame.time.Clock()

        # --- Agent Settings ---
        self.agent_radius = 15
        self.agent_pos = np.array([width/2, height/2], dtype=float)
        self.agent_angle = 0.0 # In Radians (0 = East)
        self.agent_speed = 3.0
        
        # Sensor Settings (Eyes)
        # 5 eyes at angles: -60, -30, 0, 30, 60 degrees
        self.sensor_angles = np.radians([-60, -30, 0, 30, 60])
        self.sensor_range = 150 # Vision range (pixels)
        self.sensors = np.zeros(len(self.sensor_angles)) # Read values

        # --- Map Objects ---
        self.foods = []
        self.walls = [] # (x, y, w, h)
        
        # Create a simple map
        self.build_default_map()
        self.reset()

    def build_default_map(self):
        """Add walls and obstacles"""
        # Screen border (Walls)
        self.walls.append(pygame.Rect(0, 0, self.width, 20)) # Top
        self.walls.append(pygame.Rect(0, self.height-20, self.width, 20)) # Bottom
        self.walls.append(pygame.Rect(0, 0, 20, self.height)) # Left
        self.walls.append(pygame.Rect(self.width-20, 0, 20, self.height)) # Right
        
        # An obstacle in the middle
        self.walls.append(pygame.Rect(300, 200, 50, 200))

    def reset(self):
        """Reset simulation"""
        self.agent_pos = np.array([100.0, 100.0])
        self.agent_angle = 0.0
        
        # Distribute food randomly
        self.foods = []
        for _ in range(5):
            self.spawn_food()
            
        return self.get_state()

    def spawn_food(self):
        """Spawn food avoiding walls"""
        while True:
            x = np.random.randint(50, self.width - 50)
            y = np.random.randint(50, self.height - 50)
            rect = pygame.Rect(x, y, 10, 10)
            
            # Check for wall collision
            if rect.collidelist(self.walls) == -1:
                self.foods.append(rect)
                break

    def raycast(self):
        """Lidar logic: Raycast from agent eyes to measure distance to walls"""
        readings = []
        
        for angle in self.sensor_angles:
            # Ray angle = Agent Direction + Sensor Angle
            ray_angle = self.agent_angle + angle
            
            # Ray vector (Unit)
            dir_x = math.cos(ray_angle)
            dir_y = math.sin(ray_angle)
            
            # Find closest collision
            closest_dist = self.sensor_range
            
            # March ray step by step (Basic Raymarching)
            # Note: Optimize with Line-Intersection formula later, basic loop is enough for now.
            for dist in range(0, self.sensor_range, 5): # Check every 5 pixels
                check_x = self.agent_pos[0] + dir_x * dist
                check_y = self.agent_pos[1] + dir_y * dist
                
                # Wall check
                point_rect = pygame.Rect(check_x, check_y, 1, 1)
                if point_rect.collidelist(self.walls) != -1:
                    closest_dist = dist
                    break
            
            # Normalize (1.0 = Very Close, 0.0 = Far/Invisible)
            # For Encoding: Close (0.1 distance) -> High Signal needed. 
            # Therefore we invert: 1.0 - (dist / max)
            # Result: 1.0 (Very Close), 0.0 (Far)
            normalized_val = 1.0 - (closest_dist / self.sensor_range)
            readings.append(max(0.0, normalized_val))
            
        self.sensors = np.array(readings)
        return self.sensors

    def step(self, left_motor, right_motor):
        """
        Physics Engine Step
        Args:
            left_motor (float): Speed between -1.0 and 1.0
            right_motor (float): Speed between -1.0 and 1.0
        Returns:
            sensors (np.array): Eye data
            reward (float): Earned reward
        """
        reward = 0.0
        
        # 1. MOVEMENT PHYSICS (Differential Drive)
        # Limit speeds
        v_l = np.clip(left_motor, -1.0, 1.0) * self.agent_speed
        v_r = np.clip(right_motor, -1.0, 1.0) * self.agent_speed
        
        # Linear and Angular Velocity
        linear_vel = (v_l + v_r) / 2
        angular_vel = (v_r - v_l) / 10.0 # 10 = Wheel base constant
        
        self.agent_angle += angular_vel
        self.agent_pos[0] += math.cos(self.agent_angle) * linear_vel
        self.agent_pos[1] += math.sin(self.agent_angle) * linear_vel
        
        # 2. COLLISION CHECK (WALL)
        agent_rect = pygame.Rect(self.agent_pos[0]-15, self.agent_pos[1]-15, 30, 30)
        if agent_rect.collidelist(self.walls) != -1:
            # Simple response: Bounce back
            self.agent_pos[0] -= math.cos(self.agent_angle) * linear_vel * 1.5
            self.agent_pos[1] -= math.sin(self.agent_angle) * linear_vel * 1.5
            # Penalty can be added but kept neutral for now
        
        # 3. FOOD CHECK
        eat_index = agent_rect.collidelist(self.foods)
        if eat_index != -1:
            reward = 1.0 # DOPAMINE!
            self.foods.pop(eat_index)
            self.spawn_food() # Place new food
            
        # 4. UPDATE SENSORS
        sensors = self.raycast()
        
        return sensors, reward

    def render(self):
        """Draw to screen"""
        self.screen.fill(BLACK)
        
        # Walls
        for wall in self.walls:
            pygame.draw.rect(self.screen, RED, wall)
            
        # Foods
        for food in self.foods:
            pygame.draw.rect(self.screen, GREEN, food)
            
        # Agent (Body)
        pygame.draw.circle(self.screen, BLUE, (int(self.agent_pos[0]), int(self.agent_pos[1])), self.agent_radius)
        
        # Agent (Direction Line)
        end_x = self.agent_pos[0] + math.cos(self.agent_angle) * 20
        end_y = self.agent_pos[1] + math.sin(self.agent_angle) * 20
        pygame.draw.line(self.screen, WHITE, self.agent_pos, (end_x, end_y), 3)
        
        # Sensor Rays (Visualization)
        for i, angle in enumerate(self.sensor_angles):
            ray_angle = self.agent_angle + angle
            
            # Color brightness based on sensor value (Bright yellow if close, dim if far)
            intensity = int(self.sensors[i] * 255)
            ray_color = (intensity, intensity, 0)
            
            # Calculate ray length (Inverse operation)
            dist = (1.0 - self.sensors[i]) * self.sensor_range
            
            end_ray_x = self.agent_pos[0] + math.cos(ray_angle) * dist
            end_ray_y = self.agent_pos[1] + math.sin(ray_angle) * dist
            
            pygame.draw.line(self.screen, ray_color, self.agent_pos, (end_ray_x, end_ray_y), 1)

        pygame.display.flip()
        self.clock.tick(60) # 60 FPS

    def get_state(self):
        return self.sensors