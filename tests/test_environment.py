import pygame
from src.environment import Environment

def manual_drive():
    env = Environment()
    running = True
    
    print("🎮 Manual Control Mode")
    print("Drive using Arrow Keys. Sensor data will be printed to console.")

    while running:
        # Check Input Keys
        keys = pygame.key.get_pressed()
        left_motor = 0
        right_motor = 0
        
        if keys[pygame.K_UP]:
            left_motor = 1.0
            right_motor = 1.0
        elif keys[pygame.K_DOWN]:
            left_motor = -0.5
            right_motor = -0.5
        elif keys[pygame.K_LEFT]:
            left_motor = -0.5
            right_motor = 0.5
        elif keys[pygame.K_RIGHT]:
            left_motor = 0.5
            right_motor = -0.5
            
        # Handle Window Close (X button)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Execute Physics Step
        sensors, reward = env.step(left_motor, right_motor)
        
        # Render to Screen
        env.render()
        
        # Print Info
        if reward > 0:
            print("🍔 FOOD EATEN! Dopamine +++")
        
        # Print sensor data only when close to a wall (to keep console clean)
        if max(sensors) > 0.1:
            print(f"Sensors: {sensors.round(2)}")

    pygame.quit()

if __name__ == "__main__":
    manual_drive()