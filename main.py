import numpy as np
import pygame

from src.environment.simulation import Environment
from src.encoding import Encoder
from src.decoding import Decoder
from src.network import Network

def main():
    
    # Configure the environment
    DT = 1.0 # Simulation time step
    NUM_SENSORS = 5 # Number of sensors (EYES)
    NUM_MOTORS = 2 # Left and Right Wheels
    NUM_HIDDEN = 50 # Number of hidden neurons (capacity of the brain)

    TOTAL_NEURONS = NUM_SENSORS + NUM_HIDDEN + NUM_MOTORS

    # SETUP

    print("System is starting up...")

    # World Setup
    env = Environment(width=800, height=600)

    # Network Setup
    brain = Network(
        num_neurons= TOTAL_NEURONS,
        num_inputs= NUM_SENSORS,
        num_outputs= NUM_MOTORS,
        dt= DT
    )

    # Making the brain more aggrasive at the beginning to make it move
    brain.base_threshold = 1.2

    # Nervous System Setup
    encoder = Encoder(
        min_val=0.0,
        max_val=1.0,
        max_freq=100.0,
        dt=DT
    )

    decoder = Decoder(
        num_outputs=NUM_MOTORS,
        tau=10.0,
        dt=DT
    )

    print("RHEO Agent is ready! Simulation starting...")
    print("Press M for manual control")


    # Main Simulation Loop

    running = True
    manual_mode = False
    current_reward = 0.0

    while running:
        # User input handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_m:
                    manual_mode = not manual_mode
                    decoder.reset()
                    print("Manual mode: ", "ON" if manual_mode else "Auto-Pilot (AI)")
        
        # Brain Loop:

        # observation
        sensor_data = env.get_state()

        # Encoding: Convert the data to current
        input_currents = encoder.step_current(sensor_data, gain=30.0)
        input_currents += np.random.uniform(0.0, 2.0, size=input_currents.shape)

        # Brain Processing: Run the network for one time step
        all_spikes = brain.step(external_inputs=input_currents, reward=current_reward)

        # Decoding: Convert the spikes to motor commands
        output_spikes = all_spikes[-NUM_MOTORS:] # Motor neurons are at the end of the list
        motor_commands = decoder.step(output_spikes)

        # Movement: Act.
        left_motor = 0.0
        right_motor = 0.0

        if manual_mode:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                left_motor = 1.0
                right_motor = 1.0
            if keys[pygame.K_DOWN]:
                left_motor = -0.5
                right_motor = -0.5
            if keys[pygame.K_LEFT]:
                left_motor = -0.5
                right_motor = 0.5
            if keys[pygame.K_RIGHT]:
                left_motor = 0.5
                right_motor = -0.5
        else:
            # Auto-Pilot (AI)
            left_motor = motor_commands[0] * 3
            right_motor = motor_commands[1] * 3

            # Bias: Even if nothing happens, agent should still move
            left_motor += 0.5
            right_motor += 0.5


        # Physics and results

        _, reward = env.step(left_motor, right_motor)

        current_reward = reward

        env.render()
        out_fire_count = output_spikes.sum()
        if out_fire_count > 0 or reward > 0:
            print(f"Sensor: {sensor_data.max():.2f} | Output Spikes: {int(out_fire_count)} | Motor: {motor_commands.round(2)}")

        if reward > 0:
            print(f"Reward is received! Dopamin is increasing: {reward}")
        
    pygame.quit()

if __name__ == "__main__":
    main()
