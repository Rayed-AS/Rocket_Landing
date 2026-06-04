import pygame
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

#AI CODE STARTS HERE
# 1. Initialize Pygame and Font
pygame.init()
WIDTH, HEIGHT = 600, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Rocket Grid Visualizer")
font = pygame.font.SysFont("Arial", 16)

# Grid Settings
GRID_SIZE = 100  # Draw a line every 100 pixels
GRID_COLOR = (60, 60, 90)  # Faint blue/gray for the grid lines
TEXT_COLOR = (120, 120, 160)

#Setup Positions
rocket_x = WIDTH//2
rocket_y = 100
ground_y = HEIGHT - 60
#NOT AI
t=0
acceleration = 9.81
rocket_acceleration = -20
velocity = 40+acceleration*t
fuel = 100

def reset(seed = None):
    global screen, rocket_x, rocket_y, ground_y, velocity, GRID_SIZE, GRID_COLOR, TEXT_COLOR, font, acceleration, fuel
    pygame.init()
    WIDTH, HEIGHT = 600, 800
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Rocket Grid Visualizer")
    font = pygame.font.SysFont("Arial", 16)

    GRID_SIZE = 100  # Draw a line every 100 pixels
    GRID_COLOR = (60, 60, 90)  # Faint blue/gray for the grid lines
    TEXT_COLOR = (120, 120, 160)

    fuel = 100
    rocket_x = WIDTH // 2
    rocket_y = 100
    ground_y = HEIGHT - 60
    velocity = 40
    return np.array([rocket_y, velocity])

def step(action):
    global rocket_x, rocket_y, ground_y, velocity, fuel
    if action == 0 and fuel != 0:
        fuel -= 1
        velocity += rocket_acceleration + acceleration
        rocket_y += velocity
    if action == 1:
        velocity += acceleration
        rocket_y += velocity
    if (rocket_y > ground_y and velocity > 10) or rocket_y < 0:
        return np.array([rocket_y, velocity]), -50, True, False, {}
    elif rocket_y >= ground_y and velocity <= 10:
        return np.array([rocket_y, velocity]), 1000, True, False, {}
    else:
        return np.array([rocket_y, velocity]), -1, False, False, {}

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(2, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

state_size = 2
action_size = 2

gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.997
learning_rate = 0.0005
batch_size = 64
memory_size = 10000

memory = deque(maxlen=memory_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy_net = DQN(state_size, action_size).to(device)
target_net = DQN(state_size, action_size).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

def get_action(state, epsilon):
    if random.random() < epsilon:
        return random.choice(range(action_size))
    else:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = policy_net(state)
        return q_values.argmax().item()

def replay():
    if len(memory) < batch_size:
        return

    minibatch = random.sample(memory, batch_size)

    states, actions, rewards, next_states, dones = zip(*minibatch)
    states = np.array(states)
    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions).unsqueeze(1).to(device)
    rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
    next_states = np.array(next_states)
    next_states = torch.FloatTensor(next_states).to(device)
    dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

    current_q = policy_net(states).gather(1, actions)

    next_q = target_net(next_states).max(1)[0].detach().unsqueeze(1)
    target_q = rewards + (gamma * next_q * (1-dones))

    loss = loss_fn(current_q, target_q)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
#Training process
episodes = 3300
target_update_freq = 10
start_training = time.time()
for episode in range(episodes):
    reset_result = reset()
    state = reset_result
    total_reward = 0

    for t in range(500):
        action = get_action(state, epsilon)
        step_result = step(action)

        if len(step_result) == 5:
            next_state, reward, terminated, truncated, _ = step_result
            done = terminated or truncated
        else:
            next_state, reward, done, _ = step_result

        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        replay()
        if done:
            break
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    if episode % target_update_freq == 0:
        target_net.load_state_dict(policy_net.state_dict())

    if episode % 100 == 0:
        print(f"Episode: {episode}, Total Reward: {total_reward}")
clock = pygame.time.Clock()
done = False
state = reset()
step_count = 0
cumulative_reward = 0
#AI
while not done and step_count <= 500:
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        action = torch.argmax(policy_net(state)).item()
        t += 0.1
        next_state, reward, terminated, truncated, _ = step(action)
        cumulative_reward += reward
        done = terminated or truncated
        step_count += 1
        state = next_state
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    screen.fill((15, 15, 30))
    pygame.draw.rect(screen, (120, 120, 120), (0, ground_y, WIDTH, 60))

    # CRASH / EXPLOSION CHECK
    # If the bottom of the rocket reaches or passes the ground
    if rocket_y >= ground_y and velocity <= 10:
        pygame.draw.rect(screen, (240, 240, 240), (rocket_x - 15, rocket_y - 60, 30, 60))
        pygame.draw.polygon(screen, (255, 50, 50), [
            (rocket_x - 15, rocket_y - 60),
            (rocket_x + 15, rocket_y - 60),
            (rocket_x, rocket_y - 85)])
        time.sleep(1)
        win_text = font.render("ROCKET LANDED!", True, (255, 50, 50))
    elif rocket_y >= ground_y:
        # 1. Draw a multi-layered explosion at the crash site
        pygame.draw.circle(screen, (255, 150, 0), (int(rocket_x), int(ground_y)), 50)  # Outer orange blast
        pygame.draw.circle(screen, (255, 220, 0), (int(rocket_x), int(ground_y)), 30)  # Inner yellow core
        pygame.draw.circle(screen, (255, 255, 255), (int(rocket_x), int(ground_y)), 15)  # White center

        # Render "CRASH!" text onto the screen
        crash_text = font.render("ROCKET CRASHED!", True, (255, 50, 50))
        screen.blit(crash_text, (WIDTH // 2 - 80, HEIGHT // 2))

        # Update the display one last time so the explosion actually renders
        pygame.display.flip()

        # 2. Freeze the game for 1.5 seconds so you can see the destruction
        time.sleep(1.5)

        # 3. Reset the simulation states back to the top of the screen
        # This calls your reset function and unpacks the new starting positions
        rocket_y, velocity = reset()
        done = False
    else:
        # If the rocket hasn't crashed, draw it normally
        # Draw the Rocket Body
        pygame.draw.rect(screen, (240, 240, 240), (rocket_x - 15, rocket_y - 60, 30, 60))

        # Draw the Rocket Nose Cone
        pygame.draw.polygon(screen, (255, 50, 50), [
            (rocket_x - 15, rocket_y - 60),
            (rocket_x + 15, rocket_y - 60),
            (rocket_x, rocket_y - 85)
        ])
    # ==========================================
    # REFRESH SCREEN
    # ==========================================
    # Remove the old time.sleep(1) from here so the game runs smoothly while flying
    pygame.display.flip()
    clock.tick(30)
#Anything below this section uses the specific variables created using AI but the specific implementation and integration of these variables within the project is a whole is my work

print(cumulative_reward)