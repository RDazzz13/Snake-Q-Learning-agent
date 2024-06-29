import pygame
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

# Initialize Pygame
pygame.init()

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Game dimensions
BLOCK_SIZE = 20
GRID_WIDTH = 20
GRID_HEIGHT = 20
SCREEN_WIDTH = BLOCK_SIZE * GRID_WIDTH
SCREEN_HEIGHT = BLOCK_SIZE * GRID_HEIGHT

# Set up the display
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Snake AI")

clock = pygame.time.Clock()

class Snake:
    def __init__(self):
        self.body = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
        self.direction = random.choice(["UP", "DOWN", "LEFT", "RIGHT"])

    def move(self, action):
        # Update direction based on action
        if action == "UP" and self.direction != "DOWN":
            self.direction = "UP"
        elif action == "DOWN" and self.direction != "UP":
            self.direction = "DOWN"
        elif action == "LEFT" and self.direction != "RIGHT":
            self.direction = "LEFT"
        elif action == "RIGHT" and self.direction != "LEFT":
            self.direction = "RIGHT"

        # Move the snake
        head = self.body[0]
        if self.direction == "UP":
            new_head = (head[0], head[1] - 1)
        elif self.direction == "DOWN":
            new_head = (head[0], head[1] + 1)
        elif self.direction == "LEFT":
            new_head = (head[0] - 1, head[1])
        else:  # RIGHT
            new_head = (head[0] + 1, head[1])

        self.body.insert(0, new_head)

    def grow(self):
        # Add a new segment to the snake
        self.body.append(self.body[-1])

    def check_collision(self):
        # Check if the snake has collided with walls or itself
        head = self.body[0]
        return (head[0] < 0 or head[0] >= GRID_WIDTH or
                head[1] < 0 or head[1] >= GRID_HEIGHT or
                head in self.body[1:])

class Game:
    def __init__(self):
        self.snake = Snake()
        self.food = self.generate_food()
        self.score = 0

    def generate_food(self):
        # Generate food at a random position not occupied by the snake
        while True:
            food = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
            if food not in self.snake.body:
                return food

    def step(self, action):
        # Perform one step of the game
        self.snake.move(action)
        reward = 0
        game_over = False

        if self.snake.check_collision():
            game_over = True
            reward = -10
        elif self.snake.body[0] == self.food:
            self.snake.grow()
            self.food = self.generate_food()
            self.score += 1
            reward = 10
        else:
            self.snake.body.pop()

        return reward, game_over, self.score

    def get_state(self):
        # Get the current state of the game
        head = self.snake.body[0]
        point_l = (head[0] - 1, head[1])
        point_r = (head[0] + 1, head[1])
        point_u = (head[0], head[1] - 1)
        point_d = (head[0], head[1] + 1)

        dir_l = self.snake.direction == "LEFT"
        dir_r = self.snake.direction == "RIGHT"
        dir_u = self.snake.direction == "UP"
        dir_d = self.snake.direction == "DOWN"

        state = [
            # Danger straight
            (dir_u and self.is_collision(point_u)) or
            (dir_d and self.is_collision(point_d)) or
            (dir_l and self.is_collision(point_l)) or
            (dir_r and self.is_collision(point_r)),

            # Danger right
            (dir_u and self.is_collision(point_r)) or
            (dir_d and self.is_collision(point_l)) or
            (dir_u and self.is_collision(point_u)) or
            (dir_d and self.is_collision(point_d)),

            # Danger left
            (dir_u and self.is_collision(point_l)) or
            (dir_d and self.is_collision(point_r)) or
            (dir_r and self.is_collision(point_u)) or
            (dir_l and self.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            self.food[0] < self.snake.body[0][0],  # food left
            self.food[0] > self.snake.body[0][0],  # food right
            self.food[1] < self.snake.body[0][1],  # food up
            self.food[1] > self.snake.body[0][1]   # food down
        ]

        return np.array(state, dtype=int)

    def is_collision(self, point):
        # Check if a point is a collision
        return (point in self.snake.body or
                point[0] < 0 or point[0] >= GRID_WIDTH or
                point[1] < 0 or point[1] >= GRID_HEIGHT)

    def render(self):
        # Render the game state
        screen.fill(BLACK)
        for segment in self.snake.body:
            pygame.draw.rect(screen, GREEN, (segment[0]*BLOCK_SIZE, segment[1]*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(screen, RED, (self.food[0]*BLOCK_SIZE, self.food[1]*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
        pygame.display.flip()

class QLearningAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = {}
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99999
        self.alpha = 0.1
        self.gamma = 0.9

    def get_action(self, state):
        # Choose an action using epsilon-greedy policy
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            return np.argmax(self.get_q_values(state))

    def get_q_values(self, state):
        # Get Q-values for a given state
        state_str = str(state)
        if state_str not in self.q_table:
            self.q_table[state_str] = np.zeros(self.action_size)
        return self.q_table[state_str]

    def train(self, state, action, reward, next_state, done):
        # Update Q-values using Q-learning algorithm
        q_values = self.get_q_values(state)
        next_q_values = self.get_q_values(next_state)
        td_target = reward + self.gamma * np.max(next_q_values) * (not done)
        td_error = td_target - q_values[action]
        q_values[action] += self.alpha * td_error

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def save_training_state(agent, game_count, scores, avg_scores):
    # Save the current training state
    state = {
        'q_table': agent.q_table,
        'epsilon': agent.epsilon,
        'game_count': game_count,
        'scores': scores,
        'avg_scores': avg_scores
    }
    with open('training_state.pkl', 'wb') as f:
        pickle.dump(state, f)

def load_training_state():
    # Load the previous training state if it exists
    if os.path.exists('training_state.pkl'):
        with open('training_state.pkl', 'rb') as f:
            return pickle.load(f)
    return None

def train(n_games):
    # Load previous training state or initialize new agent
    prev_state = load_training_state()
    if prev_state:
        agent = QLearningAgent(11, 4)
        agent.q_table = prev_state['q_table']
        agent.epsilon = prev_state['epsilon']
        start_game = prev_state['game_count']
        scores = prev_state['scores']
        avg_scores = prev_state['avg_scores']
        print(f"Resuming training from game {start_game}")
    else:
        agent = QLearningAgent(11, 4)  # 11 state features, 4 actions
        start_game = 0
        scores = []
        avg_scores = []

    try:
        for i in range(start_game, start_game + n_games):
            game = Game()
            done = False
            score = 0

            while not done:
                state = game.get_state()
                action = agent.get_action(state)
                reward, done, score = game.step(["UP", "DOWN", "LEFT", "RIGHT"][action])
                next_state = game.get_state()
                agent.train(state, action, reward, next_state, done)

                game.render()
                pygame.time.delay(50)  # Adjust this value to control game speed

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt

            scores.append(score)
            avg_score = np.mean(scores[-100:])
            avg_scores.append(avg_score)
            
            print(f"Game {i}, Score: {score}, Avg Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.4f}")

            if i % 1000 == 0:
                save_training_state(agent, i, scores, avg_scores)

    except KeyboardInterrupt:
        print("Training interrupted. Saving current state...")
        save_training_state(agent, i, scores, avg_scores)

    pygame.quit()

    plt.plot(avg_scores)
    plt.title('Average Scores over Training')
    plt.xlabel('Game')
    plt.ylabel('Average Score (last 100 games)')
    plt.savefig('training_scores.png')
    plt.show()

def play_game(agent):
    # Play the game using the trained agent
    game = Game()
    done = False
    score = 0

    while not done:
        state = game.get_state()
        action = agent.get_action(state)
        _, done, score = game.step(["UP", "DOWN", "LEFT", "RIGHT"][action])

        game.render()
        pygame.time.delay(100)  # Slower speed for viewing

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

    print(f"Game Over. Score: {score}")
    pygame.quit()

if __name__ == "__main__":
    choice = input("Do you want to (t)rain or (p)lay? ")
    
    if choice.lower() == 't':
        n_games = int(input("Enter the number of games to train: "))
        train(n_games)
    elif choice.lower() == 'p':
        prev_state = load_training_state()
        if prev_state:
            agent = QLearningAgent(11, 4)
            agent.q_table = prev_state['q_table']
            agent.epsilon = 0  # Set epsilon to 0 for deterministic policy
            play_game(agent)
        else:
            print("No trained model found. Please train the model first.")
    else:
        print("Invalid choice. Please run the script again and choose 't' or 'p'.")