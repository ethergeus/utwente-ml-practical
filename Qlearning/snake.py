#!/usr/bin/env python3

from enum import Enum
from random import randint, choice
from sys import maxsize

from qlearning import QTable, State

# Enum for the spaces on the board
class Space(Enum):
    OUT_OF_BOUNDS = -1
    EMPTY = 0
    SNAKE = 1
    FOOD = 2

# Enum for the directions
class Direction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

# Class for the board
class Board:
    # Rewards
    BASE_REWARD = 1
    OUT_OF_BOUNDS_REWARD = -100
    SELF_COLLISION_REWARD = -100
    FOOD_REWARD = 100

    # Training
    ALPHA = 0.1 # Learning rate
    GAMMA = 0.9 # Discount factor
    EPSILON = 0.1 # Exploration rate
    RADIUS = 1 # Radius of the environment

    def __init__(self, size_x: int = 32, size_y: int = 16, snake: tuple = None):
        self.size_x = size_x
        self.size_y = size_y

        # Create the board
        self.board = [[Space.EMPTY for x in range(size_x)] for y in range(size_y)]

        # Place the snake
        if snake is None:
            snake = (randint(0, size_x - 1), randint(0, size_y - 1))
        self.snake = [snake]

        # List of empty spaces
        self.empty = [(x, y) for x in range(size_x) for y in range(size_y) if (x, y) not in self.snake]

        # Place the food
        self.place_food()
    
    def get_space(self, x: int, y: int) -> Space:
        if x < 0 or x >= self.size_x or y < 0 or y >= self.size_y:
            return Space.OUT_OF_BOUNDS # Out of bounds
        return self.board[y][x]
    
    def set_space(self, x: int, y: int, space: Space):
        self.board[y][x] = space
    
    def place_food(self):
        # Place the food
        self.food = choice(self.empty)
        self.set_space(self.food[0], self.food[1], Space.FOOD)
        self.empty.remove(self.food)
    
    def move(self, direction: Direction):
        reward = self.BASE_REWARD

        # Move the snake
        head = self.snake[0]
        if direction == Direction.UP:
            new_head = (head[0], head[1] - 1)
        elif direction == Direction.DOWN:
            new_head = (head[0], head[1] + 1)
        elif direction == Direction.LEFT:
            new_head = (head[0] - 1, head[1])
        elif direction == Direction.RIGHT:
            new_head = (head[0] + 1, head[1])
        
        # Check if the snake is dead
        if self.get_space(new_head[0], new_head[1]) == Space.OUT_OF_BOUNDS:
            return (False, self.OUT_OF_BOUNDS_REWARD)
        if new_head in self.snake:
            return (False, self.SELF_COLLISION_REWARD)
        
        # Move the snake
        self.snake.insert(0, new_head)
        self.set_space(head[0], head[1], Space.SNAKE)

        # Check if the snake ate the food
        if new_head == self.food:
            reward = Board.FOOD_REWARD
            self.place_food()
        else:
            tail = self.snake.pop()
            self.set_space(tail[0], tail[1], Space.EMPTY)
            self.empty.append(tail)
        
        return (True, reward)
    
    def get_state(self, radius: int = RADIUS):
        head = self.snake[0]
        food = self.food
        environment = tuple(tuple(self.get_space(x, y) for x in range(head[0] - radius, head[0] + 1 + radius)) for y in range(head[1] - radius, head[1] + 1 + radius))
        
        # Direction to the food
        food_vector = (food[0] - head[0], food[1] - head[1])

        # Normalize the vector to be between -1 and 1
        norm_food_vector = (0 if food_vector[0] == 0 else food_vector[0] / abs(food_vector[0]), 0 if food_vector[1] == 0 else food_vector[1] / abs(food_vector[1]))

        return State(environment + norm_food_vector)
    
    def __str__(self) -> str:
        # Print the board
        string = ''
        for y in range(self.size_y):
            for x in range(self.size_x):
                if (x, y) in self.snake:
                    string += '🐍'
                elif (x, y) == self.food:
                    string += '🍕'
                else:
                    string += '⬜'
            string += '\n'
        return string

# Main method
if __name__ == '__main__':
    q_table = QTable(actions=[d.value for d in Direction], alpha=Board.ALPHA, gamma=Board.GAMMA, epsilon=Board.EPSILON)
    highscore = 0
    try:
        for i in range(maxsize):
            # Reset the board
            board = Board()
            state = board.get_state()
            q_table.init_q(state)

            alive = True
            while alive:
                # Get the best action
                action = q_table.epsilon_greedy(state)

                # Move the snake, evaluate the reward and get the new state
                alive, reward = board.move(Direction(action))
                new_state = board.get_state()
                q_table.init_q(new_state)

                # Update the Q table
                q_table.update_q(state, action, reward, new_state)
                state = new_state
            
            highscore = max(highscore, len(board.snake))
            
            if i % 1000 == 0:
                print(f'Generation: {i}, Highscore: {highscore}, Q-Table entries: {len(q_table.q_table)}')
        
    except KeyboardInterrupt:
        print(f'Generation: {i}, Highscore: {highscore}')