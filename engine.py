from collections import namedtuple
import numpy as np

from parameters import *

Point = namedtuple('Point', 'x y')

class Engine():
    def __init__(self, h, w):
        # These are some parameters / enums for the class
        # Using Point as a speed vector
        self.direction_map = {"Up":Point(0, -1), "Down":Point(0, 1), "Right":Point(1, 0), "Left":Point(-1,0), "UpLeft":Point(-1, -1), "UpRight":Point(1,-1), "DownLeft":Point(-1, 1), "DownRight":Point(1,1)}
    
        self.cur_direction = self.direction_map["Right"]
        
        self.width = w
        self.height = h

        self.board = np.zeros((h+2, w+2))
        # Make the edges BODY type (For collision dection?)
        for i in range(h+2):
            self.board[0][i] = Cell.BODY
            self.board[w+1][i] = Cell.BODY
        
        for i in range(w+2):
            self.board[i][0] = Cell.BODY
            self.board[i][h+1] = Cell.BODY

        self.board[2][2] = Cell.BODY

        self.head = Point(2,2)
        self.snake = []
        self.snake.append(self.head)
        self.snake_len = 3

        self.food_history = []
        self.food = Point(0,0)
        self.place_food()
        
        self.done = False
        self.step_count = 0

    def place_food(self):
        
        while True:
            new_food = Point(np.random.randint(0, self.width), np.random.randint(0, self.height))
            if new_food.x != self.food.x and new_food.y != self.food.y and self.board[new_food.x][new_food.y] ==0:
                break
        
        self.food = new_food
        self.board[self.food.x][self.food.y] = Cell.FOOD
        self.food_history.append(Point(self.food.x, self.food.y))


    # -- External Interfaces -- #
    def set_direction(self, direction):
        self.cur_direction = self.direction_map[direction]

    def check_collision(self, new_head):
        # I made this confusing, 'check walls' returns true if things are ok..
        return self.check_walls(new_head) and self.check_self(new_head)
    
    # returns true if a wall was hit
    def check_walls(self, head):
        if self.board[head.x][head.y] == Cell.BODY:
            return False
        
        return True

    def check_self(self, new_head):
        for point in self.snake:
            if point.x == new_head.x and point.y==new_head.y:
                return False
        
        return True

    def check_eat(self, head):
        if head.x == self.food.x and head.y == self.food.y:
            return True
        
        return False
    
    def step(self):
        reward = 0 # step reward defaults to 0

        # Get direction vector and update head
        new_head = Point(self.head.x + self.cur_direction.x, self.head.y + self.cur_direction.y)

        if self.step_count > 10000: # ending game early
            self.done = True
            reward = -100 
            print("Timed Out")

        # verify game
        if self.check_walls(new_head) == False:
            reward = -100
            self.done = True
        
        if self.check_self(new_head) == False:
            reward = -100
            self.done = True
        
        if self.check_eat(new_head):
            # increment score
            # place new food
            self.place_food()
            self.snake_len += 1
            reward = 50

        if not self.done:
            self.step_count = self.step_count + 1
            self.head = new_head
            self.board[self.head.x][self.head.y] = Cell.BODY
            self.snake.append(self.head)
            self.board[self.food.x][self.food.y] = Cell.FOOD

            if len(self.snake) > self.snake_len:
                tail = self.snake.pop(0)

                self.board[tail.x][tail.y] = Cell.EMPTY

        # finished, reward, state
        return self.done, reward, self.get_state()
  
    def get_state(self):
        # state to represent the type of object in the neighboring cells
        state = [
            self.board[self.head.x + self.direction_map["Left"].x][self.head.y + self.direction_map["Left"].y],
            self.board[self.head.x + self.direction_map["Up"].x][self.head.y + self.direction_map["Up"].y],
            self.board[self.head.x + self.direction_map["Down"].x][self.head.y + self.direction_map["Down"].y],
            self.board[self.head.x + self.direction_map["Right"].x][self.head.y + self.direction_map["Right"].y],

            self.food.x < self.head.x,
            self.food.y < self.head.y,
            self.food.x > self.head.x,
            self.food.y > self.head.y
        ]

        return tuple(state)

    def find_distance(self, head, direction, cell_type):
        count = 1

        x = self.direction_map[direction].x + head.x
        y = self.direction_map[direction].y + head.y

        while self.board[x][y] != cell_type:
            x += self.direction_map[direction].x
            y += self.direction_map[direction].y

            count += 1

        return count

    # This state will represent 8 directions from the snakes current head. A floating point value normalized from 0-1 on distance away.
    # let's also represent the distance to food along 8 directions
    def get_extended_state(self):
        state = [
            # self.find_distance(self.head, "Left", Cell.BODY),
            # self.find_distance(self.head, "Up", Cell.BODY),
            # self.find_distance(self.head, "Right", Cell.BODY),
            # self.find_distance(self.head, "Down", Cell.BODY),
            self.board[self.head.x + self.direction_map["Left"].x][self.head.y + self.direction_map["Left"].y],
            self.board[self.head.x + self.direction_map["Up"].x][self.head.y + self.direction_map["Up"].y],
            self.board[self.head.x + self.direction_map["Down"].x][self.head.y + self.direction_map["Down"].y],
            self.board[self.head.x + self.direction_map["Right"].x][self.head.y + self.direction_map["Right"].y],

            self.board[self.head.x + self.direction_map["UpLeft"].x][self.head.y + self.direction_map["UpLeft"].y],
            self.board[self.head.x + self.direction_map["UpRight"].x][self.head.y + self.direction_map["UpRight"].y],
            self.board[self.head.x + self.direction_map["DownLeft"].x][self.head.y + self.direction_map["DownLeft"].y],
            self.board[self.head.x + self.direction_map["DownRight"].x][self.head.y + self.direction_map["DownRight"].y],

            self.food.x < self.head.x,
            self.food.y < self.head.y,
            self.food.x > self.head.x,
            self.food.y > self.head.y
        ]

        return tuple(state)