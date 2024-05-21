from collections import namedtuple
import numpy as np

from parameters import *

Point = namedtuple('Point', 'x y')

class Engine():
    def __init__(self, h, w):
        # These are some parameters / enums for the class
        # Using Point as a speed vector
        self.direction_map = {"Up":Point(0, -1), "Down":Point(0, 1), "Right":Point(1, 0), "Left":Point(-1,0)}
    
        self.cur_direction = self.direction_map["Right"]
        
        self.width = w
        self.height = h

        self.board = np.zeros((h, w))
        self.board[0][0] = Cell.BODY

        self.head = Point(0,0)
        self.snake = []
        self.snake.append(self.head)
        self.snake_len = 3

        self.food = Point(0,0)
        self.place_food()

    def place_food(self):
        
        while True:
            new_food = Point(np.random.randint(0, self.width), np.random.randint(0, self.height))
            if new_food.x != self.food.x and new_food.y != self.food.y and self.board[new_food.x][new_food.y] ==0:
                break
        
        self.food = new_food
        self.board[self.food.x][self.food.y] = Cell.FOOD


    # -- External Interfaces -- #
    def set_direction(self, direction):
        self.cur_direction = self.direction_map[direction]

    # returns true if a wall was hit
    def check_walls(self, head):
        if head.x >= self.width:
            return False
        if head.x < 0:
            return False
        if head.y >= self.height:
            return False
        if head.y < 0:
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
        
        # Get direction vector and update head
        new_head = Point(self.head.x + self.cur_direction.x, self.head.y + self.cur_direction.y)

        # verify game
        if self.check_walls(new_head) == False:
            return False
        
        if self.check_self(new_head) == False:
            return False
        
        if self.check_eat(new_head):
            # increment score
            # place new food
            self.place_food()
            self.snake_len += 1

        self.head = new_head
        self.board[self.head.x][self.head.y] = Cell.BODY
        self.snake.append(self.head)
        self.board[self.food.x][self.food.y] = Cell.FOOD

        if len(self.snake) > self.snake_len:
            tail = self.snake.pop(0)

            self.board[tail.x][tail.y] = Cell.EMPTY

        return True

