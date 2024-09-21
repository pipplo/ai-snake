from enum import IntEnum

# Game Configs
ROW_COUNT = 24

class Cell(IntEnum):
    EMPTY = 0
    FOOD = 1
    BODY = 2

# Colors
BG_COLOR = 0xededed
FOOD_COLOR = 0xd30000
BODY_COLOR = 0x444976
EMPTY_COLOR = 0xd3d3d3