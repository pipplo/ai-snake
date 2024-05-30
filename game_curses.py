import curses

from engine import Engine
from parameters import *

class GameCurses():
    def __init__(self):
        self.screen = curses.initscr()
        self.screen.nodelay(True)

        # lines, columns, start line, start column
        self.my_window = curses.newwin(16, 25, 0, 0)
        self.my_window.nodelay(1)

    def draw_board(self, board):
        self.screen.clear()
        for y, row in enumerate(game_engine.board):
            for x, value in enumerate(row):
                self.my_window.addstr(x, y, str(int(value)))
        
        self.my_window.refresh()

    def draw_score(self, score):
        self.my_window.addstr(0, 13, 'Score: ' +str(int(score)))
        self.my_window.refresh()

if __name__ == "__main__":
    game_engine = Engine(ROW_COUNT, ROW_COUNT)
    game_ui = GameCurses()

    while True:
        key = game_ui.my_window.getch()
        if key == ord('s'):
            game_engine.set_direction('Down')
        if key == ord('w'):
            game_engine.set_direction('Up')
        if key == ord('a'):
            game_engine.set_direction('Left')
        if key == ord('d'):
            game_engine.set_direction('Right')

        curses.napms(100)

        done, _, _ =  game_engine.step()
        
        if done == True:
            break

        game_ui.draw_board(game_engine.board)
        game_ui.draw_score(game_engine.snake_len)
        
        

    curses.endwin()
