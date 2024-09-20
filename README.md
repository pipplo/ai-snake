# AI Snake

This is an experiment with using AI/ML in a practical use case. Eventually using Deep Q Learning to play a simple game of snake.

# Instructions

Install the necessary python requirements (requirements.txt) and use one of the following command lines

## Play the game

There is a console version of the game which uses wasd controls that can be launched using

```
python game_curses.py
```

There is a GUI version of the game which uses direction keys

```
python game_ui.py
```

## Traditional Q-Learning
Launch an agent to do training using traditional Q-Learning. This will run 100k training episodes to fill in the Q table and then launch the gui to play a few games using the trained q-table so you can visualize how well the system learns.

```
python simple_agent.py
```

## Deep Q-Learning
This path uses a pytorch model to learn the Q-Table using Deep QLearning. 

### Training
Run deep_q_agent.py to run training. Paramaters are configurable in the training function (number of games, etc). The model is saved to disk periodically

```
python deep_q_agent.py
```

### Playing
You can watch the model play the game based on the last saved version using the deep_q_play script

```
python deep_q_play.py
```