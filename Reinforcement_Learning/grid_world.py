# Implementation of a simple grid world environment for testing RL algorithms
import numpy as np
import matplotlib.pyplot as plt

# Create the gridworld object
class Gridworld:
    def __init__(self, rows, cols, start_position):
        if rows < 2 or cols < 2:
            ValueError("The provided values are too small, please select at least a 2x2 grid")
        self.rows = rows
        self.cols = cols
        self.r = start_position[0]
        self.c = start_position[1]

    # Sets the rewards and actions for each position
    def set_(self, rewards, actions):
        # rewards should be a dict of: (i, j): r (row, col): reward
        # actions should be a dict of: (i, j): A (row, col): list of possible actions
        self.rewards = rewards
        self.actions = actions

    # Manually override state
    def set_state(self, state):
        self.r = state[0]
        self.c = state[1]

    # Prints the grid
    def display(self):
        print("The state grid:\n", self.grid)
        print("The rewards grid:\n", self.rewards)

    # Returns the current state
    def current_state(self):
        return (self.r, self.c)

    # Checks if the current state is terminal
    def is_terminal(self, s):
        return s not in self.actions

    # Returns reward
    def move(self, action):
        # Actions are U, D, L, R 
        if action == 'L':
            self.c -= 1
        elif action == 'R':
            self.c += 1
        elif action == 'U':
            self.r -= 1
        elif action == :
            self.r += 1
        
        return self.rewards.get((self.r, self.c), 0)

    def undo_move(self, action):
        # These are the opposite of what U/D/L/R should normally do
        if action == 'U':
            self.i += 1
        elif action == 'D':
            self.i -= 1
        elif action == 'R':
            self.j -= 1
        elif action == 'L':
            self.j += 1
        # Raise an exception if we arrive somewhere we shouldn't be should never happen
        assert(self.current_state() in self.all_states())

    # Check if we are in a terminal state
    def game_over(self, state):
        return (self.r, self.c) not in self.actions 
    
    # Returns all states
    def all_states(self):
        return set(self.actions.keys() | set(self.rewards.keys()))

def standard_grid():
  # define a grid that describes the reward for arriving at each state
  # and possible actions at each state
  # the grid looks like this
  # x means you can't go there
  # s means start position
  # number means reward at that state
  # .  .  .  1
  # .  x  . -1
  # s  .  .  .
    g = Grid(3, 4, (2, 0))
    rewards = {(0, 3): 1, (1, 3): -1}
    actions = {
    (0, 0): ('D', 'R'),
    (0, 1): ('L', 'R'),
    (0, 2): ('L', 'D', 'R'),
    (1, 0): ('U', 'D'),
    (1, 2): ('U', 'D', 'R'),
    (2, 0): ('U', 'R'),
    (2, 1): ('L', 'R'),
    (2, 2): ('L', 'R', 'U'),
    (2, 3): ('L', 'U'),
  }
  g.set(rewards, actions)
  return g

def negative_grid(step_cost=-0.1):
  # in this game we want to try to minimize the number of moves
  # so we will penalize every move
  g = standard_grid()
  g.rewards.update({
    (0, 0): step_cost,
    (0, 1): step_cost,
    (0, 2): step_cost,
    (1, 0): step_cost,
    (1, 2): step_cost,
    (2, 0): step_cost,
    (2, 1): step_cost,
    (2, 2): step_cost,
    (2, 3): step_cost,
  })
  return g

if __name__ == "__main__":
    g = Gridworld(3,3, (2,0))
    g.display()
    moves = ['W', 'A', 'S', 'D']
    i = 0
    while not g.game_over():
        action = np.random.randint(0, 4)
        action = moves[action]
        print("Our action:", action)
        g.move(action)
        print("Current state:", g.current_state())
        if i > 100:
            break
