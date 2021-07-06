# -*- coding: utf-8 -*-
from ..minigrid import *
from ..register import register


class SimpleEmptyEnv(SimpleMiniGridEnv):
    def __init__(self, grid_size):
        super().__init__(grid_size=grid_size)

    def reset(self):
        # Step count since episode start
        self.step_count = 0

        # Create grid
        self.create_grid(self.width, self.height)
        self.create_outer_wall()

        # Select a random initial state and goal
        self.reset_state_goal()

        # Add goal
        self.goals = list()
        self.add_goal(self.goal_pos)

        return self.state, self.goal_pos


class SimpleEmptyEnv5x5(SimpleEmptyEnv):
    def __init__(self):
        super().__init__(grid_size=5)


class SimpleEmptyEnv10x10(SimpleEmptyEnv):
    def __init__(self):
        super().__init__(grid_size=10)


class SimpleEmptyEnv15x15(SimpleEmptyEnv):
    def __init__(self):
        super().__init__(grid_size=15)


class SimpleEmptyEnv20x20(SimpleEmptyEnv):
    def __init__(self):
        super().__init__(grid_size=20)


class SimpleEmptyEnv25x25(SimpleEmptyEnv):
    def __init__(self):
        super().__init__(grid_size=25)


register(
    _id='Simple-MiniGrid-Empty-5x5-v0',
    entry_point='gym_simple_minigrid.envs:SimpleEmptyEnv5x5'
)

register(
    _id='Simple-MiniGrid-Empty-10x10-v0',
    entry_point='gym_simple_minigrid.envs:SimpleEmptyEnv10x10'
)

register(
    _id='Simple-MiniGrid-Empty-15x15-v0',
    entry_point='gym_simple_minigrid.envs:SimpleEmptyEnv15x15'
)

register(
    _id='Simple-MiniGrid-Empty-20x20-v0',
    entry_point='gym_simple_minigrid.envs:SimpleEmptyEnv20x20'
)

register(
    _id='Simple-MiniGrid-Empty-25x25-v0',
    entry_point='gym_simple_minigrid.envs:SimpleEmptyEnv25x25'
)
