# -*- coding: utf-8 -*-
from ..minigrid import *
from ..register import register


class SimpleFourRoomsEnv(SimpleMiniGridEnv):
    def __init__(self, grid_size):
        assert grid_size >= 5
        super().__init__(grid_size=grid_size)

        self.max_steps = 8 * grid_size

    def reset(self):
        # Step count since episode start
        self.step_count = 0

        # Create grid
        self.create_grid(self.width, self.height)
        self.create_outer_wall()
        self.create_room_walls()
        self.create_room_doors()

        # Select a random initial state and goal
        self.reset_state_goal()

        # Add goal
        self.goals = list()
        self.add_goal(self.goal_pos)

        return self.state, self.goal_pos

    def create_room_walls(self):
        x = self.grid.width // 2
        y = self.grid.height // 2
        self.grid.vert_wall(x, 0)
        self.grid.horz_wall(0, y)

    def create_room_doors(self):
        x_wall = self.grid.width // 2
        y_wall = self.grid.height // 2

        x_door = self.grid.width // 4
        y_door = self.grid.height // 4

        # Remove walls at door positions
        self.grid.set(x_wall, y_door, None)
        self.grid.set(x_wall, self.grid.height - 1 - y_door, None)
        self.grid.set(x_door, y_wall, None)
        self.grid.set(self.grid.width - 1 - x_door, y_wall, None)


class SimpleFourRoomsEnv5x5(SimpleFourRoomsEnv):
    def __init__(self):
        super().__init__(grid_size=5)


class SimpleFourRoomsEnv10x10(SimpleFourRoomsEnv):
    def __init__(self):
        super().__init__(grid_size=9)


class SimpleFourRoomsEnv15x15(SimpleFourRoomsEnv):
    def __init__(self):
        super().__init__(grid_size=15)


class SimpleFourRoomsEnv20x20(SimpleFourRoomsEnv):
    def __init__(self):
        super().__init__(grid_size=19)


class SimpleFourRoomsEnv25x25(SimpleFourRoomsEnv):
    def __init__(self):
        super().__init__(grid_size=25)


register(
    _id='Simple-MiniGrid-FourRooms-5x5-v0',
    entry_point='gym_simple_minigrid.envs:SimpleFourRoomsEnv5x5'
)

register(
    _id='Simple-MiniGrid-FourRooms-10x10-v0',
    entry_point='gym_simple_minigrid.envs:SimpleFourRoomsEnv10x10'
)

register(
    _id='Simple-MiniGrid-FourRooms-15x15-v0',
    entry_point='gym_simple_minigrid.envs:SimpleFourRoomsEnv15x15'
)

register(
    _id='Simple-MiniGrid-FourRooms-20x20-v0',
    entry_point='gym_simple_minigrid.envs:SimpleFourRoomsEnv20x20'
)

register(
    _id='Simple-MiniGrid-FourRooms-25x25-v0',
    entry_point='gym_simple_minigrid.envs:SimpleFourRoomsEnv25x25'
)
