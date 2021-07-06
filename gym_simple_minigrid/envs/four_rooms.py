# -*- coding: utf-8 -*-
from ..minigrid import *
from ..register import register

from ..controller.optimal import get_n_steps_fr, optimal_action_fr


class SimpleFourRoomsEnv(SimpleMiniGridEnv):
    def __init__(self, grid_size):
        assert grid_size >= 5
        self.geo_data = None
        super().__init__(grid_size=grid_size)

        # TODO maxsteps
        # TODO put geo_data in init, not in reset

    def reset(self):
        # Step count since episode start
        self.step_count = 0

        # Create grid
        self.geo_data = dict()
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
        self.geo_data['x_wall'] = (x - 1)
        self.geo_data['y_wall'] = (y - 1)

    def create_room_doors(self):
        x_wall = self.grid.width // 2
        y_wall = self.grid.height // 2

        x_door = self.grid.width // 4
        y_door = self.grid.height // 4

        self.grid.set(x_wall, y_door, None)
        self.grid.set(x_wall, self.grid.height - 1 - y_door, None)
        self.grid.set(x_door, y_wall, None)
        self.grid.set(self.grid.width - 1 - x_door, y_wall, None)
        self.geo_data['l_door'] = (x_door - 1)
        self.geo_data['r_door'] = (self.width - x_door)
        self.geo_data['t_door'] = (y_door - 1)
        self.geo_data['b_door'] = (self.height - y_door)

    def get_n_steps(self, state, goal, count_turns=False):
        return get_n_steps_fr(state[:2], goal, self.geo_data)

    def optimal_action(self, state, goal):
        return optimal_action_fr(state, goal, self.geo_data)


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
    entry_point='gym_minigrid_simple.envs:SimpleFourRoomsEnv5x5'
)

register(
    _id='Simple-MiniGrid-FourRooms-10x10-v0',
    entry_point='gym_minigrid_simple.envs:SimpleFourRoomsEnv10x10'
)

register(
    _id='Simple-MiniGrid-FourRooms-15x15-v0',
    entry_point='gym_minigrid_simple.envs:SimpleFourRoomsEnv15x15'
)

register(
    _id='Simple-MiniGrid-FourRooms-20x20-v0',
    entry_point='gym_minigrid_simple.envs:SimpleFourRoomsEnv20x20'
)

register(
    _id='Simple-MiniGrid-FourRooms-25x25-v0',
    entry_point='gym_minigrid_simple.envs:SimpleFourRoomsEnv25x25'
)
