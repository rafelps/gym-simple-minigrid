import gym
from enum import IntEnum
from gym import spaces
from gym.utils import seeding
from .rendering import *

# Map of color names to RGB values
COLORS = {
    'black': np.array((0, 0, 0)),
    'red': np.array((255, 0, 0)),
    'green': np.array((0, 255, 0)),
    'blue': np.array((0, 0, 255)),
    'purple': np.array((112, 39, 195)),
    'yellow': np.array((255, 255, 0)),
    'dd_grey': np.array((50, 50, 50)),
    'd_grey': np.array((100, 100, 100)),
    'l_grey': np.array((200, 200, 200)),

    # Red gradient
    # 'grad_0':  np.array((255, 0, 0)),
    'grad_0': np.array((0, 0, 0)),
    'grad_1': np.array((255, 55, 0)),
    'grad_2': np.array((255, 99, 0)),
    'grad_3': np.array((254, 145, 28)),
    'grad_4': np.array((252, 195, 86)),
    'grad_5': np.array((255, 237, 150)),
}

# Used to map colors to integers
COLOR_TO_IDX = {k: i for i, k in enumerate(COLORS)}

# Map of object type to integers
OBJECT_TO_IDX = {
    'wall': 0,
    'goal': 1,
}

# Map of agent direction indices to vectors
DIRS = [
    # Right (positive X)
    (1, 0),
    # Down (positive Y)
    (0, 1),
    # Left (negative X)
    (-1, 0),
    # Up (negative Y)
    (0, -1),
]
DIR_TO_VEC = {i: np.array(d) for i, d in enumerate(DIRS)}
# VEC_TO_DIR = {d: i for i, d in enumerate(DIRS)}


class WorldObj:
    """
    Base class for grid world objects
    """

    def __init__(self, _type, color):
        assert _type in OBJECT_TO_IDX, _type
        assert color in COLOR_TO_IDX, color
        self.type = _type
        self.color = color

    def render(self, r):
        """Draw this object with the given renderer"""
        raise NotImplementedError

    def encode(self):
        """Encode the a description of this object as a 2-tuple of integers"""
        return OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color]


class Goal(WorldObj):
    def __init__(self, level=0):
        color = f"grad_{min(level, 5)}"
        super().__init__('goal', color)

    def render(self, img):
        fill_coords(img, point_in_rect(0.1, 0.9, 0.1, 0.9), COLORS[self.color])


class Wall(WorldObj):
    # def __init__(self, color='l_grey'):
    def __init__(self, color='dd_grey'):
        super().__init__('wall', color)

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


class Grid:
    """
    Represent a grid and operations on it
    """

    # Static cache of pre-renderer tiles
    tile_cache = {}

    def __init__(self, width, height):
        assert width >= 3
        assert height >= 3

        self.width = width
        self.height = height

        self.grid = [None] * width * height

    def set(self, i, j, v):
        assert 0 <= i < self.width
        assert 0 <= j < self.height
        old_obj = self.grid[j * self.width + i]
        self.grid[j * self.width + i] = v
        return old_obj

    def get(self, i, j):
        assert 0 <= i < self.width
        assert 0 <= j < self.height
        return self.grid[j * self.width + i]

    def horz_wall(self, x, y, length=None, obj_type=Wall):
        if length is None:
            length = self.width - x
        for i in range(0, length):
            self.set(x + i, y, obj_type())

    def vert_wall(self, x, y, length=None, obj_type=Wall):
        if length is None:
            length = self.height - y
        for j in range(0, length):
            self.set(x, y + j, obj_type())

    def wall_rect(self, x=0, y=0, w=None, h=None):
        if w is None:
            w = self.width
        if h is None:
            h = self.height

        self.horz_wall(x, y, w)
        self.horz_wall(x, y + h - 1, w)
        self.vert_wall(x, y, h)
        self.vert_wall(x + w - 1, y, h)

    @classmethod
    def render_tile(cls, obj, agent_dir=None, tile_size=32, subdivs=3, thick=0.031):
        """
        Render a tile and cache the result
        """

        # Hash map lookup key for the cache
        key = (agent_dir,)
        key = obj.encode() + key if obj else key

        if key in cls.tile_cache:
            return cls.tile_cache[key]

        img = np.zeros(shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8)

        # Draw background
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS['d_grey'])

        # Draw the grid lines
        fill_coords(img, point_in_rect(0, thick, 0, 1), COLORS['l_grey'])
        fill_coords(img, point_in_rect(0, 1, 0, thick), COLORS['l_grey'])
        fill_coords(img, point_in_rect(1 - thick, 1, 0, 1), COLORS['l_grey'])
        fill_coords(img, point_in_rect(0, 1, 1 - thick, 1), COLORS['l_grey'])

        if obj is not None:
            obj.render(img)

        # Overlay the agent on top
        if agent_dir is not None:
            tri_fn = point_in_triangle((0.12, 0.19), (0.87, 0.50), (0.12, 0.81))

            # Rotate the agent based on its direction
            tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * agent_dir)
            # Agent color
            fill_coords(img, tri_fn, COLORS['green'])

        # Downsample the image to perform supersampling/anti-aliasing
        img = downsample(img, subdivs)

        # Cache the rendered tile
        cls.tile_cache[key] = img

        return img

    def render(self, tile_size, agent_pos=None, agent_dir=None):
        """
        Render this grid at a given scale
        :param tile_size: tile size in pixels
        :param agent_pos:
        :param agent_dir:
        """

        # Compute the total grid size
        width_px = self.width * tile_size
        height_px = self.height * tile_size

        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        # Render the grid
        for j in range(0, self.height):
            for i in range(0, self.width):
                cell = self.get(i, j)

                agent_here = np.array_equal(agent_pos, (i, j))
                tile_img = Grid.render_tile(cell, agent_dir=agent_dir if agent_here else None, tile_size=tile_size)

                ymin = j * tile_size
                ymax = (j + 1) * tile_size
                xmin = i * tile_size
                xmax = (i + 1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img

        return img


class SimpleMiniGridEnv(gym.Env):
    """
    2D grid world game environment
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 10
    }

    # Enumeration of possible actions
    class Actions(IntEnum):
        # Turn left, turn right, move forward
        left = 0
        right = 1
        forward = 2

    def __init__(self, grid_size=None, width=None, height=None, max_steps=None, seed=9):
        # Env name
        self.name = self.__class__.__name__

        # Can't set both grid_size and width/height
        if grid_size:
            assert width is None and height is None
            width = grid_size
            height = grid_size

        # Default max_steps
        if max_steps is None:
            max_steps = 4 * (width + height)

        # Action enumeration for this environment
        self.actions = SimpleMiniGridEnv.Actions

        # Actions are discrete integer values
        self.action_space = spaces.Discrete(len(self.actions))

        # Observations are dictionaries containing an
        # encoding of the grid and a textual 'mission' string
        self.observation_space = spaces.Box(
            low=np.array((0, 0, 0)),
            high=np.array((width - 1, height - 1, 2)),
            dtype=np.int
        )

        # Range of possible rewards
        self.reward_range = (-1, 0)

        # Window to use for human rendering mode
        self.window = None

        # Environment configuration
        self.width = width
        self.height = height
        self.max_steps = max_steps

        # Initialize the RNG
        self.np_random = None
        self.seed(seed=seed)

        # Initialize the environment
        self.agent_pos = self.agent_dir = self.goal_pos = self.step_count = self.grid = self.goals = None
        self.reset()

    def reset(self):
        raise NotImplementedError("Reset should be implemented by each environment type")

    def add_goal(self, goal_pos, goal_level=None):
        if goal_level is None:
            goal_level = self.goal_level
        old_obj = self.put_object(Goal(goal_level), goal_pos)
        self.goals.append((goal_pos, old_obj))
        return

    def remove_goal(self):
        goal_pos, obj_ = self.goals.pop()
        self.put_object(obj_, goal_pos)
        return

    def seed(self, seed=None):
        # Seed the random number generator
        self.np_random, _ = seeding.np_random(seed)
        return [seed]

    @property
    def state(self):
        return np.append(self.agent_pos, self.agent_dir)

    @property
    def goal_level(self):
        return len(self.goals)

    def reset_state_goal(self):
        # Pick a random position and direction for the agent
        while True:
            agent_x = self.np_random.randint(self.width)
            agent_y = self.np_random.randint(self.height)
            if self.grid.get(*self.to_grid_coords(np.array((agent_x, agent_y)))) is None:
                break
        agent_dir = self.np_random.randint(4)

        # Pick a different random position for the goal
        while True:
            goal_x = self.np_random.randint(self.width)
            goal_y = self.np_random.randint(self.height)
            if self.grid.get(*self.to_grid_coords(np.array((goal_x, goal_y)))) is None \
                    and (agent_x, agent_y) != (goal_x, goal_y):
                break

        self.agent_pos = np.array((agent_x, agent_y))
        self.agent_dir = agent_dir
        self.goal_pos = np.array((goal_x, goal_y))

        return

    def step(self, action):
        self.step_count += 1

        reward = -1
        done = False
        info = {}

        # Rotate left
        if action == self.actions.left:
            self.agent_dir = (self.agent_dir - 1) % 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            fwd = self.agent_pos + DIR_TO_VEC[self.agent_dir]
            if not isinstance(self.grid.get(*self.to_grid_coords(fwd)), Wall):
                self.agent_pos = fwd

        else:
            raise ValueError('Action out of bounds')

        if self.step_count >= self.max_steps:
            done = True
            info['TimeLimit.truncated'] = True

        if np.array_equal(self.agent_pos, self.goal_pos):
            done = True
            reward = 0

        return self.state, reward, done, info

    def render(self, mode='human', close=False, tile_size=32):
        """
        Render the whole-grid human view
        """

        if close:
            if self.window:
                self.window.close()
            return

        if mode == 'human' and not self.window:
            from .window import Window
            self.window = Window(self.name)
            self.window.show(block=False)

        # Render the whole grid
        img = self.grid.render(tile_size, self.to_grid_coords(self.agent_pos), self.agent_dir)

        if mode == 'human':
            # TODO add info, could be passed through a param
            # self.window.set_caption(self.mission)
            self.window.show_img(img)

        return img

    def close(self):
        if self.window:
            self.window.close()
        return

    def create_grid(self, width, height):
        self.grid = Grid(width + 2, height + 2)
        return

    def create_outer_wall(self):
        self.grid.wall_rect()
        return

    def put_object(self, obj: WorldObj, pos: np.ndarray):
        old_obj = self.grid.set(*self.to_grid_coords(pos), obj)
        return old_obj

    @staticmethod
    def to_grid_coords(a: np.ndarray):
        assert len(a) == 2
        return a + np.array((1, 1))

    @property
    def state_goal_mapper(self):
        def fn(state: np.ndarray):
            return state[:2]

        return fn
