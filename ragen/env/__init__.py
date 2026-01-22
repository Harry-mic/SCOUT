# from .alfworld.config import AlfredEnvConfig
# from .alfworld.env import AlfredTXTEnv
from .bandit.config import BanditEnvConfig
from .bandit.env import BanditEnv
from .countdown.config import CountdownEnvConfig
from .countdown.env import CountdownEnv
from .sokoban.config import SokobanEnvConfig
from .sokoban.env import SokobanEnv
from .frozen_lake.config import FrozenLakeEnvConfig
from .frozen_lake.env import FrozenLakeEnv
from .metamathqa.env import MetaMathQAEnv
from .metamathqa.config import MetaMathQAEnvConfig
from .lean.config import LeanEnvConfig
from .lean.env import LeanEnv
from .game_2048.config import Game2048EnvConfig
from .game_2048.env import Game2048Env
from .blackjack.config import BlackjackEnvConfig
from .blackjack.env import BlackjackEnv
from .rubikscube.config import RubiksCube2x2Config
from .rubikscube.env import RubiksCube2x2Env
from .sudoku.config import SudokuEnvConfig
from .sudoku.env import SudokuEnv


REGISTERED_ENVS = {
    'bandit': BanditEnv,
    'countdown': CountdownEnv,
    'sokoban': SokobanEnv,
    'frozen_lake': FrozenLakeEnv,
    # 'alfworld': AlfredTXTEnv,
    'metamathqa': MetaMathQAEnv,
    'lean': LeanEnv,
    'game_2048': Game2048Env,
    'blackjack': BlackjackEnv,
    'rubikscube': RubiksCube2x2Env,
    'sudoku': SudokuEnv,
}

REGISTERED_ENV_CONFIGS = {
    'bandit': BanditEnvConfig,
    'countdown': CountdownEnvConfig,
    'sokoban': SokobanEnvConfig,
    'frozen_lake': FrozenLakeEnvConfig,
    # 'alfworld': AlfredEnvConfig,
    'metamathqa': MetaMathQAEnvConfig,
    'lean': LeanEnvConfig,
    'game_2048': Game2048EnvConfig,
    'blackjack': BlackjackEnvConfig,
    'rubikscube': RubiksCube2x2Config,
    'sudoku': SudokuEnvConfig,
}

try:
    from .webshop.env import WebShopEnv
    from .webshop.config import WebShopEnvConfig
    REGISTERED_ENVS['webshop'] = WebShopEnv
    REGISTERED_ENV_CONFIGS['webshop'] = WebShopEnvConfig
except ImportError:
    pass
