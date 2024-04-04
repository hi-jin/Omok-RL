import numpy as np
import gymnasium as gym
from omok.omok import Omok, StateAfterPutStone, Stone


class OmokEnv(gym.Env):
    def __init__(
        self,
        omok: Omok,
        agent_stone: Stone = Stone.BLACK,
        render_mode: str = "human",
    ):
        super().__init__()
        self.main_agent = None
        self.copied_agent = None
        self.render_mode = render_mode

        self.omok = omok
        self.agent_stone = agent_stone

        self.action_space = gym.spaces.MultiDiscrete([15, 15])
        self.observation_space = gym.spaces.Box(
            low=0,
            high=2,
            shape=(15, 15),
            dtype=np.uint8,
        )

    def observation(self):
        if self.agent_stone == Stone.BLACK:
            return np.array(self.omok.observation())
        else:
            return np.array(self.omok.opposite_observation())

    def render(self):
        if self.render_mode == "human":
            self.omok.render_current_board()
        else:
            return self.omok.render_current_board(render_mode=self.render_mode)  # type: ignore

    def reset(self, seed=None):
        self.omok.reset()
        self.render()
        return self.observation(), {}

    def step(self, action):
        x, y = action
        state_after_put_stone = self.omok.put_stone(self.agent_stone, x, y)
        self.render()
        match state_after_put_stone:
            case StateAfterPutStone.BLACK_WIN:
                return (
                    self.observation(),
                    10 if self.agent_stone == Stone.BLACK else -10,
                    True,
                    False,
                    {},
                )
            case StateAfterPutStone.WHITE_WIN:
                return (
                    self.observation(),
                    10 if self.agent_stone == Stone.WHITE else -10,
                    True,
                    False,
                    {},
                )
            case StateAfterPutStone.CONTINUE:
                # opponent's turn
                # TODO : MinMAX 알고리즘
                while True:
                    action, _ = self.copied_agent.predict(
                        self.observation(),
                    )
                    x, y = action
                    state_after_put_stone = self.omok.put_stone(
                        Stone.WHITE if self.agent_stone == Stone.BLACK else Stone.BLACK,
                        x,
                        y,
                    )
                    self.render()

                    match state_after_put_stone:
                        case StateAfterPutStone.BLACK_WIN:
                            return (
                                self.observation(),
                                10 if self.agent_stone == Stone.BLACK else -10,
                                True,
                                False,
                                {},
                            )
                        case StateAfterPutStone.WHITE_WIN:
                            return (
                                self.observation(),
                                10 if self.agent_stone == Stone.WHITE else -10,
                                True,
                                False,
                                {},
                            )
                        case StateAfterPutStone.CONTINUE:
                            return (
                                self.observation(),
                                0,
                                False,
                                False,
                                {},
                            )
                        case StateAfterPutStone.DRAW:
                            return (
                                self.observation(),
                                0,
                                True,
                                False,
                                {},
                            )
                        case _:
                            continue
            case StateAfterPutStone.FAIL_INVALID_POSITION:
                return (
                    self.observation(),
                    -1,
                    False,
                    False,
                    {},
                )
            case StateAfterPutStone.FAIL_INVALID_TURN:
                return (
                    self.observation(),
                    -100,  # never reach here (maybe....)
                    False,
                    False,
                    {},
                )
