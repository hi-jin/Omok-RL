import gymnasium as gym
import socket
from typing import Literal, Any
import ast
import numpy as np
import torch


class OmokEnv(gym.Env):
    def __init__(
        self,
        server_port: int,
        who: Literal["black", "white"],
    ) -> None:
        super().__init__()
        self.server_port = server_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server = ("localhost", server_port)
        self.who = who

        self.observation_space = gym.spaces.Box(
            low=0,
            high=2,
            shape=(15, 15),
            dtype=np.int8,
        )
        self._last_obs = None
        self.action_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(15, 15),
            dtype=np.float32,
        )

        self._register_to_server()

    def _register_to_server(self):
        self.sock.sendto(f"whoami {self.who}".encode(), self.server)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        board, _, _ = self._get_obs()
        return board, {}

    def _get_obs(self):
        message, _ = self.sock.recvfrom(2048)
        message = message.decode()
        commands = message.split(" ")
        cmd = commands.pop(0)
        args = commands

        board = None
        reward = 0
        done = False

        if cmd == "obs":
            board = ast.literal_eval("".join(args))
            board = np.array(board, dtype=np.int8)
            self._last_obs = board
        else:
            if cmd == "win":
                reward = 1
                done = True
            elif cmd == "lose":
                reward = -1
                done = True
            elif cmd == "draw":
                reward = 0
                done = True
            else:
                raise ValueError(f"unknown cmd {cmd}")

        return board, reward, done

    def step(self, action):
        invalid = self._last_obs != 0
        action[invalid] = -np.inf
        flat_probs = torch.softmax(torch.tensor(action.flatten()), dim=0)
        chosen_index = np.random.choice(len(flat_probs), p=flat_probs.numpy())
        chosen_action = np.unravel_index(chosen_index, action.shape)

        i, j = chosen_action[0], chosen_action[1]
        self.sock.sendto(f"put {i} {j}".encode(), self.server)
        board, reward, done = self._get_obs()

        return board, reward, done, False, {}

    def render(self):
        # TODO
        return None
