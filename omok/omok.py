import pygame
from typing import Tuple
from enum import Enum, auto


BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
EMPTY = (128, 128, 128)


class StateAfterPutStone(Enum):
    BLACK_WIN = 0
    WHITE_WIN = auto()
    DRAW = auto()
    CONTINUE = auto()
    FAIL_INVALID_POSITION = auto()
    FAIL_INVALID_TURN = auto()


class Stone(Enum):
    EMPTY = 0
    BLACK = auto()
    WHITE = auto()


class Omok:
    def __init__(
        self,
        board_size: int = 15,
        verbose: bool = False,
    ):
        self.board_size: Tuple[int, int] = (board_size, board_size)
        self.verbose: bool = verbose
        self.reset()

        self._pygame_init_if_not_initialized()

    def put_stone(self, stone: Stone, x: int, y: int) -> StateAfterPutStone:
        if not self._is_valid_position(x, y):
            return StateAfterPutStone.FAIL_INVALID_POSITION

        if not self._is_valid_turn(stone):
            return StateAfterPutStone.FAIL_INVALID_TURN

        self.board[x][y] = stone

        if self._check_five_in_line(x, y):
            if self.verbose:
                print(f"{stone} wins!")
            return StateAfterPutStone.BLACK_WIN if stone == Stone.BLACK else StateAfterPutStone.WHITE_WIN

        if self._is_board_full():
            if self.verbose:
                print("Draw!")
            return StateAfterPutStone.DRAW

        self._toggle_turn()

        return StateAfterPutStone.CONTINUE

    def render_current_board(self):
        self._pygame_init_if_not_initialized()
        self.screen.fill(EMPTY)

        ##### render grid #####
        for i in range(self.board_size[0] + 1):
            pygame.draw.line(
                self.screen,
                BLACK,
                self._translate_position_about_padding(0, i * self.cell_size),
                self._translate_position_about_padding(self.screen_size, i * self.cell_size),
            )
            pygame.draw.line(
                self.screen,
                BLACK,
                self._translate_position_about_padding(i * self.cell_size, 0),
                self._translate_position_about_padding(i * self.cell_size, self.screen_size),
            )

        ##### render stones #####
        for row in range(self.board_size[0]):
            for col in range(self.board_size[1]):
                if self.board[row][col] == Stone.BLACK:
                    pygame.draw.circle(
                        self.screen,
                        BLACK,
                        self._translate_position_about_padding(
                            col * self.cell_size + self.cell_size // 2,
                            row * self.cell_size + self.cell_size // 2,
                        ),
                        self.cell_size // 2,
                    )
                elif self.board[row][col] == Stone.WHITE:
                    pygame.draw.circle(
                        self.screen,
                        WHITE,
                        self._translate_position_about_padding(
                            col * self.cell_size + self.cell_size // 2,
                            row * self.cell_size + self.cell_size // 2,
                        ),
                        self.cell_size // 2,
                    )

        pygame.display.flip()

    def reset(self):
        self.board = [[Stone.EMPTY for _ in range(self.board_size[1])] for _ in range(self.board_size[0])]
        self.current_turn = Stone.BLACK

    def observation(self):
        result = []
        for row in self.board:
            line = []
            for cell in row:
                line.append(cell.value)
            result.append(line)
        return result

    def opposite_observation(self):
        """Return the board with the opposite stone color.

        Agent would be trained as black stone.
        So, when the agent need to do a role of white stone, it should use this method.
        """
        result = []
        for row in self.board:
            line = []
            for cell in row:
                if cell == Stone.BLACK:
                    line.append(Stone.WHITE.value)
                elif cell == Stone.WHITE:
                    line.append(Stone.BLACK.value)
                else:
                    line.append(Stone.EMPTY.value)
            result.append(line)
        return result

    def _is_board_full(self):
        return all(cell != Stone.EMPTY for row in self.board for cell in row)

    def _check_five_in_line(self, row, col):
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for direction in directions:
            dx, dy = direction
            count = 1
            ops = ["+", "-"]
            for op in ops:
                for i in range(1, 5):
                    x = row + i * dx if op == "+" else row - i * dx
                    y = col + i * dy if op == "+" else col - i * dy

                    if not self._is_valid_range(x, y):
                        break

                    if self.board[x][y] == self.board[row][col]:
                        count += 1
                    else:
                        break
            if count >= 5:
                return True
        return False

    def _pygame_init_if_not_initialized(self):
        should_initialize = False
        try:
            should_initialize = self.screen is None
        except AttributeError:
            should_initialize = True

        if should_initialize:
            pygame.init()
            self.screen_size = 600
            self.padding_size = 50
            self.screen = pygame.display.set_mode(
                (self.screen_size + self.padding_size * 2, self.screen_size + self.padding_size * 2)
            )
            self.cell_size = self.screen_size // self.board_size[0]

    def _translate_position_about_padding(self, x: int, y: int) -> Tuple[int, int]:
        return x + self.padding_size, y + self.padding_size

    def _is_valid_position(self, x: int, y: int) -> bool:
        return self._is_valid_range(x, y) and self.board[x][y] == Stone.EMPTY

    def _is_valid_range(self, x: int, y: int) -> bool:
        return 0 <= x < self.board_size[0] and 0 <= y < self.board_size[1]

    def _is_valid_turn(self, stone: Stone) -> bool:
        assert stone != Stone.EMPTY
        return self.current_turn == stone

    def _toggle_turn(self):
        self.current_turn = Stone.BLACK if self.current_turn == Stone.WHITE else Stone.WHITE


if __name__ == "__main__":
    omok = Omok()

    while True:
        omok.render_current_board()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                x = (x - omok.padding_size) // omok.cell_size
                y = (y - omok.padding_size) // omok.cell_size

                state = omok.put_stone(omok.current_turn, y, x)
                match state:
                    case StateAfterPutStone.BLACK_WIN:
                        print("BLACK WIN!")
                        omok.reset()
                    case StateAfterPutStone.WHITE_WIN:
                        print("WHITE WIN!")
                        omok.reset()
                    case StateAfterPutStone.FAIL_INVALID_POSITION:
                        print("Invalid position")
                    case StateAfterPutStone.FAIL_INVALID_TURN:
                        print("Invalid turn")
                    case StateAfterPutStone.CONTINUE:
                        pass
