import numpy as np


class TictactoeGame:

    def __init__(self):
        self.grid = [['', '', ''],
                     ['', '', ''],
                     ['', '', '']]
        self.winning_player = ''

    def number_of_fields_in_grid(self):
        return len(self.grid) * len(self.grid[0])

    def grid_size(self):
        return (len(self.grid), len(self.grid[0]))

    def play(self,
             player: str,
             row: int,
             col: int):

        if self.grid[row][col] == '':
            self.grid[row][col] = player
        else:
            raise ValueError

    def check_winner_for_row(self,
                             player: str,
                             row: int):

        if self.grid[row][0] == player and self.grid[row][1] == player and self.grid[row][2] == player:
            return True

    def check_winner_for_col(self,
                             player: str,
                             col: int):

        if self.grid[0][col] == player and self.grid[0][col] == player and self.grid[0][col] == player:
            return True


    def check_winner_for_diagonal(self,
                             player: str):

        if self.grid[0][0] == player and self.grid[1][1] == player and self.grid[2][2] == player:
            return True


    def check_if_grid_is_full(self):
        status = [True for x in range(len(self.grid)) for y in self.grid[x] if y !='']
        if status == [True] * 9:
            return True

    def check_game_over(self):
        for i in range(3):
            if self.check_winner_for_col('X', i) == True:
                self._player_wins('X')
            elif self.check_winner_for_col('O', i) == True:
                self._player_wins('O')
            elif self.check_winner_for_row('X', i) == True:
                self._player_wins('X')
            elif self.check_winner_for_row('O', i) == True:
                self._player_wins('O')

        if self.check_if_grid_is_full() == True:
            self._game_over()

        elif self.check_winner_for_diagonal('X')  == True:
            self._player_wins('X')

        elif self.check_winner_for_diagonal('O') == True:
            self._player_wins('O')

        else:
            print('Continue')

    def _player_wins(self, player: str):
        self.winning_player = player
        return (f'{self.winning_player} WINS')

    def _game_over(self):
        print('Draw')

if __name__ == '__main__':

    game  = TictactoeGame()

    game.play('X', 0, 0)
    game.check_game_over()
    game.play('0', 0, 1)
    game.check_game_over()
    game.play('X', 0, 1)
    game.check_game_over()

