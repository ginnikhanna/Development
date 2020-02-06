import unittest
from Katas.TicTacToe import tictactoegame


class TicTacToeGameTest(unittest.TestCase):

    def test_tictactoe(self):
        game = tictactoegame.TictactoeGame()

    def test_number_of_fields_in_grid(self):
        game = tictactoegame.TictactoeGame()
        self.assertEqual(9, game.number_of_fields_in_grid())

    def test_grid_size(self):
        game = tictactoegame.TictactoeGame()
        self.assertEqual((3, 3), game.grid_size())

    def test_play_X_on_empty_field(self):
        game = tictactoegame.TictactoeGame()
        game.play('X', 0, 0)
        self.assertEqual('X', game.grid[0][0])

    def test_play_O_on_empty_field(self):
        game = tictactoegame.TictactoeGame()
        game.play('O', 0, 1)
        self.assertEqual('O', game.grid[0][1])

    def test_playing_X_on_full_field_should_raise_error(self):
        game = tictactoegame.TictactoeGame()
        game.play('O', 0, 1)
        self.assertRaises(ValueError, game.play, 'X', 0, 1)

    def test_player_wins_when_row_is_full(self):
        game = tictactoegame.TictactoeGame()
        game.play('X', 0, 0)
        game.play('X', 0, 1)
        game.play('X', 0, 2)

        self.assertEqual(True, game.check_winner_for_row('X', 0))

    def test_player_wins_when_col_is_full(self):
        game = tictactoegame.TictactoeGame()
        game.play('X', 0, 0)
        game.play('X', 1, 0)
        game.play('X', 2, 0)

        self.assertEqual(True, game.check_winner_for_col('X', 0))

    def test_player_wins_when_diagonal_is_full(self):
        game = tictactoegame.TictactoeGame()
        game.play('X', 0, 0)
        game.play('X', 1, 1)
        game.play('X', 2, 2)

        self.assertEqual(True, game.check_winner_for_col('X', 0))

    def test_grid_is_full(self):
        game = tictactoegame.TictactoeGame()
        game.play('X', 0, 0)
        game.play('0', 0, 1)
        game.play('X', 0, 2)

        game.play('0', 1, 0)
        game.play('X', 1, 1)
        game.play('0', 1, 2)

        game.play('X', 2, 0)
        game.play('0', 2, 1)
        game.play('X', 2, 2)

        self.assertEqual(True, game.check_if_grid_is_full())

    def test_game_over_X_wins(self):
        game = tictactoegame.TictactoeGame()
        game.play('X', 0, 0)
        game.play('X', 1, 1)
        game.play('X', 2, 2)

        game.check_game_over()
        self.assertEqual('X', game.winning_player)

    def test_game_over_O_wins(self):
        game = tictactoegame.TictactoeGame()
        game.play('O', 0, 0)
        game.play('O', 0, 1)
        game.play('O', 0, 2)

        game.check_game_over()
        self.assertEqual('O', game.winning_player)

    def test_game_over_draws(self):
        game = tictactoegame.TictactoeGame()

        game.play('X', 0, 0)
        game.play('0', 0, 1)
        game.play('X', 0, 2)

        game.play('0', 1, 0)
        game.play('X', 1, 1)
        game.play('0', 1, 2)

        game.play('X', 2, 0)
        game.play('0', 2, 1)
        game.play('X', 2, 2)

        self.assertEqual('', game.winning_player)


if __name__ == '__main__':
    unittest.main()
