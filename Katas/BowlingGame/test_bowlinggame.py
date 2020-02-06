import unittest
from Katas.BowlingGame import bowlinggame

def roll_many(game: bowlinggame.Game,
              number_of_throws : int,
              pins_hit : int):

    for i in range(number_of_throws):
        game.roll(pins_hit)

    return game

def roll_strike(game:bowlinggame.Game):
    game.roll(10)


def roll_spare(game:bowlinggame.Game):
    game.roll(5)
    game.roll(5)


class BowlingGameTest(unittest.TestCase):


    def test_gutter_game(self):
        number_of_throws = 20
        pins_hit = 0
        game = bowlinggame.Game()
        game = roll_many(game, number_of_throws, pins_hit)
        self.assertEqual(0, game.get_score())

    def test_game_score_with_all_ones_in_all_throws(self):
        number_of_throws = 20
        pins_hit = 1
        game = bowlinggame.Game()
        game = roll_many(game, number_of_throws, pins_hit)
        self.assertEqual(20, game.get_score())


    def test_game_with_different_pins_hit_on_different_throws(self):
        pins_hit_list = [2,4]
        game = bowlinggame.Game()
        game.roll(pins_hit_list[0])
        game.roll(pins_hit_list[1])
        roll_many(game, 18, 0)
        self.assertEqual(6, game.get_score())


    def test_game_with_one_spare(self):
        game = bowlinggame.Game()
        roll_spare(game)
        game.roll(5)
        roll_many(game, 17, 0)
        self.assertEqual(20, game.get_score())

    def test_game_with_one_strike(self):
        game = bowlinggame.Game()
        roll_strike(game)
        game.roll(4)
        game.roll(4)
        game.roll(5)
        game.roll(4)
        roll_many(game,15,0)

        self.assertEqual(35, game.get_score())

    def test_over_all(self):
        game = bowlinggame.Game()
        for i in range(12):
            roll_strike(game)

        self.assertEqual(300, game.get_score())





if __name__ == '__main__':
    unittest.main()
