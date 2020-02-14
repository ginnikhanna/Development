import unittest
from Katas.GameOfLife import lifegrid

class LifeGrid(unittest.TestCase):
    def test_that_grid_is_1x1(self):
        lg = lifegrid.LifeGrid()
        self.assertEqual(lg.make_grid(1,1), [[' ']])

    def test_that_grid_is_1x2(self):
         lg = lifegrid.LifeGrid()
         self.assertEqual(lg.make_grid(1,2), [[' ', ' ']])

    def test_that_grid_is_2x1(self):
         lg = lifegrid.LifeGrid()
         self.assertEqual(lg.make_grid(2,1), [[' '],[' ']])

    def test_that_grid_is_2x2(self):
         lg = lifegrid.LifeGrid()
         self.assertEqual(lg.make_grid(2, 2), [[' ', ' '], [' ', ' ']])

    def test_that_position_0_0_of_grid_is_filled(self):
        lg = lifegrid.LifeGrid()
        grid = lg.make_grid(2,2)
        self.assertEqual(lg.update_grid(grid, 0,0), [['*', ' '], [' ', ' ']] )

    def test_that_position_1_2_of_grid_is_filled(self):
        lg = lifegrid.LifeGrid()
        grid = lg.make_grid(3,3)
        self.assertEqual(lg.update_grid(grid, 1,2), [[' ', ' ', ' '],
                                                     [' ', ' ', '*'],
                                                     [' ', ' ', ' ']] )


if __name__ == '__main__':
    unittest.main()
