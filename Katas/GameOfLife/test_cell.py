import unittest
from Katas.GameOfLife import cell
from Katas.GameOfLife.cell import State

class CellShould(unittest.TestCase):

    def test_cell_state_is_alive(self):
        state = State.ALIVE
        creature = cell.Cell(state)
        self.assertEqual(creature.cell_state(), cell.State.ALIVE)

    def test_cell_state_is_dead(self):
        state = State.DEAD
        creature = cell.Cell(state)
        self.assertEqual(creature.cell_state(), cell.State.DEAD)

    def test_cell_is_dead_with_0_neighbors(self):
        state = State.ALIVE
        creature = cell.Cell(state)
        self.assertEqual(creature.deadoralive(0), cell.State.DEAD)

    def test_cell_is_dead_with_1_neighbors(self):
        state = State.ALIVE
        creature = cell.Cell(state)
        self.assertEqual(creature.deadoralive(1), cell.State.DEAD)

    def test_cell_is_alive_with_2_neighbors(self):
        state = State.ALIVE
        creature = cell.Cell(state)
        self.assertEqual(creature.deadoralive(2), cell.State.ALIVE)

    def test_cell_is_alive_with_3_neighbors(self):
        state = State.ALIVE
        creature = cell.Cell(state)
        self.assertEqual(creature.deadoralive(3), cell.State.ALIVE)

    def test_cell_is_dead_with_more_than_3_neighbours(self):
        state = State.ALIVE
        creature = cell.Cell(state)
        self.assertEqual(creature.deadoralive(4), cell.State.DEAD)

if __name__ == '__main__':
    unittest.main()
