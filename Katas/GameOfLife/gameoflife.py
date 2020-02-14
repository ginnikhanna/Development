from Katas.GameOfLife import lifegrid

class Game:

    def __init__(self):
        self._grid = lifegrid.LifeGrid().make_grid(4,4)
        self._seeds = []

    def start_at(self, position):
        return

    def status(self):
