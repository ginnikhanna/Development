class LifeGrid:

    def __init__(self):
        pass
    def _make_grid(self, nr_rows:int, nr_cols: int):
        return [[' ']*nr_cols for _ in range(nr_rows)]

    def make_grid(self, nr_rows :int, nr_cols:int):
        return self._make_grid(nr_rows, nr_cols)

    def update_grid(self, grid, row_index:int, col_index:int):
        grid[row_index][col_index] = '*'
        return grid