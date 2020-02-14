from enum import Enum

class State(Enum):
    ALIVE = 1
    DEAD = 0

class Cell:

    def __init__(self, state :State):
        self._state = state

    def cell_state(self):
        return self._state

    def deadoralive(self, nr_of_neighbors:int):
        if nr_of_neighbors == 0 or nr_of_neighbors == 1:
            self._state = State.DEAD
        elif nr_of_neighbors == 2 or nr_of_neighbors == 3:
            self._state = State.ALIVE
        elif self._state == State.ALIVE and nr_of_neighbors > 3:
            self._state = State.DEAD

        return self._state

