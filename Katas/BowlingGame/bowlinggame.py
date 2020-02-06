class Game:
    def __init__(self):
        self.score = 0
        self.score_list = []

    def roll(self, pins_hit):
        if pins_hit == 'x':
            self.score_list.append(0)
        else:
            self.score_list.append(pins_hit)

    def get_score(self):
        score_final = 0
        frame_index = 0

        for frame in range(10):
            if self._isstrike(frame_index):
                score_final += 10 + self._strike_bonus(frame_index)
                frame_index+=1
            elif self._isspare(frame_index):
                score_final += 10 + self._spare_bonus(frame_index)
                frame_index +=2
            else:
                score_final += self.score_list[frame_index] + self.score_list[frame_index+1]
                frame_index += 2
        return score_final

    def _isspare(self, index):
        return self.score_list[index] + self.score_list[index+1] == 10

    def _isstrike(self, index):
        return self.score_list[index] == 10

    def _spare_bonus(self, index):
        return self.score_list[index + 2]

    def _strike_bonus(self, index):
        return self.score_list[index + 1] + self.score_list[index + 2]



