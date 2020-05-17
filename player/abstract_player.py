import abc

class AbstractPlayer(abc.ABC):
    @abc.abstractmethod
    def __init__(self):
        super().__init__()
        self.wincount = 0

    @abc.abstractmethod
    def select_piece_to_move(self, observation):
        pass
