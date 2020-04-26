import abc

class AbstractPlayer(abc.ABC):
    @abc.abstractmethod
    def select_piece_to_move(self, observation):
        pass
