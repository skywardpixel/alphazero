import enum


class Player(enum.Enum):
    @property
    def opponent(self):
        raise NotImplementedError
