from dataclasses import dataclass


@dataclass(eq=True, frozen=True)
class GoPoint:
    x: int
    y: int

    def neighbors(self):
        return [
            GoPoint(self.x - 1, self.y),
            GoPoint(self.x + 1, self.y),
            GoPoint(self.x, self.y - 1),
            GoPoint(self.x, self.y + 1),
        ]

    def __str__(self):
        return f"({self.x},{self.y})"
