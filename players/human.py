from mancala import STORE

class Human():
    def __init__(self, name='human'):
        self.name = name

    def act(self, game):
        pit = int(input("Enter pit number (0-5): "))
        while pit not in range(STORE) or game.is_pit_empty(game.current_player, pit):
            pit = int(input("Invalid choice. Enter pit number (0-5) that has stones: "))
        return pit