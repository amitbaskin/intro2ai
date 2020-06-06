import time


class SummaryDisplay(object):
    def __init__(self):
        super(SummaryDisplay, self).__init__()
        self.scores = []
        self.highest_tile = []
        self.game_durations = []
        self.game_start_time = None

    def initialize(self, initial_state):
        self.game_start_time = time.time()

    def update_state(self, new_state, action, opponent_action):
        if new_state.done:
            game_end_time = time.time()
            game_duration = game_end_time - self.game_start_time
            print("score: %s\nhighest tile: %s\ngame_duration: %s" % (new_state.score, new_state.board.max(),
                                                                      game_duration))
            self.scores.append(new_state.score)
            self.highest_tile.append(new_state.board.max())
            self.game_durations.append(game_duration)

    def mainloop_iteration(self):
        pass

    def print_stats(self):
        win_rate = len(list(filter(lambda x: x >= 2048, self.highest_tile))) / len(self.highest_tile)
        print("="*30)
        print("scores: %s" % self.scores)
        print("highest tile: %s" % self.highest_tile)
        print("game_durations: %s" % self.game_durations)
        print("win rate: %s" % win_rate)
        print("games: %s" % len(self.scores))
        print("64: %s" % len([x for x in self.highest_tile if x == 64]))
        print("128: %s" % len([x for x in self.highest_tile if x == 128]))
        print("512: %s" % len([x for x in self.highest_tile if x == 512]))
        print("1024: %s" % len([x for x in self.highest_tile if x == 1024]))
        print("2048: %s" % len([x for x in self.highest_tile if x == 2048]))
        print("4096: %s" % len([x for x in self.highest_tile if x == 4096]))
        print("Percents: ")
        print("64: %s" % (len([x for x in self.highest_tile if x == 64]) / len(self.scores)))
        print("128: %s" % (len([x for x in self.highest_tile if x == 128]) / len(self.scores)))
        print("512: %s" % (len([x for x in self.highest_tile if x == 512]) / len(self.scores)))
        print("1024: %s" % (len([x for x in self.highest_tile if x == 1024]) / len(self.scores)))
        print("2048: %s" % (len([x for x in self.highest_tile if x == 2048]) / len(self.scores)))
        print("4096: %s" % (len([x for x in self.highest_tile if x == 4096]) / len(self.scores)))
