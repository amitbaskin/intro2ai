# TODO: DO NOT SUBMIT
__author__ = "Ram Friedman, Yonny Hammer"

import os
import time
import winsound

if __name__ == "__main__":

    programs = []
    programs.append('python game.py -p tiny_set.txt -s 4 7 -z fill')
    programs.append('python game.py -p tiny_set.txt -f bfs -s 4 7 -z fill')
    programs.append('python game.py -p tiny_set_2.txt -f bfs -s 6 6 -z corners')
    programs.append('python game.py -p tiny_set_2.txt -f ucs -s 6 6 -z corners')
    programs.append('python game.py -p small_set.txt -f ucs -s 5 5 -z corners')
    programs.append('python game.py -p tiny_set_2.txt -f astar -s 6 6 -z corners -H null_heuristic')
    programs.append('python game.py -p tiny_set_2.txt -f astar -s 8 8 -z corners -H blokus_corners_heuristic')
    programs.append('python game.py -p small_set.txt -f astar -s 6 6 -H null_heuristic -z cover -x 3 3 "[(2,2), (5, 5), (1, 4)]"')
    programs.append('python game.py -p small_set.txt -f astar -s 10 10 -H blokus_cover_heuristic -z cover -x 3 3 "[(2,2), (5, 5), (6, 7)]"')
    programs.append('python game.py -p valid_pieces.txt -s 10 10 -z sub-optimal -x 7 7 "[(5,5), (8,8), (4,9)]"')
    programs.append('python game.py -p valid_pieces.txt -s 10 10 -z sub-optimal -x 5 5 "[(3,4), (6,6), (7,5)]"')
    programs.append('python game.py -p valid_pieces.txt -s 10 10 -z mini-contest "[(0,1),(4,9),(9,2)]"')
    # programs.append('')

    for idx, program in enumerate(programs):
        print(f"{idx} - {program}")

        start = time.time()
        os.system(program)
        end = time.time()
        d = end-start
        print(f"{int(d//60)}m {round(d%60,2)}s")

        winsound.Beep(frequency=300, duration=200)
        print()
