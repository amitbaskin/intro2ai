311604938
312259013
*****
Comments:
To make sure that the board stays in a state that is conducive to merges we made a few heuristics:
1. The board is monotone in a direction, this means that the highest tile is more likely to stay at a corner and the
   further away you get from it, the smaller the numbers get, this means that merges of small numbers happen in one
   corner of the board, and big merges happen in the opposite side. We ignore empty tiles that are in between the
   tiles.
2. To make sure tiles stay in places where the can be merged, we want adjacent tiles to be similar. We ignore empty
   tiles that are in between the tiles.
3. We want to make sure that there are places in which tiles can move, so we want to maximize the number of empty
   tiles.
4. We obviously want the game to have the highest tile possible.
5. Also for the sum  of the tiles on board.
6. We want to make extra sure that the biggest tile is in one of the corners to make sure that this event cannot
   happen: The biggest tile moves for "one turn" to make something happen, and then a tile spawns in the corner where
           it would have returned to.
   And so, we added this heuristic.
7.  Obviously, the score itself is important as well.

All these heuristics are weighed in a manner such that they are all of the same order of magnitude more or less.