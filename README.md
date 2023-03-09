# CS7IS2 Artificial Intelligence Assignment 1
This is the first assignment for the module Artificial Intelligence in Trinity College Dublin (TCD) for the year 2022/2023.

The base code for the pacman maze was adapted from Pacman AI project from UC Berkeley (http://ai.berkeley.edu).

## Algorithms Currently Working
- [x] depthFirstSearch
- [x] breadthFirstSearch
- [x] aStarSearch
- [x] MDP value iteration
- [ ] MDP policy iteration

## Commands
If you want to run the basic pacman game.
```
python pacman.py
```
### Search Algorithms
For running search algorithms to solve mazes of different sizes.
```
python pacman.py -l tinyMaze -p SearchAgent -a fn=breadthFirstSearch
python pacman.py -l tinyMaze -p SearchAgent -a fn=depthFirstSearch
python pacman.py -l tinyMaze -p SearchAgent -a fn=aStarSearch
python pacman.py -l tinyMaze -p SearchAgent -a fn=aStarSearch,heuristic=manhattanHeuristic
python pacman.py -l tinyMaze -p SearchAgent -a fn=aStarSearch,heuristic=euclideanHeuristic

python pacman.py -l mediumMaze -p SearchAgent -a fn=breadthFirstSearch
python pacman.py -l mediumMaze -p SearchAgent -a fn=depthFirstSearch
python pacman.py -l mediumMaze -p SearchAgent -a fn=aStarSearch
python pacman.py -l mediumMaze -p SearchAgent -a fn=aStarSearch,heuristic=manhattanHeuristic
python pacman.py -l mediumMaze -p SearchAgent -a fn=aStarSearch,heuristic=euclideanHeuristic

python pacman.py -l bigMaze -p SearchAgent -a fn=breadthFirstSearch
python pacman.py -l bigMaze -p SearchAgent -a fn=depthFirstSearch
python pacman.py -l bigMaze -p SearchAgent -a fn=aStarSearch
python pacman.py -l bigMaze -p SearchAgent -a fn=aStarSearch,heuristic=manhattanHeuristic
python pacman.py -l bigMaze -p SearchAgent -a fn=aStarSearch,heuristic=euclideanHeuristic
```
### Markov Decision Process Algorithms
```
python pacman.py -l tinyMaze -p MDPAgent
python pacman.py -l mediumMaze -p MDPAgent
python pacman.py -l bigMaze -p MDPAgent
```
