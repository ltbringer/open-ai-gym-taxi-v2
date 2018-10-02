# open-ai-gym-taxi-v2

[The official description](https://gym.openai.com/envs/Taxi-v2/)

## Installation

### Training the agent

Requires `Python 3`
```
$ pip install -r requirements.txt
$ python main.py
```

### Testing the agent
The training generates a csv with Q-table info of the form `alpha_{alpha_value}_gamma_{gamma_value}_score_{score}__{timestamp}.csv` 
```
$ python test.py <path-to-qtable.csv>
```

1. The taxi agent generally gets trained with a best average reward of **~9** in *50k episodes*.
2. The taxi agent gathers upto **8.5+** average rewards in a test of 100 episodes.

---

## Problem Statement
### 1. Environment description
```
+---------+
|R: | : :G|
| : : : : |
| : : : : |
| | : | : |
|Y| : |B: |
+---------+
```
1. The map is a `5x5` gridworld.
2. The alphabets `R`, `G`, `B`, `Y` are 4 locations.
3. A passenger can be at any of the 4 locations.
4. A passenger's destination can be any of the left 3 locations.
5. The pipe symbol `|` denotes a wall.
6. The colon symbol `:` denotes a pass.
7. The taxi can pass through `:` but not `|`
8. The environment rewards `20` points when a passenger is dropped to their destination.
9. The environment penalizes `-10` points if pickup operation is performed on a cell where there is no passenger.
10. The environment penalizes `-10` points if drop operation is performed if no passenger had boarded the taxi.
11. The environment penalizes `-1` for every other action.

### 2. Initial conditions
1. At the start, the taxi will be at any of the 25 positions on the map (from `Environment description[1]`).
2. A passenger will be at any of `R`, `G`, `B`, `Y` locations.
3. A destination will be at any of the `R`, `G`, `B`, `Y` locations.

### 3. Expected behaviour
1. The taxi must find the passenger traveling the shortest path.
2. The taxi must pickup the passenger.
3. The taxi must find the shortest path to the passenger's destination.
4. Drop the passenger at their destination traversing the shortest path.

## Qualify
OpenAI Gym defines ["solving" taxi-v1](https://gym.openai.com/envs/Taxi-v1/) task as getting average return of 9.7 over 100 consecutive trials.

