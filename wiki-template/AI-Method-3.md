# AI Method 3 -Breadth First Search


### Motivation  

As a skeleton BFS agent was provided, we wanted to improve this agent so that it could serve as a testing agent prototype and provide insight for our design in the MCTS agent. 

### Application  
The provided agent automatically executes the first available move (gem collection), as all actions have 0 reward in the skeleton code. We added an invocation of getLegalActions to obtain a list of actions available, and implemented rewards for buying cards or receiving noble cards. For each new state, the action with the highest reward is selected, and if all actions yield 0 reward, their paths and states are added to the queue for further exploration. 

### Trade-offs  
For simplicity, we only granted a reward based on the action type, while not taking the game states into account. We did not consider how the game state changes upon executing an action, and how the board state and opponent state may have an impact on the reward received for an action. Moreover, we only applied rewards to “buy” and “noble” actions, while other actions are associated with 0 reward. 

#### *Advantages*  
It is an easy implementation and significantly improves the original BFS agent as it now preferably selects actions that may lead to a reward. Moreover, it is more computationally efficient compared to the previous one, as we replaced deepcopy with copy by serialising and deserializing the game states. 


#### *Disadvantages*
The implementation is incomplete compared to our MCTS agent as we overlooked the ever-changing game states and rewards for other actions.As a result, the agent is unable to plan ahead and choose an action that can maximize its score in the long run.  It spends most of the time collecting and returning random gems.



