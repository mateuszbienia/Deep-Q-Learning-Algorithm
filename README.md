# Deep-Q-Learning-Algorithm

This project was created for my master thesis "Weak and strong artificial intelligence on the example of the game of "Connect 4". Project show implementation of reinforcement learning algorithm - Q-learning with deep network model which make Deep Q-learning Network[1]. Main objective of the work was to compare older game-solving algorithm such as min-max with newer - ones that uses neural networks to find optimal solutions. 

Project also includes 2 game enviorments:
 - Tic Tac Toe,
 - Connect 4,

 made for machine learning porpouse. Environments were created to reduce computation time, their implenetration is based on bit arrays. Impementation allows to use any enviormant as long as it is compatibile with OpenAI gym package. Different environments may require different main game loops, depending on the representation of the game state and experience (ReplayMemory).

## Abstract
Machine learning is becoming an increasingly popular field of computer science. As a result, many people apply it in various directions, one of which is the field of games. More and more complex algorithms are being created, outperforming even the greatest professionals in a given game. 
For the benefit of the paper, a comparison was made between a weak artificial intelligence --- Alpha-Beta pruning, which relies on human knowledge, in the form of a game state evaluation function, and a strong artificial intelligence, which learns to play on its own, based only on information gained directly from playing many games of a given game. It was tested whether the ,,strong'' agent is able to compete with the long-known Alpha-Beta pruning algorithm in the Connect 4 game. Using reinforcement learning methods and neural networks, an agent was created using a combination of the Q-learning algorithm and deep neural network --- Deep Q-learning Network. It was investigated which of its parameters are key to achieving high movement accuracy. It was then compared with the frequently used Alpha-Beta pruning algorithm. 

## References
Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., ... & Hassabis, D. (2015). Human-level control through deep reinforcement learning. nature, 518(7540), 529-533.