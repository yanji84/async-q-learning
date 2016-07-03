# async-q-learning

My inplementation of the asynchronous deep reinforcement learning update. The key idea is to utilize multiple agents running simultaneously to break the training experience correlation and scale the model convergence linearly with the number of threads

Link to paper: https://arxiv.org/pdf/1602.01783v2.pdf

Required Dependencies
- tensorflow
- opencv
- openai gym API