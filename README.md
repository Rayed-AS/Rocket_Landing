The aim of this project was to see whether Artificial Intelligence could be used to land rockets in an environment that ignores air resistance
and horizontal movement.

A Deep Q Network Model was deployed in a virtual environment built using turtle where the model had to activate and vary thrust to control the
velocity of the rocket so that it does not crash into the surface.

The code behind the GUI and turtle implementation was AI generated however, the Deep Q Network was based on the GeeksForGeeks implementation 
but I used my own knowledge (such as the parts of the training process behind DQN) to modify the code to fit this problem and make it my own work.

A variable called time was introduced that was increased with each iteration. The velocity was found by adding the initial velocity (40) and the acceleration multiplied by the time.
This allowed a dynamic environment to be created where the DQN model has to navigate consistently changing properties while also being mindfuel of fuel.
The rocket starts with 100 fuel and with each action, the fuel is decreased. This prompts the model to be efficient and conservative.

The goal of this project was to gain a deeper understanding into how machine learning and artificial intelligence are utilised in the astronautical industry. Although this simulation is not very accurate, it gives an idea on how real organisations train models to land rockets in real life.

Original Code:
https://www.geeksforgeeks.org/deep-learning/implementing-deep-q-learning-using-tensorflow/
