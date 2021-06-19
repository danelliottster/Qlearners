# Overview

The purpose of this repository is to provide meaninful baselines for a variety reinforcement learning approaches.  The RL approaches are evaluated with an emphasis on the average performance accross multiple agents.

# Q-learning algorithms

# Recipies

<table>

   <tr>
      <td> <h2> <b> OpenAI cartpole v1 </b> </h2> </td>
   </tr>
   <tr>
      <td> Description: See https://gym.openai.com/envs/CartPole-v1/ for more information. </td>
   </tr>
   <tr>
      <td> <h4> Agent description </h4> </td>
      <td> <h4> Representative parameters </h4> </td>
      <td> <h4> Mean performance accross thirty agents </h4> </td>
   </tr>
   <tr>
      <td> SGD with feedforward ANN </td>
   </tr>
   <tr>
      <td> <h2> <b> Colorado State Univ cartpole swing-up and balance task </b> </h2> </td>
   </tr>
   <tr>
      <td> An inverted pendulum on a cart initially developed by Chuck Anderson (chuck.anderson@colostate.edu). An evaluation episode begins with the pole pointing down, the cart in the center of the track, with both the cart and pole with zero velocity. </td>
   </tr>
   <tr>
      <td> <h4> Agent description </h4> </td>
      <td> <h4> Representative parameters </h4> </td>
      <td> <h4> Mean performance accross thirty agents </h4> </td>
   </tr>
   <tr>
      <td> SGD with feedforward ANN </td>
      <td> Adam with feedforward ANN </td>
   </tr>
   
</table>

## Carpole

# Other stuff:

## How to create package

<source >

python3 setup.py sdist bdist_wheel

python3 -m twine upload dist/*

</source>

# To create a private release

<code>
   $ python3 setup.py sdist bdist_wheel
</code>

