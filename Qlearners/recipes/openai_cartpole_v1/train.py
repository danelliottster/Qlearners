import gym
import numpy as np
import Qlearners.Simple_ANN as ANN
import Qlearners.Qlearner as ql
import Qlearners.epsilon_decay as epsilon_decay

M_H = [ 20 ]
LR = 0.001
GAMMA = 0.95
NUM_EPISODES = 10000
MAX_EPISODE_LEN = 100
EVAL_EPISODE_LEN = 500
EPSILON_DECAY_RATE = 0.99
MIN_EPSILON = 0.1
NUM_REPLAYS_PER_UPDATE = 1
BATCH_SIZE = 100


domain = gym.make('CartPole-v0')
available_actions = [ [ 0 ] , [ 1 ] ]             # left and right

M_state = len( domain.observation_space.high )
M_act = 1


def advance_environment_f( selected_action_idx , render_p=False ):

    state , reward , finished_p , _ = domain.step( selected_action_idx )
    if render_p :
        domain.render()
    # return ( np.expand_dims( state , axis=0 ) , reward , finished_p )
    return ( np.expand_dims( state , axis=0 ) , reward , False )

def reset_domain_f():

    return np.expand_dims( domain.reset() , axis=0 )

Q_approx = ANN.Net( M_state+M_act , M_H , 1 , LR )

agent = ql.Qlearner( Q_approx , available_actions , M_state , gamma=GAMMA )

epsilon = 1.
for episode_i in range( NUM_EPISODES ) :

    episode = agent.generate_episode( MAX_EPISODE_LEN ,
                                      step_f = advance_environment_f ,
                                      init_f = reset_domain_f ,
                                      epsilon = epsilon )

    agent.add_to_memory( episode )

    agent.learn( num_updates=NUM_REPLAYS_PER_UPDATE , batch_size=BATCH_SIZE )

    epsilon = epsilon_decay.decay( epsilon , EPSILON_DECAY_RATE , MIN_EPSILON )

    eval_episode = agent.generate_episode( EVAL_EPISODE_LEN ,
                                           step_f = lambda x: advance_environment_f(x,render_p=(episode_i%25==0) ) ,
                                           init_f = reset_domain_f ,
                                           epsilon = 0.0 )

    eval_sum_r = np.sum( [ ee["r"] for ee in eval_episode ] )
    print( "episode" , episode_i , "eval reward=" , eval_sum_r , "train ep length=" , len(episode) , "eval length=" , len(eval_episode) )
