import argparse , math
import gym
import matplotlib.pyplot as plt
import numpy as np
import Qlearners.Qlearner as ql
import Qlearners.epsilon_decay as epsilon_decay

def main( args ):

    if args.saveEvalHist:
        saveEvalFile = tempfile.NamedTemporaryFile(mode="w",delete=False,
                                                   dir=args.saveDir,
                                                   prefix=args.savePrefix,
                                                   suffix=".outfile")

    # 
    # Create a training and eval environment
    # 
    domain = gym.make( "LunarLander-v2" )
    available_actions = [ [ 0 , 0 , 0 ] ,
                          [ 1 , 0 , 0 ] ,
                          [ 0 , 1 , 0 ] ,
                          [ 0 , 0 , 1 ] ] #  noop, fire left, fire main, fire right

    # ####################
    # create functions needed by Qlearner
    # initialize Q-learning agent
    #

    def advance_environment_f( selected_action_idx ):

        state , reward , finished_p , _ = domain.step( selected_action_idx )
        return ( np.expand_dims( state , axis=0 ) , reward , finished_p )

    def reset_domain_f():

        return np.expand_dims( domain.reset() , axis=0 )

    Q_approx = ANN.Net( 8 , args.M_H , [] , alg='scg' , alg_params={'scgI':20} )

    agent = ql.Qlearner( Q_approx , available_actions , gamma=args.gamma )

    #
    # ####################

    # ####################
    #  loop over episodes
    #

    episode_i = 0
    epsilon = args.epsilon
    while episode_i < args.episodes_max:

        epsilon = epsilon_decay( epsilon , args.epsilon_decay_rate , args.min_epsilon )

        episode = agent.generate_episode( args.episode_max_len ,
                                          step_f = advance_environment_f ,
                                          init_f = reset_domain_f ,
                                          epsilon = epsilon )

        agent.add_to_memory( episode )

        agent.learn( num_updates=args.numReplays , batch_size=args.batch_size )

        eval_episode = agent.generate_episode( args.episode_max_len ,
                                               step_f = advance_environment_f ,
                                               init_f = reset_domain_f ,
                                               epsilon = 0.0 )

        eval_sum_r = np.sum( [ ee["r"] for ee in eval_episode ] )
        print( "episode" , episode_i , "eval reward=" , eval_sum_r , "eval length=" , len(eval_episode) )

        episode_i += 1
    #
    # ####################

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument( "-v", action="store_true", default=False, help="not implemented" )
    parser.add_argument( "--M_H", nargs="+", type=int, default=[512,256] )
    parser.add_argument( "--scgI", type=int, default=20 )
    parser.add_argument( "--gamma", type=float, default=0.9 )
    parser.add_argument( "--rerunNum", type=int )
    parser.add_argument( "--numReplays", type=int, default=5 )
    parser.add_argument( "--episode_max_len", type=int, default=5000 )
    parser.add_argument( "--episodes_max", type=int, default=1000 )
    parser.add_argument( "--batch_size", type=int , default=1000 )
    parser.add_argument( '--epsilon' , type=float , default = 0.1 )
    parser.add_argument( '--epsilon_decay_rate' , type=float , default = 0.99 )
    parser.add_argument( '--min_epsilon' , type=float , default = 0.1 )
    parser.add_argument( "--graph",action="store_true", default=False )
    parser.add_argument( "--saveEvalHist",action="store_true", default=False )
    parser.add_argument( "--saveWeightsInterval", type=int, default=0, help="how often to save weights.  zero for don't save. Value of one will save every eval." )
    parser.add_argument( "--evalLength" , type=int , default=2000 )
    parser.add_argument( "--saveDir" )
    parser.add_argument( "--savePrefix" )
    args = parser.parse_args()

    main( args )
