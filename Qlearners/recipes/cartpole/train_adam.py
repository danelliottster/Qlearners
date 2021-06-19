from collections import OrderedDict
import torch
import torch.nn as nn
from torch import Tensor, LongTensor, optim

import argparse , math
import gym
import numpy as np
import Qlearners.Qlearner as ql
import Qlearners.epsilon_decay as ep_decay
import Qlearners.recipes.cartpole.cartpolebox2d as cartpole

class Net( nn.Module ) :

    def __init__( self , M_H , lr , adam_epsilon=None ,
                  adam_beta1=None , adam_beta2=None ) :

        super( Net , self ).__init__()
        self.M_I = 5
        self.M_H = M_H
        self.M_O = 1

        hidden_layers = OrderedDict()
        tmp_M_H = [ self.M_I ] + self.M_H
        for mh_i in range(1,len(tmp_M_H)) :
            hidden_layers[ 'layer_h'+str(mh_i) ] = nn.Linear( tmp_M_H[mh_i-1] , tmp_M_H[mh_i] )
            hidden_layers[ 'tanh'+str(mh_i) ] = nn.Tanh()
        self.hidden_layers = nn.Sequential( hidden_layers )
        self.output = nn.Linear( self.M_H[-1] , self.M_O )

        self._optimizer = optim.Adam( self.parameters() , lr=lr ,
                                      betas=[adam_beta1,adam_beta2] ,
                                      eps=adam_epsilon )
        self._criterion = nn.MSELoss( reduction='sum' )

    def forward( self , x , grad_p ) :

        x = torch.tensor( x , dtype=torch.float , requires_grad=False )
        with torch.set_grad_enabled( grad_p ):
            tmp = self.hidden_layers( x )
            Q = self.output( tmp )
        return Q

    def compute_Q( self , x ) :

        Q = self.forward( x , False )
        return Q.numpy()

    def update( self , X , T ) :

        X_torch = torch.tensor( X , dtype=torch.float )
        T_torch = torch.tensor( T , dtype=torch.float )
        Y = self.forward( X , True)
        loss = self._criterion( Y , T_torch )
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

def main( M_H , lr , adam_epsilon , adam_beta1 , adam_beta2 , gamma , numReplays , episodes_max , episode_len , batch_size , evalLength , epsilon , epsilon_decay=1.0 , epsilon_min=0.1 , post_ep_hook=None ) :

    actions = ( [-1.0] , [0.0] , [1.0] )

    # #
    # # prepare plotting
    # #
    # plt.ion(); firstGraph=True;

    #
    # setup some constant parameters
    #

    eval_startPos = [0]                  # start eval in middle only
    eval_startAngles = [-math.pi,0.0]    # start eval on up and down positions respectively
    eval_r_hist = [ [] for i in range( len(eval_startPos)*len(eval_startAngles) ) ]
    actionDuration = 2

    #
    # method to scale the function appoximator inputs
    # 
    def scaleState(tmpI):
        tmpI[:,0] = tmpI[:,0] / scaleState.positionRange[1]     
        tmpI[:,1] = tmpI[:,1] / scaleState.velocityRange[1]     
        tmpI[:,2] = tmpI[:,2] / scaleState.angleRange[1]
        tmpI[:,3] = tmpI[:,3] / scaleState.angleVelocityRange[1]
        return(tmpI)
    scaleState.positionRange = (-2.2,2.2)
    scaleState.velocityRange = (-6.,6.)
    scaleState.angleVelocityRange = (-14.,14.)
    scaleState.angleRange = (-math.pi,math.pi)

    #
    # method (and helper method) to generate 
    # 
    def reinforcement(angle):
        angle = abs(angle)
        if angle > math.pi * 0.75:
            return -1
        elif angle < math.pi * 0.25:
            return 1
        else:
            return 0

    # ####################
    # initialize cartpole environment
    #

    def get_domain():
        return get_domain.cpdomain
    get_domain.cpdomain = cartpole.CartPole()

    #
    # ####################

    # ####################
    # create functions needed by Qlearner
    # initialize Q-learning agent
    #

    def advance_environment_f( selected_action_idx , domain ):

        # move simulation forward, get new state, scale state, return scaled state
        for ai in range(actionDuration):
            domain.act( actions[selected_action_idx][0] )
        (x,xdot,a,adot) = domain.sense()
        # compute reward
        reward = reinforcement( a )
        # scale state
        state_vec = np.array( [ [x,xdot,a,adot] ] )
        scaleState( state_vec )
        # return scaled state and reward and indicate not done (sim always not done)
        return ( state_vec , reward , False )

    def reset_domain_f():

        # doesn't actually reset: just picks up where previous episode left off
        # if want to start from same position, need to create a new instance of simulation

        domain = get_domain()
        (x,xdot,a,adot) = domain.sense()
        state_vec = np.array( [ [x,xdot,a,adot] ] )
        scaleState( state_vec )
        return state_vec


    def reset_domain_eval_f( domain ):

        (x,xdot,a,adot) = domain.sense()
        state_vec = np.array( [ [x,xdot,a,adot] ] )
        scaleState( state_vec )
        return state_vec


    Q_approx = Net(  M_H , lr , adam_epsilon , adam_beta1 , adam_beta2 )
    agent = ql.Qlearner( Q_approx , actions , 4 , gamma=gamma )

    # 
    # ####################

    # ####################
    # loop over episodes
    # 

    episode_i = 0 ; train_r_hist = [] ; eval_r_hist = [] ;
    while episode_i < episodes_max:

        print( 'epsilon='  , epsilon )
        episode = agent.generate_episode( episode_len ,
                                          step_f = lambda sa: advance_environment_f(sa , get_domain() ) ,
                                          init_f = reset_domain_f ,
                                          epsilon = 0.1 )
        epsilon = ep_decay.decay( epsilon , epsilon_decay , epsilon_min )
        agent.add_to_memory( episode )
        agent.learn( num_updates=numReplays , batch_size=batch_size )


        eval_domain = cartpole.CartPole()
        eval_episode = agent.generate_episode( evalLength ,
                                               step_f = lambda sa: advance_environment_f( sa , eval_domain ) ,
                                               init_f = lambda: reset_domain_eval_f( eval_domain ) ,
                                               epsilon = None )

        eval_sum_r = np.sum( [ ee["r"] for ee in eval_episode ] )
        train_sum_r = np.sum( [ te["r"] for te in episode ] )

        print( "episode" , episode_i , "eval reward=" , eval_sum_r )

        train_r_hist += np.sum( [ ee["r"] for ee in episode ] )
        eval_r_hist += np.sum( [ ee["r"] for ee in episode ] )

        # execute the post-episode hook
        if post_ep_hook:
            post_ep_hook( locals() )

        # the episode has concluded
        episode_i += 1
    
    # 
    # ####################

    return eval_r_hist , train_r_hist


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument( "-v", action="store_true", default=False, help="not implemented")
    parser.add_argument( "--alg" , default="scg" )
    parser.add_argument( "--M_H", nargs="+", type=int, default=[20,20])
    parser.add_argument( "--lr", type=float, default=0.0001)
    parser.add_argument( "--beta1", type=float, default=0.9)
    parser.add_argument( "--beta2", type=float, default=0.999)
    parser.add_argument( "--adam_eps", type=float, default=pow(10,-8) )
    parser.add_argument( "--epsilon", type=float, default=0.1 )
    parser.add_argument( "--epsilon_decay", type=float, default=1. ) # no decay
    parser.add_argument( "--epsilon_min", type=float, default=0.1 )
    parser.add_argument( "--gamma", type=float, default=0.9)
    parser.add_argument( "--numReplays", type=int, default=5)
    parser.add_argument( "--episode_len", type=int, default=1000)
    parser.add_argument( "--episodes_max", type=int, default=200)
    parser.add_argument( "--batch_size", type=int, default=1000)
    parser.add_argument( "--graph",action="store_true", default=False)
    parser.add_argument( "--evalLength" , type=int , default=2000)
    parser.add_argument( "--saveEvalHist",action="store_true", default=False)
    parser.add_argument( "--saveDir")
    parser.add_argument( "--savePrefix")
    args = parser.parse_args()

    main( M_H=args.M_H , lr=args.lr ,
          adam_epsilon=args.adam_eps , adam_beta1=args.beta1 , adam_beta2=args.beta2 ,
          gamma=args.gamma ,
          numReplays=args.numReplays , episodes_max=args.episodes_max , episode_len=args.episode_len ,
          batch_size=args.batch_size , evalLength=args.evalLength ,
          epsilon=args.epsilon , epsilon_decay=args.epsilon_decay , epsilon_min=0.1 )

    if args.saveEvalHist:
        saveEvalFile = tempfile.NamedTemporaryFile(mode="w",delete=False,
                                                   dir=args.saveDir,
                                                   prefix=args.savePrefix,
                                                   suffix=".outfile")

