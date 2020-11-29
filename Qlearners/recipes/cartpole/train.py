import argparse , math
import Qlearners.recipes.cartpole.cartpolebox2d as cartpole
import matplotlib.pyplot as plt
import numpy as np
import Qlearners.Qlearner as ql

parser = argparse.ArgumentParser()
parser.add_argument( "-v", action="store_true", default=False, help="not implemented")
parser.add_argument( "--M_H", nargs="+", type=int, default=[20,20])
parser.add_argument( "--scgI", type=int, default=20)
parser.add_argument( "--gamma", type=float, default=0.9)
parser.add_argument( "--rerunNum", type=int)
parser.add_argument( "--numReplays", type=int, default=5)
parser.add_argument( "--episode_len", type=int, default=1000)
parser.add_argument( "--episodes_max", type=int, default=200)
parser.add_argument( "--batch_size", type=int, default=1000)
parser.add_argument( "--graph",action="store_true", default=False)
parser.add_argument( "--saveEvalHist",action="store_true", default=False)
parser.add_argument( "--saveWeightsInterval", type=int, default=0, help="how often to save weights.  zero for don't save. Value of one will save every eval.")
parser.add_argument( "--evalLength" , type=int , default=2000)
parser.add_argument( "--saveDir")
parser.add_argument( "--savePrefix")
args = parser.parse_args()

if args.saveEvalHist:
    saveEvalFile = tempfile.NamedTemporaryFile(mode="w",delete=False,
                                               dir=args.saveDir,
                                               prefix=args.savePrefix,
                                               suffix=".outfile")

actions = ( [-1.0] , [0.0] , [1.0] )

#
# prepare plotting
#

plt.ion(); firstGraph=True;

#
# setup some constant parameters
#

eval_startPos = [0]                  # start eval in middle only
eval_startAngles = [-math.pi,0.0]    # start eval on up and down positions respectively
eval_r_hist = [[] for i in range(len(eval_startPos)*len(eval_startAngles))]
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
        domain.act( selected_action[selected_action_idx][0] )
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


agent = ql.Qlearner( 4 , actions , args.M_H , gamma=args.gamma )

# 
# ####################

# ####################
# loop over episodes
# 

episode_i = 0
while episode_i < args.episodes_max:

    episode = agent.generate_episode( args.episode_len ,
                                      step_f = lambda sa: advance_environment_f(sa , get_domain() ) ,
                                      init_f = reset_domain_f ,
                                      epsilon = 0.1 )
    agent.add_to_memory( episode )
    agent.learn( num_updates=args.numReplays , batch_size=args.batch_size )


    eval_domain = cartpole.CartPole()
    eval_episode = agent.generate_episode( args.evalLength ,
                                           step_f = lambda sa: advance_environment_f( sa , eval_domain ) ,
                                           init_f = lambda: reset_domain_eval_f( eval_domain ) ,
                                           epsilon = None )
    
    eval_sum_r = np.sum( [ ee["r"] for ee in eval_episode ] )

    print( "episode" , episode_i , "eval reward=" , eval_sum_r )

    episode_i += 1
    
# 
# ####################
