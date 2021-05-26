# TODO: remove num_updates parameter from learn method?
# TODO: add ability to send a list of parameters to the Q-func approximator via the learn method

import random, tempfile, uuid
import numpy as np

class Qlearner:
    """
    The plan is to make this class increasingly abstract as we add more varieties of Q-learning
    """

    def __init__( self , Q_approx , actions , M_I_state , gamma=0.9):
        
        self._replay_memory = []
        self._actions = actions

        self._gamma = gamma

        self._M_I_state = M_I_state
        self._M_I_actions = len(self._actions[0])
        self._M_I = self._M_I_state + self._M_I_actions

        self._Q_approx = Q_approx

    def select_action( self , state_in , epsilon=None ):

        if not epsilon:
            epsilon = 0.0
        if random.random() > epsilon:

            # compute q-value for each action
            qvals = np.zeros( len(self._actions) )
            net_in = np.append( state_in , np.zeros( ( 1 , self._M_I_actions ) ) , axis=1 ) # add elements for action
            for ai,a in enumerate( self._actions ):
                net_in[ 0, self._M_I_state: ] = self._actions[ ai ]
                Qout = self._Q_approx.compute_Q( net_in )
                # store Q-value
                qvals[ai] = Qout[0,0]
            # select the winning action
            selected_action_i = np.argmax( qvals )

        else:
            selected_action_i = random.sample( range(len(self._actions)) , k=1 )[0]
            
        return self._actions[ selected_action_i ] , selected_action_i

    def generate_episode( self , episode_len , step_f , init_f , epsilon=None ):

        episode = []

        # get initial state for episode
        current_state = init_f()

        # loop over episode time steps
        for episode_t in range(episode_len):

            # select action
            current_action , current_action_idx = self.select_action( current_state , epsilon=epsilon )
            # advance simulation
            (next_state , current_reward , finished_p) = step_f( current_action_idx )
            # add to episode memory
            episode += [ {"s":current_state , "ai":current_action_idx , "r":current_reward , "snext":next_state } ]
            # advance to the next state
            current_state = next_state
            # exit if episode is finished
            if finished_p:
                break

        return episode

    def add_to_memory( self , episode ):
        # add to memory
        self._replay_memory += episode

    def build_training_data( self , batch_size ):
        """
        select memories
        put states into a training input numpy ndarray
        compute target values and store in a ndarray
        return both ndarrays
        """
        # 
        # input data
        #
        batch_size = min( batch_size , len( self._replay_memory ) )
        input_data = np.zeros( (batch_size , self._M_I ) )
        # store the states in the inputs
        selected_memories = random.sample( self._replay_memory , batch_size )
        input_data[ : , :self._M_I_state ] = np.concatenate( [ sm["s"] for sm in selected_memories ] , axis=0 )
        # actions make up the rest of the inputs
        input_data[ : , self._M_I_state: ] = np.stack( [ self._actions[ sm["ai"] ] for sm in selected_memories ] , axis=0 )
        
        # 
        # compute the target values for each selected memory
        # start with computing max Q(snext,a) for all actions
        #

        # DLE: shouldn't this be the next state?
        # input_data_Qnext = np.copy( input_data )

        input_data_Qnext = np.zeros_like( input_data )
        input_data_Qnext[ : , :self._M_I_state ] = np.concatenate( [ sm["snext"] for sm in selected_memories ] , axis=0 )
        Qvals_next = np.zeros( ( batch_size , len(self._actions) ) )
        for ai in range( len(self._actions) ):
            input_data_Qnext[ : , self._M_I_state: ] = np.repeat( np.expand_dims( np.array( self._actions[ai] ) , axis=0 ) , repeats=batch_size , axis=0 )
            Qvals_next[ : , ai:ai+1 ] = self._Q_approx.compute_Q( input_data_Qnext )

        max_Qvals_next = np.expand_dims( np.max( Qvals_next , axis = 1 ) , axis = 1 )

        current_rs = np.expand_dims( np.array( [ sm["r"] for sm in selected_memories ] ) , axis = 1 )

        T = current_rs + self._gamma * max_Qvals_next

        #
        # done.  return input and target
        # 
        return input_data , T

        
    def learn( self , num_updates , batch_size ):

        for update_i in range(num_updates):
            #
            # select experiences from memory to update ANN
            # 
            X , T = self.build_training_data( batch_size )

            #
            # perform parameter update
            #
            self._Q_approx.update( X , T )
