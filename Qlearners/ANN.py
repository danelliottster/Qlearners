"""
ANN class for Q-function approximation
Has the pieces of more advanced functionality such as the ability to only update certain layers or have a forward model and a Q-function model.
"""

import functools
import torch
import torch.nn as nn
from torch import Tensor, LongTensor, optim
import Qlearners.scg as scg

class Net(nn.Module):
    """
    A totally over-built ANN class.
    """
    def __init__( self, M_I , M_H ,
                  M_H_Q=[] , M_F_O=0 ,
                  alg = 'scg' , alg_params={'scgI':20} ,
                  noQonShared=False ):
        super(Net, self).__init__()
        self.M_I_Q = M_I ; self.M_H = M_H ; self.M_H_Q = M_H_Q ;
        self.Qlayers = []; self.Flayers = [];
        self.noQonShared = noQonShared
        self._alg = alg ; self._alg_params = alg_params ;
        # create shared hidden layers
        self.h0 = nn.Linear(self.M_I_Q,self.M_H[0])
        if not self.noQonShared:
            self.Qlayers += ["h0"]
        self.Flayers += ["h0"]
        for li in range(1,len(self.M_H)):
            attrName = "h"+str(li)
            setattr(self, attrName,
                    nn.Linear(self.M_H[li-1],self.M_H[li]))
            if not self.noQonShared:
                self.Qlayers += [attrName]
            self.Flayers += [attrName]
        # create Q-learning hidden layers
        if self.M_H_Q:
            self.qh0 = nn.Linear(self.M_H[-1],self.M_H_Q[0])
            self.Qlayers += ["qh0"]
            for li in range(1,len(self.M_H_Q)):
                attrName = "qh"+str(li)
                setattr(self, attrName,
                        nn.Linear(self.M_H_Q[li-1],self.M_H_Q[li]))
                self.Qlayers += [attrName]
        # create Q-learning output layer
        if self.M_H_Q:
            self.qout = nn.Linear(self.M_H_Q[-1],1)
        else:
            self.qout = nn.Linear(self.M_H[-1],1)
        self.Qlayers += ["qout"]
        # create forward model output layer
        self.fout = nn.Linear(self.M_H[-1],M_F_O)
        self.Flayers += ["fout"]

        # create optimizer
        self._optimizer = None
        if self._alg == 'adam':
            self._optimizer = torch.optim.Adam( self._Q_parameters() )
        if self._alg == 'sgd':
            self._optimizer = torch.optim.SGD( self._Q_parameters() , lr=self._alg_params['lr'] )
        
    def compute_Q( self , x ):
        x = torch.tensor( x , dtype=torch.float , requires_grad=False )
        Q = self.forward_Q( x , grad_p=False )
        return Q.numpy()

    def forward_Q( self , x , grad_p=True ):
        with torch.set_grad_enabled( grad_p ):
            for li in range(len(self.M_H)):
                layer = getattr(self,"h"+str(li))
                x = torch.tanh(layer(x))
            for li in range(len(self.M_H_Q)):
                layer = getattr(self,"qh"+str(li))
                x = torch.tanh(layer(x))
            x = self.qout(x)
        return x

    # def forward_F(self, x):
    #     for li in range(len(self.M_H)):
    #         layer = getattr(self,"h"+str(li))
    #         x = torch.tanh(layer(x))
    #     x = torch.tanh(self.fout(x))
    #     return x

    def _Q_parameters(self):
        return iter(functools.reduce(lambda x,y: x+y, [list(mod.parameters()) for name,mod in self.named_children() if name in self.Qlayers]))

    def turnOff_H_updates(self):
        if self.M_H:
            for li in range(0,len(self.M_H)):            
                layer = getattr(self,"h"+str(li))
                for param in layer.parameters():
                    param.requires_grad = False


    def update( self , state_action_in , targets_in ):

        # TODO: add code to check inputs


        N_in , M_in = state_action_in.shape
        N_out , M_out = targets_in.shape
        
        if self._alg=='scg':

            self._update_scg( state_action_in , targets_in , self._alg_params['scgI'] )

        elif self._alg in ( 'adam' , 'sgd' ):

            targets_in_tensor = torch.tensor( targets_in , dtype=torch.float )
            tmpOut = self.forward_Q( torch.tensor( state_action_in , dtype=torch.float , requires_grad=True ) , grad_p=True )
            loss_f = torch.nn.MSELoss( reduction='mean' )
            loss = loss_f( tmpOut , targets_in_tensor )
            loss.backward()
            self._optimizer.step()

        else:

            print( "Unknown ANN update algorithm." )
            exit

    def _update_scg( self , state_action_in , targets_in , num_scg_iters):

        #
        # define closure needed by scg
        #
        def closure(grad=False):
            tmpOut = self.forward_Q( torch.tensor( state_action_in , dtype=torch.float , requires_grad=grad ) , grad_p=grad )
            delta = tmpOut.clone()
            delta.sub_( torch.tensor( targets_in , dtype=torch.float ) )
            err = delta.pow(2.)
            err = 0.5 * torch.mean(err)
            if grad:
                delta.div_(tmpOut.size()[0]-1)
                self.zero_grad()
                tmpOut.backward(gradient=delta)
            return err
        #
        # run SCG for the specified number of iterations
        # 
        optimizer = scg.SCG( self._Q_parameters() )
        for scgi in range(num_scg_iters):
            step_p = optimizer.step(closure=closure)
