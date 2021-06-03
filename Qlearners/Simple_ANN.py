"""
ANN class for simple Q-function approximation
Uses SGD to update the parameters.
"""

from collections import OrderedDict
import torch
import torch.nn as nn
from torch import Tensor, LongTensor, optim

class Net( nn.Module ) :

    def __init__( self , M_I , M_H , M_O , lr , momentum=None) :

        super( Net , self ).__init__()
        self.M_I = M_I
        self.M_H = M_H
        self.M_O = M_O

        hidden_layers = OrderedDict()
        tmp_M_H = [ self.M_I ] + self.M_H
        for mh_i in range(1,len(tmp_M_H)) :
            hidden_layers[ 'layer_h'+str(mh_i) ] = nn.Linear( tmp_M_H[mh_i-1] , tmp_M_H[mh_i] )
            hidden_layers[ 'tanh'+str(mh_i) ] = nn.Tanh()
        self.hidden_layers = nn.Sequential( hidden_layers )
        self.output = nn.Linear( self.M_H[-1] , self.M_O )

        if momentum :
            self._optimizer = optim.SGD( self.parameters() , lr=lr , momentum=momentum )
        else :
            self._optimizer = optim.SGD( self.parameters() , lr=lr )
            
        self._criterion = nn.MSELoss()

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

        self._optimizer.zero_grad()
        X_torch = torch.tensor( X , dtype=torch.float )
        T_torch = torch.tensor( T , dtype=torch.float )
        Y = self.forward( X , True)
        loss = self._criterion( Y , T_torch )
        loss.backward()
        self._optimizer.step()
