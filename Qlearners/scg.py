import sys, math
import torch
from torch import Tensor
from torch.optim import Optimizer

floatPrecision = sys.float_info.epsilon
sigma0 = 1.0e-6
betamin = 1.0e-15
betamax = 1.0e20

def unpack(Veced,unVeced):
    startIdx = 0; stopIdx = 0;
    for oP in unVeced:
        stopIdx = startIdx + torch.numel(oP)
        # print( netp )
        oP.copy_(Veced[startIdx:stopIdx].view(*oP.size()))
        # print( netp )
        startIdx = stopIdx

def pack(unVeced):
    return torch.cat([oP.view(-1) for oP in unVeced])
    
class SCG(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    If you want to resatrt SCG, just create a new SCG instance.



    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v

        where p, g, v and :math:`\rho` denote the parameters, gradient, velocity, and
        momentum respectively.

        This is in constrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
             v = \rho * v + lr * g \\
             p = p - v

        The Nesterov version is analogously modified.
    """

    def __init__(self, params, verbose=False):
        defaults = dict()
        super(SCG, self).__init__(params, defaults)

        self.verbose = verbose
        self.numParams = int(torch.sum(Tensor([torch.numel(par.data) for
                                               par in self.getOptParams()])))
        self.success = True
        self.nsuccess = 0
        self.beta = 1.0e-6
        self.fold = None
        self.fnow = None
        self.gradnew = None
        self.gradold = None
        self.d = None
        self.theta = 0          # will be computed first call to step()
        self.kappa = 0          # will be computed first call to step()


    def __setstate__(self, state):
        super(SCG, self).__setstate__(state)

    def getOptParams(self):
        optParams = []
        for group in self.param_groups:
            for p in group['params']:
                optParams += [p]
        return optParams

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, not optional): A closure that reevaluates the model
                and returns the loss.
        """

        if self.d is None:
            self.fold = closure(grad=True)
            self.fnow = self.fold
            self.gradnew = pack([op.grad.data for op in self.getOptParams()])
            self.gradold = self.gradnew.clone()
            self.d = torch.mul(self.gradold , -1.)
            self.success = True
            self.nsuccess = 0
            self.beta = 1.0e-6

        # vectorize parameters and gradients
        x = pack([op.data for op in self.getOptParams()])
        #??? self.gradnew = pack([op.grad.data for op in self.getOptParams()])

        if self.success:
            self.mu = torch.dot(self.d , self.gradnew)
            if math.isnan(self.mu):
                print( "bad things have happened, mu is NaN." )
            if self.mu >= 0:
                torch.mul(self.gradnew , -1. , out=self.d)
                self.mu = torch.dot(self.d , self.gradnew)
            self.kappa = torch.dot(self.d , self.d)
            if self.kappa < floatPrecision:
                print( "reached limit on machine precision.  quitting" )
                return
            sigma = sigma0 / math.sqrt(self.kappa)
            xplus = torch.add(x , sigma , self.d)
            unpack(xplus,[op.data for op in self.getOptParams()])
            closure(grad=True)  # model holds gplus
            gplus = pack([op.grad.data for op in self.getOptParams()])
            self.theta = torch.dot(self.d , gplus.sub(self.gradnew)) / sigma

        # increase effective curvature and evaluate step size, alpha
        delta = self.theta + self.beta * self.kappa
        if math.isnan(delta): print ("delta is NaN")
        if delta <= 0:
            delta = self.beta * self.kappa
            self.beta = self.beta - self.theta/self.kappa
        # XXX: this was added by chuck?
        # if delta == 0:
        #     alpha = 1
        # else:
        alpha = -self.mu / delta
        
        ## parameter update
        xnew = torch.add(x , alpha , self.d)
        unpack(xnew , [op.data for op in self.getOptParams()])
        fnew = closure(grad=True) # !!!CLOSURE CALL HERE!!!
        ## Calculate the comparison ratio.
        Delta = 2 * (fnew - self.fold) / (alpha * self.mu)
        if not math.isnan(Delta) and Delta  >= 0:
            self.success = True
            self.nsuccess += 1
            x = xnew
            self.fnow = fnew
        else:
            self.success = False
            self.fnow = self.fold

        # if successful in taking a step, prep for a new direction
        if self.success:
            self.fold = fnew         # TODO: this line is suspect, compare to Nabney
            self.gradold = self.gradnew
            # do not need to compute new gradient,
            # model currently holds latest gradients from closure call above
            self.gradnew = pack([op.grad.data for op in self.getOptParams()])
            if torch.dot(self.gradnew , self.gradnew) == 0:
                print( "zero gradient!" )
                return self.success
        
        # record some termination-related variables to return
        # retVals = {"xPrec":alpha*self.d, "fPrecision":fnew-self.fold,
        #            "newGrad":torch.dot(self.gradnew,self.gradnew),
        #            "success":self.success}

        ## Adjust beta according to comparison ratio.
        if math.isnan(Delta) or Delta < 0.25:
            self.beta = min(4.0*self.beta, betamax)
        elif Delta > 0.75:
            self.beta = max(0.5*self.beta, betamin)

        ## Update search direction using Polak-Ribiere formula, or re-start 
        ## in direction of negative gradient after nparams steps.
        if self.nsuccess == self.numParams:
            print( "success equal to number of paramters, re-starting search direction on next call" )
            torch.mul(self.gradnew , -1 , self.d)
            self.nsuccess = 0
        elif self.success:
            # print( "success! updating search direction." )
            gamma = torch.dot(self.gradold.sub(self.gradnew) ,
                              torch.div(self.gradnew , self.mu))
            self.d.mul_(gamma)
            self.d.sub_(self.gradnew)
        else:
            pass
            # print( "no success.  back to the drawing board." )
            
        unpack(x,[op.data for op in self.getOptParams()])
        return self.success
