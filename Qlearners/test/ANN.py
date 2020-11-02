import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import Qlearners.ANN as ANN

# test_ANN_nohidden = ANN.Net( 5 , [] , [] )

test_ANN_sine = ANN.Net( 1 , [50,10] )

sine_x = np.expand_dims( np.arange( 0 , 4 , step=0.1 ) , axis=1 )
sine_y = np.sin( sine_x )
pred_y = test_ANN_sine.compute_Q( sine_x )

fig = plt.figure()
title = fig.suptitle("blah")
ax = fig.add_subplot(1,1,1)
ax.plot( sine_x , sine_y , "kx" )
pred_y_line, = ax.plot( sine_x[:] , pred_y[:] , "rx" )


def init():
    ax.plot( sine_x , sine_y , "kx" )
    pred_y_line.set_data( [] , [] )
    title.set_text( "blah" )
    return (pred_y_line , title )

def animate(i):
    test_ANN_sine.update( sine_x , sine_y )
    pred_y_more = test_ANN_sine.compute_Q( sine_x )
    pred_y_line.set_data( sine_x , pred_y_more[:] )
    title.set_text( "epoch " + str(i) )
    return (pred_y_line , title )

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=200, interval=1000, blit=False),
                

plt.show()
