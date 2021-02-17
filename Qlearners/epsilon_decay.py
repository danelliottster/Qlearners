def decay( epsilon_in , decay_rate=0.99 , min_epsilon=0.1 ):

    return max( epsilon_in * decay_rate , min_epsilon )

