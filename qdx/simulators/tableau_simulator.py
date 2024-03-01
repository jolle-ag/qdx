import jax.numpy as jnp

class TableauSimulator():

    """
    Very simple tableau simulator of Clifford circuits that does not keep track of phases
    """
    
    def __init__(self,
                 n,
                 batch_size = 1,
                 initial_tableau = None):
        
        self.n = n # Number of qubits
        if initial_tableau is None:
            self.current_tableau = jnp.tile(jnp.eye(2*n), (batch_size, 1, 1)).astype(jnp.uint8) # Initialize the tableau with an empty circuit
        else:
            self.current_tableau = jnp.tile(initial_tableau, (batch_size, 1, 1)).astype(jnp.uint8) # Initialize the tableau with desired tableau

        self.batch_size = batch_size
        
    def h(self,i):
        # Hadamard rule: X -> Z, Z -> X. For qubit i, this means that I swap columns i and i+n
        
        h_operator = jnp.eye(2*self.n, dtype=jnp.uint8)
        
        # Swap columns i and i+n
        temp = h_operator[:,i].copy()
        h_operator = h_operator.at[:,i].set(h_operator[:,i+self.n])
        h_operator = h_operator.at[:,i+self.n].set(temp)
        
        # Update the current tableau (only right-multiplication!)
        self.current_tableau =  (self.current_tableau @ h_operator) % 2
        
    def s(self,i):
        # Phase gate rule: X -> Y, Z -> Z, Y -> -X .

        s_operator = jnp.eye(2*self.n, dtype=jnp.uint8)

        ## Make qubit i into Y 
        s_operator = s_operator.at[i, self.n + i ].set(1)

        # Update the current tableau (only right-multiplication!)
        self.current_tableau =  (self.current_tableau @ s_operator) % 2

    def sqrt_x(self,i):
        # SQRT X gate rule: X -> X, Z -> -Y, Y -> Z 

        sqrt_x_operator = jnp.eye(2*self.n, dtype=jnp.uint8)

        ## Make qubit i into Y 
        sqrt_x_operator = sqrt_x_operator.at[i + self.n, i ].set(1)
        
        # Update the current tableau (only right-multiplication!)
        self.current_tableau =  (self.current_tableau @ sqrt_x_operator) % 2
    
    def cx(self, control, target):
        # CX rule: X(c) -> X(c)X(t), Z(c) -> Z(c), X(t) -> X(t), Z(t) -> Z(c)Z(t)
        # 1: column[t] -> column[c] + column[t]
        # 2: column[c+n] -> column[c+n] + column[t+n]
        
        cx_operator = jnp.eye(2*self.n, dtype=jnp.uint8)
        
        # Transform X(c) -> X(c)X(t)
        cx_operator = cx_operator.at[control, target].set(1)
        
        # Transform Z(t) -> Z(c)Z(t)
        cx_operator = cx_operator.at[target+self.n, control+self.n].set(1)
        
        # Update the current tableau (only right-multiplication!)
        self.current_tableau = (self.current_tableau @ cx_operator)%2

    def cz(self, control, target):
        # CZ rule: X(c) -> X(c)Z(t), Z(c) -> Z(c), X(t) -> Z(c)X(t), Z(t) -> Z(t)
        # 1: column[t] -> column[c] + column[t]
        # 2: column[c+n] -> column[c+n] + column[t+n]
        
        cz_operator = jnp.eye(2*self.n, dtype=jnp.uint8)
        
        # Transform X(c) -> X(c)Z(t)
        cz_operator = cz_operator.at[control, target+self.n ].set(1)
        
        # Transform X(t) -> Z(c)X(t)
        cz_operator = cz_operator.at[target, control+self.n ].set(1)
        
        # Update the current tableau (only right-multiplication!)
        self.current_tableau = (self.current_tableau @ cz_operator)%2

    def sqrt_xx(self, control, target):
        # MS or SQRT_XX rule: X(c) -> X(c), Z(c) -> -Y(c)X(t), X(t) -> X(t), Z(t) -> -X(c)Y(t)

        ms_operator = jnp.eye(2*self.n, dtype=jnp.uint8)
        
        # Transform Z(c) -> Y(c)X(t)        
        ms_operator = ms_operator.at[control + self.n, control ].set(1)
        ms_operator = ms_operator.at[control + self.n, target ].set(1)

        # Transform Z(t) -> X(c)Y(t)
        ms_operator = ms_operator.at[target + self.n, control ].set(1)
        ms_operator = ms_operator.at[target + self.n, target ].set(1)
        
        # Update the current tableau (only right-multiplication!)
        self.current_tableau = (self.current_tableau @ ms_operator)%2


    def __iter__(self):
        '''
        Iterate through all tableau
        '''
        return TableauSimulatorIterator(self)

    def __str__(self):
        '''
        Text representation
        '''
        return str(self.current_tableau)
    
    def __repr__(self):
        '''
        Text representation
        '''
        return str(self.current_tableau)


class TableauSimulatorIterator:
   '''Iterator class'''
   def __init__(self, tableau_simulator):
       self._current_tableau = tableau_simulator.current_tableau
       self._batch_size = tableau_simulator.batch_size
       # member variable to keep track of current index
       self._index = 0

   def __next__(self):
       ''''Returns the next value from team object's lists '''
       if self._index < self._batch_size:
           result = self._current_tableau[self._index]
           self._index +=1
           return result
       # End of Iteration
       raise StopIteration    