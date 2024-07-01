import jax.numpy as jnp

class TableauSimulatorCSS():

    """
    Very simple tableau simulator of CSS circuits that does not keep track of phases
    """
    
    def __init__(self,
                 n,
                 k,
                 init_H = [],
                 bell = False):
        
        self.n = n # Number of qubits
        self.k = k # Number of logicals
        self.n_H = len(init_H) # Number of initial Hadamards

        self.H_X = jnp.zeros((self.n_H, self.n), dtype=jnp.uint8) # shape (n_H, n)
        self.H_Z = jnp.zeros((self.n - self.k - self.n_H, self.n), dtype=jnp.uint8) # shape (n-k-n_H, n)
        
        for count,H_label in enumerate(init_H):
            self.H_X = self.H_X.at[count, H_label].set(1)
            
        no_H_positions = set(range(k,n)) ^ set(init_H) # XOR between these two sets
        
        for count,no_H_label in enumerate(no_H_positions):
            self.H_Z = self.H_Z.at[count, no_H_label].set(1)
            
        if bell:
            for h_qubit in init_H:
                self.cx(h_qubit, h_qubit+1)
            
  
    def cx(self, control, target):
        # CX rule: X(c) -> X(c)X(t), Z(c) -> Z(c), X(t) -> X(t), Z(t) -> Z(c)Z(t)
        # 1: column[t] -> column[c] + column[t]
        # 2: column[c+n] -> column[c+n] + column[t+n]
        
        cx_operator = jnp.eye(self.n, dtype=jnp.uint8)
        
        # Transform X(c) -> X(c)X(t)
        cx_operator = cx_operator.at[control, target].set(1)
        
        # Transform Z(t) -> Z(c)Z(t) by transposing cx_operator (see below)
        
        # Update the tableau (only right-multiplication!)
        self.H_X = (self.H_X @ cx_operator)%2
        self.H_Z = (self.H_Z @ cx_operator.T)%2
