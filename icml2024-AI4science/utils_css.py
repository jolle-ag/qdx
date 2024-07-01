import numpy as np
import stim
import itertools 
import json
import random

import scipy.special as ss
import jax
import jax.numpy as jnp
import chex
from itertools import combinations


class UtilsCSS():
    
    def __init__(self,
                 n,
                 k,
                 tableau,
                 softness,
                ):
        
        self.n_qubits_physical = n # Number of physical qubits
        self.n_qubits_logical = k # Number of logical qubits
        self.num_H = tableau.H_X.shape[0]
        
        self.tableau = tableau
        
        self.generate_S_structure(softness) # This generates self.S_struct_X and self.S_struct_Z

    def CSS_error_operators(self, eval_d) -> chex.Array:
        
        size = int(sum([ss.binom(self.n_qubits_physical, w) for w in range(1, eval_d+1)]))

        E_mu = np.zeros((size, self.n_qubits_physical), dtype=np.int8)

        # Book-keeping
        start_idx = 0

        for w in range(1, eval_d+1):

            # Update the combinations of positions to place the ones
            comb = list(combinations(range(self.n_qubits_physical),w))

            # Convert combinations to a NumPy array
            indices = np.array(comb)

            # print(comb)

            # Use a loop to set ones at the specified positions
            for i,idx in enumerate(indices):
                # E_X[start_idx+i, idx] = 1
                E_mu[start_idx+i, idx] = 1
                

            # Update the start_idx
            start_idx += i+1
            
        return E_mu

    def S_struct_loop(self, softness, num):
        
        soft_elements = int(sum([ss.binom(num,i) for i in range(1,softness+1)]))
        
        # Create an array of zeros
        S_struct = np.zeros((soft_elements, num), dtype=int)

        # Book-keeping
        start_idx = 0

        for m in range(1,softness+1):

            # Update the combinations of positions to place the ones
            comb = list(combinations(range(num),m))

            # Convert combinations to a NumPy array
            indices = np.array(comb)

            # Use a loop to set ones at the specified positions
            for i,idx in enumerate(indices):
                S_struct[start_idx+i, idx] = 1

            # Update the start_idx
            start_idx += i+1

        assert np.prod(np.any(S_struct, axis=1)), "There is a row with all zeroes"
        
        return S_struct
    
    def generate_S_structure(self, softness):
        
        # Helper function that gives the structure of the stabilizer group
        # It's only called once at the very beginning
    
        num_X = self.num_H
        num_Z = self.n_qubits_physical - self.n_qubits_logical - self.num_H
        
        self.S_struct_X = jnp.array(self.S_struct_loop(softness, num_X), dtype=jnp.uint8)
        self.S_struct_Z = jnp.array(self.S_struct_loop(softness, num_Z), dtype=jnp.uint8)

        return
    
    def generate_S(self, tableau, S_struct):
        
        return (S_struct @ tableau) % 2
    
    
    def check_KL(self, E_mu):
        
        S_X = self.generate_S(self.tableau.H_X, self.S_struct_X)
        S_Z = self.generate_S(self.tableau.H_Z, self.S_struct_Z)

        
        inS_X = jax.vmap(jnp.logical_xor, in_axes=(None,0))(S_X, E_mu)
        inS_X = jnp.prod(jnp.logical_not(inS_X), axis=-1)
        
        inS_Z = jax.vmap(jnp.logical_xor, in_axes=(None,0))(S_Z, E_mu)
        inS_Z = jnp.prod(jnp.logical_not(inS_Z), axis=-1)
        
        
        num_KL = 2*len(E_mu) - jnp.sum(jnp.any((E_mu @ self.tableau.H_X.T)%2, axis=1), axis=0) - jnp.sum(jnp.any((E_mu @ self.tableau.H_Z.T)%2, axis=1), axis=0) -jnp.sum(inS_X) -jnp.sum(inS_Z)
        
        return num_KL
