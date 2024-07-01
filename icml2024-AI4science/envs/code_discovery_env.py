import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from gymnax.environments import environment, spaces
from typing import Tuple, Optional
import chex
from flax import struct
import itertools
import sys
sys.path.append("..") # Adds higher directory to python modules path.
from simulators import TableauSimulatorCSS
from inspect import signature
from itertools import combinations
import scipy.special as ss


@struct.dataclass
class EnvState:
    """
    This class will contain the state of the environment:

    H_X: binary array of size n_qubits_physical*num_H (Hadamards)
    H_Z: binary array of size n_qubits_physical*(n_qubits_logical - num_H)
    time: integer from 0 to max_steps
    """
    S_X: jnp.array
    S_Z: jnp.array
    time: int

# We ignore this class   
@struct.dataclass
class EnvParams:
    n: int = 7
    k: int = 1
    d: int = 3
    max_steps_in_episode: int = 20
    
    
class CodeDiscoveryCSS(environment.Environment):
    """
    Environment for the automatic discovery of QEC CSS codes and encodings.
    
    Args:
        n_qubits_physical (int): Number of physical qubits available
        n_qubits_logical (int): Number of logical qubits
        code_distance (int): Target code distance
        init_H (list(int)): Positions of initial Hadamard gates
        gates (list(CliffordGates)): List of Clifford gates to prepare the encoding circuit
        graph (list(tuple), optional): Graph of the qubit connectivity. Default: all-to-all qubit connectivity
        max_steps (int, optional): The number of maximum gates to be applied in the circuit. Default: 30
        lbda (float, optional): Global rescaling factor for the instantaneous reward. Default: 100
        pI (float, optional): Probability of no error in the noise channel. Default: 0.9
        softness (int, optional): Parameter that controls the size of the stabilizer subgroup to be generated. Default: 1
        bell (bool, optional): Whether we initialize with Bell states or not. Default: False
    """

    def __init__(self,
            n_qubits_physical,
            n_qubits_logical,
            code_distance,
            init_H,
            gates,
            graph=None,
            max_steps = 30,
            lbda = 100,
            pI=0.9,
            softness=1,
            bell = False
                ):
        super().__init__()
        
        self.n_qubits_physical = n_qubits_physical
        self.n_qubits_logical = n_qubits_logical
        self.gates = gates
        self.max_steps = max_steps
        self.d = code_distance
        self.lbda = lbda # Rescales reward for better convergence
        self.pI = pI # Probability of no error
        self.bell = bell
        
        
        self.graph = graph
        if self.graph is None:
            self.graph = []
            ## Fully connected by default
            for ii in range(self.n_qubits_physical):
                for jj in range(ii+1, self.n_qubits_physical):
                    self.graph.append((ii,jj))
                    self.graph.append((jj,ii))
                    
        
        self.obs_shape = (n_qubits_physical * (n_qubits_physical - n_qubits_logical), )
        
        # Initialize action tensor
        self.actions = self.action_matrix()
        
        self.init_H = init_H # Qubit labels where I place an initial H gate
        self.num_H = len(init_H)
        
        # Initialize error operators
        self.E_mu = self.CSS_error_operators()

        
        # Initialize error probabilities (rescale with sqrt for improved convergence)
        self.p_mu = jnp.sqrt(self.CSS_probabilities())
        
        # Initialize stabilizer group structure
        self.generate_S_structure(softness) # This generates self.S_struct_X and self.S_struct_Z
        
        # Initialize num_KL
        self.num_KL = 2 * len(self.p_mu)


    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters UNUSED
        return EnvParams()
    
    
    def S_struct_loop(self, softness, num):
         # Generate the structure of the stabilizer group (S) based on the softness parameter 
         # and the num of generators participating (will generate one struct for X and another for Z)
        
        # Calculate the number of stabilizer elements
        soft_elements = int(sum([ss.binom(num,i) for i in range(1,softness+1)]))
        
        # Create an array of zeros
        S_struct = np.zeros((soft_elements, num), dtype=int)

        # Book-keeping
        start_idx = 0

        for m in range(1,softness+1):

            # Update the combinations of positions to place ones in the S structure
            comb = list(combinations(range(num),m))
            indices = np.array(comb)

            # Fill the S structure with ones at the specified indices
            for i,idx in enumerate(indices):
                S_struct[start_idx+i, idx] = 1

            # Update the start_idx
            start_idx += i+1

        # Ensure there are no rows with all zeroes in the S structure
        assert np.prod(np.any(S_struct, axis=1)), "There is a row with all zeroes"
        
        return S_struct
    
    def generate_S_structure(self, softness):
        
        # Helper function that gives the structure of the stabilizer group
        # It's only called once at the very beginning
        num_X = self.num_H
        num_Z = self.n_qubits_physical - self.n_qubits_logical - self.num_H
        
        # Convert the S structure to a JAX array for efficient computation
        self.S_struct_X = jnp.array(self.S_struct_loop(softness, num_X), dtype=jnp.uint8)
        self.S_struct_Z = jnp.array(self.S_struct_loop(softness, num_Z), dtype=jnp.uint8)

        return
    
    def stabilizer_elements(self, tableau, S_struct):
        # Generate the S matrix by multiplying the S structure with the tableau
        return (S_struct @ tableau) % 2

    
    def action_matrix(self,
                      params: Optional[EnvParams] = EnvParams) -> chex.Array:
        
        action_matrix = []
        self.action_string = []
        self.action_string_stim = []
        z_action_move = []

        for gate in self.gates:
            ## Two qubit gates. There should not be one-qubit gates
            if len(signature(gate).parameters) == 2:
                for edge in self.graph:
                    action_matrix.append(gate(edge[0], edge[1]))                    
                    self.action_string.append('%s-%d-%d' % (gate.__name__, edge[0], edge[1]))
                    self.action_string_stim.append('.%s(%d, %d)' % (gate.__name__.lower(), edge[0], edge[1]))
                    if edge[0] < edge[1]:
                        z_action_move.append(1)
                    else:
                        z_action_move.append(-1)

        self.z_action_move = jnp.array(z_action_move, dtype=jnp.int8)
                    
        return jnp.array(action_matrix, dtype=jnp.uint8)
    
    def get_observation(self, H_X, H_Z):
        '''
        Extract the check matrix for the observation
        '''
        # Only generators without sign
        return jnp.concatenate([H_X, H_Z]).flatten()


    def CSS_error_operators(self) -> chex.Array:
        # We assume same X and Z-type errors to be detected

        size = int(sum([ss.binom(self.n_qubits_physical, w) for w in range(1, self.d)]))

        E_mu = np.zeros((size, self.n_qubits_physical), dtype=np.int8)

        # Book-keeping
        start_idx = 0

        for w in range(1, self.d):

            # Update the combinations of positions to place the ones
            comb = list(combinations(range(self.n_qubits_physical),w))

            # Convert combinations to a NumPy array
            indices = np.array(comb)

            # Use a loop to set ones at the specified positions
            for i,idx in enumerate(indices):
                E_mu[start_idx+i, idx] = 1
                

            # Update the start_idx
            start_idx += i+1
            
        return E_mu
    
    def CSS_probabilities(self) -> chex.Array:
    
        p = jnp.array([self.pI]+[(1-self.pI)/3.]*3)
        
        p_mu = p[1] ** jnp.sum(self.E_mu, axis=1)
        
        return p_mu

    def check_KL(self, state: EnvState, params: Optional[EnvParams] = EnvParams):
        # Check the Knill-Laflamme conditions for error correction. This is used to reward the agent

        anticomm_X = jnp.any((self.E_mu @ state.S_X.T)%2, axis=1)
        anticomm_Z = jnp.any((self.E_mu @ state.S_Z.T)%2, axis=1)

        size = int(ss.binom(self.n_qubits_physical, self.d - 1))

        # Determine if errors are in S by calculating the logical XOR between S and error operators, E_mu
        inS_X = jax.vmap(jnp.logical_xor, in_axes=(None,0))(state.S_X, self.E_mu[-size:])
        inS_X = jnp.prod(jnp.logical_not(inS_X), axis=-1)
        
        inS_Z = jax.vmap(jnp.logical_xor, in_axes=(None,0))(state.S_Z, self.E_mu[-size:])
        inS_Z = jnp.prod(jnp.logical_not(inS_Z), axis=-1)
        
        # Calculate the number of Knill-Laflamme conditions that are not satisfied. This is used for stopping criterion
        self.num_KL = 2*len(self.p_mu) - jnp.sum(anticomm_X, axis=0) - jnp.sum(anticomm_Z, axis=0) - jnp.sum(inS_X) - jnp.sum(inS_Z)
        
        # Reward is based on the weighted KL sum
        rew = 2 * jnp.sum(self.p_mu) - jnp.sum(self.p_mu * anticomm_X, axis=0) - jnp.sum(self.p_mu * anticomm_Z, axis=0) 
        -jnp.dot(self.p_mu[-size:], jnp.sum(inS_X, axis=-1)) -jnp.dot(self.p_mu[-size:], jnp.sum(inS_Z, axis=-1))

        # Return the weighted KL sum rescaled by lbda
        return self.lbda * (rew)

    
    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """Performs step transitions in the environment."""
        
        # prev_terminal = self.is_terminal(state, params)
        x_label = action
        z_label = action + self.z_action_move[action]
        
        # Update state
        state = EnvState( (state.S_X @ self.actions[x_label]) % 2, (state.S_Z @ self.actions[z_label]) % 2, state.time + 1)
        
        # Update KLs
        reward = -self.check_KL(state)

        # Reward extra if encoding was successful
        reward += 1 * (abs(self.num_KL) <= 1e-7)

        # Evaluate termination conditions
        done = self.is_terminal(state, params)

        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            reward,
            done,
            {"discount": self.discount(state, params)},
        )

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""
        
        tableau = TableauSimulatorCSS(self.n_qubits_physical, self.n_qubits_logical, self.init_H, bell=self.bell)
        S_X = self.stabilizer_elements(tableau.H_X, self.S_struct_X)
        S_Z = self.stabilizer_elements(tableau.H_Z, self.S_struct_Z)

        state = EnvState(
            S_X = S_X,
            S_Z = S_Z,
            time = 0
        )
        return self.get_obs(state), state

    def get_obs(self, state: EnvState, params: Optional[EnvParams] = EnvParams) -> chex.Array:
        """Applies observation function to state."""
        H_X = state.S_X[:self.num_H]
        H_Z = state.S_Z[:self.n_qubits_physical - self.n_qubits_logical - self.num_H]
        
        return self.get_observation(H_X, H_Z).flatten() 

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal."""
        # Check termination criteria
        done_encoding = (abs(self.num_KL) <= 1e-6)
        
        # Check number of steps in episode termination condition
        done_steps = state.time >= self.max_steps
        
        done = jnp.logical_or(done_encoding, done_steps)
        return done

    @property
    def name(self) -> str:
        """Environment name."""
        return "CodeDiscoveryCSS"

    @property
    def num_actions(self, params: Optional[EnvParams] = EnvParams) -> int:
        """Number of actions possible in environment."""
        return self.actions.shape[0]

    def action_space(
        self, params: Optional[EnvParams] = EnvParams
    ) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(self.num_actions)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""

        return spaces.Box(0, 1, self.obs_shape, dtype=jnp.float16)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""

        return spaces.Dict(
            {
                "tableau": spaces.Box(0, 1, self.obs_shape, jnp.float16),
                "time": spaces.Discrete(params.max_steps),
            }
        )
