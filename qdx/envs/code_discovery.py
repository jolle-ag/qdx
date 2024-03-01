from gymnax.environments import environment, spaces
import jax
import jax.numpy as jnp
from flax import struct
import chex
from inspect import signature
from typing import Tuple, Optional
import itertools
from qdx.simulators import TableauSimulator
import stim
from more_itertools import distinct_permutations
from itertools import combinations
import scipy.special as ss
import numpy as np
from jax import lax


@struct.dataclass
class EnvState:
    """
    This class will contain the state of the environment:

    tableau: binary array of size 2*n_qubits_physical*(n_qubits_physical-n_qubits_logical)
    time: integer from 0 to max_steps
    """
    tableau: jnp.array
    time: int

# We ignore this class   
@struct.dataclass
class EnvParams:
    n: int = 7
    k: int = 1
    d: int = 3
    max_steps_in_episode: int = 20
    
    
class CodeDiscovery(environment.Environment):
    """
    Environment for the automatic discovery of QEC codes and encodings.
    
    Args:
        n_qubits_physical (int): Number of physical qubits available
        n_qubits_logical (int): Number of logical qubits
        code_distance (int): Target code distance
        gates (list(CliffordGates)): List of Clifford gates to prepare the encoding circuit
        graph (list(tuple), optional): Graph of the qubit connectivity. Default: all-to-all qubit connectivity
        max_steps (int, optional): The number of maximum gates to be applied in the circuit. Default: 30
        lbda (float, optional): Global rescaling factor for the instantaneous reward. Default: 100
        pI (float, optional): Probability of no error in the noise channel. Default: 0.9
        softness (int, optional): Parameter that controls the size of the stabilizer subgroup to be generated. Default: 1
    """
    def __init__(self,
            n_qubits_physical,
            n_qubits_logical,
            code_distance,
            gates,
            graph=None,
            max_steps = 30,
            lbda = 100,
            pI=0.9,
            softness=1,
                ):
        super().__init__()
        
        self.n_qubits_physical = n_qubits_physical
        self.n_qubits_logical = n_qubits_logical
        self.gates = gates
        self.max_steps = max_steps
        self.d = code_distance
        self.lbda = lbda # Rescales reward for better convergence
        self.pI = pI # Probability of no error
        
        
        self.graph = graph
        if self.graph is None:
            self.graph = []
            # Fully connected by default
            for ii in range(self.n_qubits_physical):
                for jj in range(ii+1, self.n_qubits_physical):
                    self.graph.append((ii,jj))
                    self.graph.append((jj,ii))
                    

        
        self.obs_shape = (2 * n_qubits_physical * (n_qubits_physical - n_qubits_logical), )
        
        # Initialize action tensor
        self.actions = self.action_matrix()
        
        # Symplectic metric Omega
        self.Omega = jnp.kron(jnp.array([[0,1],[1,0]], dtype=jnp.uint8), jnp.eye(n_qubits_physical, dtype=jnp.uint8))
        
        # Initialize error operators and probabilities
        self.E_mu, self.p_mu = self.error_operators()
        
        # Initialize stabilizer group structure
        self.generate_S_structure(softness) # This generates self.S_struct
        
        # Initialize num_KL
        self.num_KL = len(self.E_mu)


    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters
        return EnvParams()
    
    def generate_S_structure(self, softness):
        # Generate the structure of the stabilizer group (S) based on the softness parameter.
    
        num = self.n_qubits_physical - self.n_qubits_logical
        
        # Calculate the number of stabilizer elements
        soft_elements = int(sum([ss.binom(num,i) for i in range(1,softness+1)]))

        # Create an array of zeros
        S_struct = np.zeros((soft_elements, num), dtype=int)

        # Book-keeping
        start_idx = 0

        for m in range(1,softness+1):

            # Generate combinations of indices to place ones in the S structure
            comb = list(combinations(range(num),m))
            indices = np.array(comb)

            # Fill the S structure with ones at the specified indices
            for i,idx in enumerate(indices):
                S_struct[start_idx+i, idx] = 1

            # Update the start_idx
            start_idx += i+1
            
        # Ensure there are no rows with all zeroes in the S structure
        assert np.prod(np.any(S_struct, axis=1)), "There is a row with all zeroes"
        
        # Convert the S structure to a JAX array for efficient computation
        self.S_struct = jnp.array(S_struct, dtype=jnp.uint8)

        return
    
    def stabilizer_elements(self, tableau):
        # Generate the S matrix by multiplying the S structure with the tableau
        return (self.S_struct @ tableau) % 2

    
    def action_matrix(self,
                      params: Optional[EnvParams] = EnvParams) -> chex.Array:
        
        action_matrix = []
        self.action_string = []
        self.action_string_stim = []

        for gate in self.gates:
            ## One qubit gate
            if len(signature(gate).parameters) == 1:
                for n_qubit in range(self.n_qubits_physical):                    
                    action_matrix.append(gate(n_qubit))
                    self.action_string.append('%s-%d' % (gate.__name__, n_qubit))
                    self.action_string_stim.append('.%s(%d)' % (gate.__name__.lower(), n_qubit))
                    

            ## Two qubit gates
            elif len(signature(gate).parameters) == 2:
                for edge in self.graph:
                    action_matrix.append(gate(edge[0], edge[1]))                    
                    self.action_string.append('%s-%d-%d' % (gate.__name__, edge[0], edge[1]))
                    self.action_string_stim.append('.%s(%d, %d)' % (gate.__name__.lower(), edge[0], edge[1]))

                    
        return jnp.array(action_matrix, dtype=jnp.uint8)
    
    def get_observation(self, tableau):
        '''
        Extract the check matrix for the observation
        '''
        ## Only generators without sign
        check_mat = tableau[self.n_qubits_physical + self.n_qubits_logical:].astype(jnp.uint8)
     
        return check_mat
    
    def error_operators(self, params: Optional[EnvParams] = EnvParams) -> chex.Array:
        
        error_list = [1,2,3] # Corresponds to X,Y,Z
        
        results = []
        
        for weight in range(1, self.d):
            # Generate all combinations of errors with repetition based on the current weight
            prod = list(list(tup) for tup in itertools.combinations_with_replacement(error_list, weight))
            
            # Pad the error list to match the number of physical qubits
            prod = [pr + [0] * (self.n_qubits_physical - weight) for pr in prod]
            
            # Generate distinct permutations of each product to form the error structure
            result = list(list(distinct_permutations(pr, self.n_qubits_physical)) for pr in prod)
            results.append(list(itertools.chain(*result)))
            
        error_structure = list(itertools.chain(*results))
        
        # Convert error structure to a JAX array for efficient computation
        E_mu = jnp.array([jnp.array(stim.PauliString(p).to_numpy()).flatten() for p in error_structure], dtype=jnp.uint8)

        # p_X = p_Y = p_Z by assumption
        p_channel = jnp.array([self.pI] + [(1.-self.pI)/3.]*3, dtype=jnp.float32)

        # Generate p_mu using p_channel and error_structure
        p_mu = jnp.array([jnp.prod(p_channel[jnp.array(error_pauli_character)]) for error_pauli_character in error_structure], dtype=jnp.float32)
        
        return E_mu, p_mu

    
    def check_KL(self, state: EnvState, params: Optional[EnvParams] = EnvParams):
        # Check the Knill-Laflamme conditions for error correction. This is used to reward the agent
        
        # Extract the stabilizer generators
        check_matrix = state.tableau[self.n_qubits_physical + self.n_qubits_logical:]
        
        # Update the stabilizer group S
        S = self.stabilizer_elements(check_matrix)
        
        # Determine if errors are in S by calculating the logical XOR between S and error operators, E_mu
        inS = jax.vmap(jnp.logical_xor, in_axes=(None,0))(S, self.E_mu)
        inS = jnp.prod(jnp.logical_not(inS), axis=-1)
        
        # Calculate the number of Knill-Laflamme conditions that are not satisfied. This is used for stopping criterion
        self.num_KL = len(self.E_mu) - jnp.sum(jnp.any(((self.E_mu @ self.Omega) @ check_matrix.T)%2, axis=1), axis=0) - jnp.sum(inS)
        
        # Return the weighted KL sum rescaled by lbda
        return self.lbda * ( jnp.sum(self.p_mu) - jnp.sum(self.p_mu * jnp.any(((self.E_mu @ self.Omega) @ check_matrix.T)%2, axis=1), axis=0) - jnp.dot(self.p_mu, jnp.sum(inS, axis=-1)) )
    
    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """Performs step transitions in the environment."""
        
        prev_terminal = self.is_terminal(state, params)
        
        # Update state
        state = EnvState( (state.tableau @ self.actions[action]) % 2, state.time + 1)
        
        # Update KLs
        reward = -self.check_KL(state) 

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
        
        tableau = TableauSimulator(self.n_qubits_physical)
        init_state = tableau.current_tableau[0]
        
        state = EnvState(
            tableau = init_state,
            time = 0
        )
        return self.get_obs(state), state

    def get_obs(self, state: EnvState, params: Optional[EnvParams] = EnvParams) -> chex.Array:
        """Applies observation function to state."""
        
        return self.get_observation(state.tableau).flatten()

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal."""
        # Check termination criteria
        done_encoding = self.num_KL == 0 # self.threshold
        
        # Check number of steps in episode termination condition
        done_steps = state.time >= self.max_steps
        
        done = jnp.logical_or(done_encoding, done_steps)
        return done

    @property
    def name(self) -> str:
        """Environment name."""
        return "CodeDiscovery"

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

        return spaces.Box(0, 1, self.obs_shape, dtype=jnp.uint8)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""

        return spaces.Dict(
            {
                "tableau": spaces.Box(0, 1, self.obs_shape, jnp.uint8),
                "time": spaces.Discrete(params.max_steps),
            }
        )
