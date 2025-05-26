from gymnax.environments import environment, spaces
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
import chex
from inspect import signature
from typing import Tuple, Optional
import itertools
from qdx.simulators import TableauSimulator
import stim
from more_itertools import distinct_permutations
from itertools import combinations
import random
import scipy.special as ss
from jax import lax


@struct.dataclass
class EnvState:
    """
    This class will contain the state of the environment:

    tableau: binary array of size 2*n_qubits_physical*(n_qubits_physical-n_qubits_logical)
    time: integer from 0 to max_steps
    cZ: bias parameter
    p_mu: probabilities of all errors considered
    """
    tableau: jax.Array
    time: int
    cZ: float
    p_mu: jax.Array

@struct.dataclass
class NoiseParams:
    """
    This class contains the possible values for cZ, pI, pX.

    It's used in reset_env() to update p_mu
    """
    # This weird way of defining the arrays is a workaround to https://github.com/jax-ml/jax/issues/14295
    cZs: int = struct.field(default_factory=lambda: jnp.array(np.arange(0.5,2.1,0.1)))
    pIs: int = struct.field(default_factory=lambda: jnp.array([0.6, 0.75, 0.9, 0.95, 0.99, 0.999]))
    pXs: int = struct.field(default_factory=lambda: jnp.array([[0.0688262, 0.0855959, 0.100139, 0.112763,
                            0.123748, 0.133333, 0.141718, 0.149065,
                            0.15551, 0.16117, 0.16614, 0.170505,
                            0.174336, 0.177696, 0.18064, 0.183216],
            
                   [0.0334936, 0.0460898, 0.0573777, 0.0672837,
                            0.0758924, 0.0833333, 0.0897416, 0.095244, 
                            0.0999552, 0.103977, 0.107401, 0.110306,
                            0.112764, 0.114835, 0.116576, 0.118034],
            
                   [0.0072949, 0.0130273, 0.0189089, 0.0243794,
                           0.0292073, 0.0333333, 0.0367819, 0.0396151,
                           0.0419095, 0.0437445, 0.0451958, 0.0463324,
                           0.0472145, 0.0478939, 0.0484136, 0.0488088],
            
                   [0.00209801, 0.00476888, 0.00798924, 0.0112246,
                           0.0141615, 0.0166667, 0.0187142, 0.0203349,
                           0.0215852, 0.0225293, 0.0232297, 0.0237417,
                           0.0241113, 0.0243756, 0.024563, 0.0246951],
            
                   [0.0000961894, 0.000403434, 0.00100751, 0.00180597,
                           0.00262342, 0.00333333, 0.00388495, 0.00428087, 
                           0.00454894, 0.00472277, 0.00483205, 0.00489927, 
                           0.00493998, 0.00496439, 0.00497894, 0.00498756]]))

# UNUSED
@struct.dataclass
class EnvParams:
    n: int = 14
    k: int = 1
    d: int = 5
    max_steps_in_episode: int = 30

    
class MetaCodeDiscovery(environment.Environment):
    """
    Environment for the noise-aware discovery of QEC codes and encodings for different error models simultaneously.
    
    Args:
        n_qubits_physical (int): Number of physical qubits available
        n_qubits_logical (int): Number of logical qubits
        gates (list(CliffordGates)): List of Clifford gates to prepare the encoding circuit
        code_distance (int): Target code distance
        graph (list(tuple), optional): Graph of the qubit connectivity. Default: all-to-all qubit connectivity
        max_steps (int, optional): The number of maximum gates to be applied in the circuit. Default: 30
        threshold (float, optional): Tolerance parameter under which we assume the KLs are zero. Default: 1e-10
        lbda (float, optional): Global rescaling factor for the instantaneous reward. Default: 100
        random_cZ (bool, optional): Whether a random value of the bias parameter c_Z should be generated. Default: True
        cZ (float, optional): Value of the bias parameter to be considered. Default: 1
        pX (float, optional): Probability of X error happening. Default: 0.1/3
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
            threshold = 1e-9,
            lbda = 100,
            random_cZ = True,
            cZ = 1,
            pX = 0.1/3,
            pI = 0.9,
            softness = 1,
                ):
        super().__init__()
        
        self.n_qubits_physical = n_qubits_physical
        self.n_qubits_logical = n_qubits_logical
        self.gates = gates
        self.threshold = threshold
        self.max_steps = max_steps
        self.d = code_distance
        self.lbda = lbda # Rescales reward for better convergence
        self.random_cZ = random_cZ
        self.cZ = cZ
        self.pX = pX
        self.pI = pI # Probability of no error
        
        
        self.pI_id = jnp.where(NoiseParams.pIs == self.pI)[0][0]
        
        
        self.graph = graph
        if self.graph is None:
            self.graph = []
            ## Fully connected by default
            for ii in range(self.n_qubits_physical):
                for jj in range(ii+1, self.n_qubits_physical):
                    self.graph.append((ii,jj))
                    self.graph.append((jj,ii))

        
        self.obs_shape = (2 * n_qubits_physical * (n_qubits_physical - n_qubits_logical) + 1, )

        # Initialize action tensor
        self.actions = self.action_matrix()
        
        # Symplectic metric Omega
        self.Omega = jnp.kron(jnp.array([[0,1],[1,0]], dtype=jnp.int8), jnp.eye(n_qubits_physical, dtype=jnp.uint8))

        # Initialize error operators
        self.E_mu = self.error_operators()

        # Separate X,Y,Z part of errors. Useful to compute p_mu
        self.H_X = self.E_mu[:,:n_qubits_physical]
        self.H_Z = self.E_mu[:,n_qubits_physical:]
        self.H_Y = self.H_X * self.H_Z
        
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
    
    def get_observation(self, tableau, cZ):
        '''
        Extract the check matrix of stabilizer generators for the observation
        '''
        ## Only generators without sign
        check_mat = tableau[self.n_qubits_physical + self.n_qubits_logical:].astype(jnp.float32)
     
        return jnp.concatenate((check_mat.flatten(), jnp.array([(cZ-0.5)/1.5])))
    
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
        
        return E_mu

    
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
        return self.lbda * ( jnp.sum(state.p_mu) - jnp.sum(state.p_mu * jnp.any(((self.E_mu @ self.Omega) @ check_matrix.T)%2, axis=1), axis=0) - jnp.dot(state.p_mu, jnp.sum(inS, axis=-1)) )
    
    
    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """Performs step transitions in the environment."""
        
        prev_terminal = self.is_terminal(state, params)
        
        # Update state
        state = EnvState( (state.tableau @ self.actions[action]) % 2, state.time + 1, state.cZ, state.p_mu)
        
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
        self, key: chex.PRNGKey, params: EnvParams, noise: Optional[NoiseParams] = NoiseParams
    ) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""
        
        tableau = TableauSimulator(self.n_qubits_physical)
        init_state = tableau.current_tableau[0]

            
        if self.random_cZ:
            # Choose a value of cZ using a uniform PDF
            cZ_id = jax.random.choice(key, jnp.array(range(len(noise.cZs))))
            
            # Select the noise parameters accordingly
            cZ = noise.cZs[cZ_id] 
            # pI_id = jnp.where(noise.pIs == self.pI)[0][0]
            pX = noise.pXs[self.pI_id, cZ_id]
                
        else:
            cZ = self.cZ
            pX = self.pX
        
        # Update p
        p = jnp.array([self.pI, pX, pX, pX**cZ])
        
        # Update p_mu
        p_mu = p[1] ** jnp.sum(self.H_X - self.H_Y, axis=1) * p[2] ** jnp.sum(self.H_Y, axis=1) * p[3] ** jnp.sum(self.H_Z - self.H_Y, axis=1) * p[0] ** (self.n_qubits_physical - jnp.sum(self.H_X - self.H_Y + self.H_Z, axis=1)) 
        # Normalize p_mu
        p_mu /= jnp.max(p_mu)        

        state = EnvState(
            tableau = init_state,
            time = 0,
            cZ = cZ,
            p_mu = p_mu,
        )
        
        return self.get_obs(state), state

    def get_obs(self, state: EnvState, params: Optional[EnvParams] = EnvParams) -> chex.Array:
        """Applies observation function to state."""
        
        return self.get_observation(state.tableau, state.cZ).flatten()

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal."""
        # Check termination criteria
        done_encoding = jnp.abs(self.num_KL) <= self.threshold
        
        # Check number of steps in episode termination condition
        done_steps = state.time >= self.max_steps
        
        done = jnp.logical_or(done_encoding, done_steps)
        return done

    @property
    def name(self) -> str:
        """Environment name."""
        return "MetaCodeDiscovery"

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

        return spaces.Box(0, 2, self.obs_shape, dtype=jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""

        return spaces.Dict(
            {
                "tableau": spaces.Box(0, 1, self.obs_shape, jnp.uint8),
                "time": spaces.Discrete(params.max_steps),
            }
        )
