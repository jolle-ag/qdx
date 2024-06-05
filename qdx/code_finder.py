from qdx.envs.code_discovery import CodeDiscovery
from qdx.envs.meta_code_discovery import MetaCodeDiscovery
from qdx.envs.delta_code_discovery import DeltaCodeDiscovery
from qdx.envs.max_code_discovery import MaxCodeDiscovery
from qdx.simulators.clifford_gates import CliffordGates
import os 
import jax
from qdx.make_train import make_train, ActorCritic
from qdx.utils import Utils
import time 
import json
import pickle
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
import stim

class CodeFinder:
    
    def __init__(self, config):
        self.config = config   
        # Initialize the RL environment
        self.make_environment()
        
    def make_graph(self):

        if self.config["GRAPH"] == "All-to-All":
            graph = []
            # Fully connected
            for ii in range(self.config["N"]):
                for jj in range(ii+1, self.config["N"]):
                    graph.append((ii,jj))
                    graph.append((jj,ii))
                    
        elif self.config["GRAPH"] == "NN-1":
            # x - x - x - x - x - x - x
            # with open boundary conditions
            graph = []
            for ii in range(self.config["N"]-1):
                graph.append((ii,(ii+1)%self.config["N"]))
                graph.append(((ii+1)%self.config["N"],ii))

        elif self.config["GRAPH"] == "NN-2":
            # Next-to-nearest neighbor with open boundary conditions
            graph = []
            for ii in range(self.config["N"]-1):
                graph.append((ii,ii+1))
                graph.append((ii+1,ii))
            for ii in range(self.config["N"]-2):
                graph.append((ii,ii+2))
                graph.append((ii+2,ii))
                
        self.graph = graph
        
        return 
        
    def make_environment(self):
        
        self.make_graph()
        
        gates = CliffordGates(self.config["N"])
        
        gate_set = []
        for g in self.config["WHICH_GATES"]:
            gate_set.append(eval("gates.%s" % g))
            
        ## For some reason this throws out the error: "'gates' is not defined"
        #gate_set = [eval("gates.%s" % g) for g in self.config["WHICH_GATES"]]
        
        if self.config["ENV_TYPE"] == "STANDARD":
            self.env = CodeDiscovery(self.config["N"], 
                                     self.config["K"], 
                                     self.config["D"],
                                     gate_set, 
                                     graph = self.graph, 
                                     max_steps = self.config["MAX_STEPS"], 
                                     lbda = self.config["LAMBDA"], 
                                     pI = self.config["P_I"], 
                                     softness = self.config["SOFTNESS"])
            
        elif self.config["ENV_TYPE"] == "NOISE-AWARE":
            self.env = MetaCodeDiscovery(self.config["N"], 
                                         self.config["K"], 
                                         self.config["D"],
                                         gate_set, 
                                         graph = self.graph, 
                                         max_steps = self.config["MAX_STEPS"],
                                         lbda = self.config["LAMBDA"],
                                         pI = self.config["P_I"], 
                                         softness = self.config["SOFTNESS"])
            
        elif self.config["ENV_TYPE"] == "DELTA":
            self.env = DeltaCodeDiscovery(self.config["N"], 
                                         self.config["K"], 
                                         self.config["D"],
                                         gate_set, 
                                         graph = self.graph, 
                                         max_steps = self.config["MAX_STEPS"],
                                         lbda = self.config["LAMBDA"],
                                         pI = self.config["P_I"], 
                                         softness = self.config["SOFTNESS"])
            
        elif self.config["ENV_TYPE"] == "MAX":
            self.env = MaxCodeDiscovery(self.config["N"], 
                                         self.config["K"], 
                                         self.config["D"],
                                         gate_set, 
                                         graph = self.graph, 
                                         max_steps = self.config["MAX_STEPS"],
                                         lbda = self.config["LAMBDA"],
                                         pI = self.config["P_I"], 
                                         softness = self.config["SOFTNESS"])

        return
    
    def train(self):
        """ Training the agent. """
        
        #### Training
        rng = jax.random.PRNGKey(self.config["SEED"])
        rngs = jax.random.split(rng, self.config['NUM_AGENTS'])
        train_vjit = jax.jit(jax.vmap(make_train(self.config, self.env)))
        t0 = time.perf_counter()
        print("==== Training started")
        outs = jax.block_until_ready(train_vjit(rngs))
        self.config["TIME"] = time.perf_counter() - t0
        print("==== Training finished, time elapsed: %.5f s" % (self.config["TIME"]))
        
        if self.config["COMPUTE_METRICS"]:
            metrics = {'returned_episode_returns': np.array(outs["metrics"]['returned_episode_returns']),
                       'returned_episode_lengths': np.array(outs["metrics"]['returned_episode_lengths'])}
            
        else:
            metrics = {'returned_episode_returns': np.array([0]),
                       'returned_episode_lengths': np.array([0])}
            
        self.params = outs['params']
        
        return self.params, metrics
    
    def evaluate(self):
        
        if self.config["ENV_TYPE"] != "NOISE-AWARE":

            data = []

            for index in range(self.config["NUM_AGENTS"]):

                ## Create a copy of parameter of the network manually
                params_eval = {}
                params_eval['params'] = {}

                for key in self.params['params'].keys():
                    params_eval['params'][key] = {} 

                    for key2 in self.params['params'][key].keys():
                        params_eval['params'][key][key2]= self.params['params'][key][key2][index]

                eval_env = self.env
                env_params = None
                # Reset
                reset_rng = jax.random.PRNGKey(self.config["SEED"]+1)
                obsv, env_state = eval_env.reset(reset_rng, None)
                # Prepare network
                network = ActorCritic(eval_env.action_space().n, activation=self.config["ACTIVATION"], hidden_dim = self.config["HIDDEN_DIM"])
                actions = []
                done = False

                while not done:
                    # Sample action
                    pi, value = network.apply(params_eval, obsv)
                    action = jnp.argmax(nn.softmax(pi.logits))
                    actions.append(action)

                    # STEP ENV
                    rng_step = jax.random.PRNGKey(self.config["SEED"]+2)
                    obsv, env_state, reward, done, info = eval_env.step(
                        rng_step, env_state, action, env_params
                    )

                gates = [eval_env.action_string_stim[action] for action in actions]

                # This is the true code distance
                code_distance = self.compute_distance(gates)


                # print(f"Agent: {index+1}", end="\r")


                data.append({'n': self.config["N"], 'k': self.config["K"], 'd': code_distance, 'gates': gates})
                
        elif self.config["ENV_TYPE"] == "NOISE-AWARE":

            data = self.evaluate_noise_aware()

        return data
    
    def compute_distance(self, gates):
        
        eval_d = self.config["D"]
        
        utils = Utils(self.config["N"], self.config["K"], gates, softness=self.config["N"])

        # We check KLs based on weight
        KLs = []
        for w in range(1,eval_d+1):
            E_mu = utils.error_operators(w)
            KLs.append(utils.check_KL(E_mu))

        distance = KLs.count(0)+1
        # count number of zeros

        return distance
    
    def evaluate_noise_aware(self):
        
        self.make_graph()
        gates = CliffordGates(self.config["N"])
        
        gate_set = []
        for g in self.config["WHICH_GATES"]:
            gate_set.append(eval("gates.%s" % g))
        
        data = []
        cZ_vals = [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2]
        pX_vals = [0.0072949, 0.0130273, 0.0189089, 0.0243794,0.0292073, 0.0333333, 0.0367819, 0.0396151, 0.0419095, 0.0437445, 0.0451958, 0.0463324,0.0472145, 0.0478939, 0.0484136, 0.0488088]

        for index in range(self.config["NUM_AGENTS"]):

            ## Create a copy of parameter of the network manually
            params_eval = {}
            params_eval['params'] = {}

            for key in self.params['params'].keys():
                params_eval['params'][key] = {} 

                for key2 in self.params['params'][key].keys():
                    params_eval['params'][key][key2]= self.params['params'][key][key2][index]
            
            for i,cZ in enumerate(cZ_vals):
                eval_env = MetaCodeDiscovery(self.config["N"], 
                                             self.config["K"], 
                                             self.config["D"],
                                             gate_set, 
                                             graph = self.graph, 
                                             max_steps = self.config["MAX_STEPS"],
                                             lbda = self.config["LAMBDA"],
                                             pI = self.config["P_I"], 
                                             softness = self.config["SOFTNESS"],
                                             random_cZ = False,
                                             cZ = cZ,
                                             pX = pX_vals[i],
                                            )

                env_params = None
                # Reset
                reset_rng = jax.random.PRNGKey(self.config["SEED"]+1)
                obsv, env_state = eval_env.reset(reset_rng, None)
                # Prepare network
                network = ActorCritic(eval_env.action_space().n, activation=self.config["ACTIVATION"], hidden_dim = self.config["HIDDEN_DIM"])
                actions = []
                done = False

                while not done:
                    # Sample action
                    pi, value = network.apply(params_eval, obsv)
                    action = jnp.argmax(nn.softmax(pi.logits))
                    actions.append(action)

                    # STEP ENV
                    rng_step = jax.random.PRNGKey(self.config["SEED"]+2)
                    obsv, env_state, reward, done, info = eval_env.step(
                        rng_step, env_state, action, env_params
                    )

                gates = [eval_env.action_string_stim[action] for action in actions]

                # This is the true code distance
                code_distance = self.compute_effective_distance(gates, cZ)
                
                data.append({'n': self.config["N"], 'k': self.config["K"], 'cZ': cZ, 'd_eff': code_distance, 'gates': gates})


            # print(f"Agent: {index+1}", end="\r")
            
        return data

    def compute_effective_distance(self, gates, cZ):
        
        eval_d = self.config["D"]
        
        utils = Utils(self.config["N"], self.config["K"], gates, softness=self.config["N"])

        # We check KLs based on weight
        KLs = []
        for w in range(1, min(eval_d+1, self.config["N"]+1)):
            E_mu = utils.error_operators(w)
            KLs.append(utils.check_KL_cZ(E_mu, cZ))

        distance = min(KLs)
        # Smallest undetected effective weight

        return distance
            
        