from envs.code_discovery_env import CodeDiscoveryCSS
from simulators.tableau_simulator_css import TableauSimulatorCSS
from simulators.clifford_gates_css import CliffordGatesCSS
import os 
import jax
from make_train import make_train, ActorCritic
from utils_css import UtilsCSS
import time 
import json
import pickle
import numpy as np
import jax.numpy as jnp
import flax.linen as nn

class CodeFinderCSS:
    
    def __init__(self, config):
        self.config = config   
        # Initialize the RL environment
        self.make_environment()
        
    def make_graph(self):

        if self.config["GRAPH"] == "All-to-All":
            graph = None

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
        
        gates = CliffordGatesCSS(self.config["N"])
        
        gate_set = []
        for g in self.config["WHICH_GATES"]:
            gate_set.append(eval("gates.%s" % g))
            
        ## For some reason this throws out the error: "'gates' is not defined"
        #gate_set = [eval("gates.%s" % g) for g in self.config["WHICH_GATES"]]
        

        self.env = CodeDiscoveryCSS(self.config["N"], 
                                    self.config["K"], 
                                    self.config["D"],
                                    self.config["INIT_H"],
                                    gate_set, 
                                    graph = self.graph, 
                                    max_steps = self.config["MAX_STEPS"], 
                                    lbda = self.config["LAMBDA"], 
                                    pI = self.config["P_I"], 
                                    softness = self.config["SOFTNESS"],
                                    bell = self.config["BELL"])

        return
    
    def train(self, params):
        """ Training the agent. """
        
        #### Training
        rng = jax.random.PRNGKey(self.config["SEED"])
        rngs = jax.random.split(rng, self.config['NUM_AGENTS'])
        train_vjit = jax.jit(jax.vmap(make_train(self.config, self.env, network_params_init = params)))
        t0 = time.perf_counter()
        print("==== Training started")
        outs = jax.block_until_ready(train_vjit(rngs, network_params_init = params))
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
    
    def flatten_list(self, nested_list):

        flat_list = []
        for item in nested_list:
            if isinstance(item, list):  # Check if the item is a list
                flat_list.extend(self.flatten_list(item))  # Recursively flatten the list
            else:
                flat_list.append(item)  # If it's a string, just add it to the flat list
        return flat_list
    
    def evaluate(self):

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
                # print('a:', action)

                # STEP ENV
                rng_step = jax.random.PRNGKey(self.config["SEED"]+2)
                obsv, env_state, reward, done, info = eval_env.step(
                    rng_step, env_state, action, env_params
                )
                # print('H_X:', env_state.H_X)
                # print('H_Z:', env_state.H_Z)
                # print('done:', done)

            gates = [eval_env.action_string_stim[action] for action in actions]
            gates = self.flatten_list(gates)

            tableau = TableauSimulatorCSS(self.config["N"], self.config["K"], init_H=self.config["INIT_H"], bell=self.config["BELL"])

            for g in gates:
                eval("tableau%s" % g)


            H_X = np.array(tableau.H_X).tolist()
            H_Z = np.array(tableau.H_Z).tolist()


            # This is the true code distance
            code_distance = self.compute_distance(gates)

            print(f"Agent: {index+1}", end="\r")

            if self.config["BELL"]:
                gate_instructions = ['.h('+str(i)+')' for i in self.config["INIT_H"]]+['.cx('+str(i)+','+str(i+1)+')' for i in self.config["INIT_H"]]#+['.cx('+str(i)+','+str(i-1)+')' for i in self.config["INIT_H"]]+['.cx('+str(i-1)+','+str(i)+')' for i in self.config["INIT_H"]]

                data.append({'n': self.config["N"], 'k': self.config["K"], 'd': code_distance, 'G_X': H_X, 'G_Z': H_Z, 'gates': gate_instructions + gates})
            else:
                data.append({'n': self.config["N"], 'k': self.config["K"], 'd': code_distance, 'G_X': H_X, 'G_Z': H_Z, 'gates': ['.h('+str(i)+')' for i in self.config["INIT_H"]] + gates})

        return data
    
    def compute_distance(self, gates):
        
        eval_d = self.config["D"]

        tableau = TableauSimulatorCSS(self.config["N"], self.config["K"], init_H=self.config["INIT_H"], bell=self.config["BELL"])
        
        for g in gates:
            eval("tableau%s" % g)

        utils = UtilsCSS(self.config["N"], self.config["K"], tableau, softness=self.config["SOFTNESS"])

        # We check KLs based on weight
        KLs = []
        for w in range(1,eval_d+1):
            E_mu = utils.CSS_error_operators(w)
            KLs.append(int(utils.check_KL(E_mu)))
        print('KLs:', KLs)
        distance = KLs.count(0)+1
        # count number of zeros

        return distance