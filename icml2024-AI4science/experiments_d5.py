import os
import sys
from code_finder import CodeFinderCSS
import pickle
import json
import stim
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from qiskit import QuantumCircuit

def convert_stim_to_qiskit_circuit(encoding_circuit):
    qc = QuantumCircuit(n_qubits_physical)

    for circ in encoding_circuit:
        # Split the text
        spl = str(circ).strip().split(' ')
        gate_name = spl[0]
        # two qubit gates
        if gate_name == 'CX':
            for ii in range(0, (len(spl) - 1), 2):
                qc.cx(int(spl[ii+1]), int(spl[ii+2]))
        # one qubit gate
        elif gate_name == 'H':
            for ii in range(1, len(spl)):
                qc.h(int(spl[ii]))
        elif gate_name == 'S':
            for ii in range(1, len(spl)):
                qc.s(int(spl[ii]))
    return qc

n_qubits_physical = int(sys.argv[1])
n_qubits_logical = int(sys.argv[2])
code_distance = int(sys.argv[3])
graph_connectivity = str(sys.argv[4])
bell = bool(int(sys.argv[5]))


if n_qubits_physical <= 20:
    max_steps = 128 - 64 * (graph_connectivity == "All-to-All") - 48 * (graph_connectivity == "NN-2") - (n_qubits_physical-n_qubits_logical)//2 * (bell)
elif n_qubits_physical < 25:
    max_steps = 128 + 12 * (graph_connectivity == "NN-1") - 58 * (graph_connectivity == "All-to-All") - (n_qubits_physical-n_qubits_logical)//2 * (bell)
else:
    max_steps = 128 + 22 * (graph_connectivity == "NN-1") - 64 * (graph_connectivity == "All-to-All") - (n_qubits_physical-n_qubits_logical)//2 * (bell)

# With a single config variable we define our training
init_H = [n_qubits_logical+2*i for i in range((n_qubits_physical-n_qubits_logical)//2)]
config = {
    "N": n_qubits_physical,
    "K": n_qubits_logical,
    "D": code_distance,
    "INIT_H": init_H,
    "BELL": bell,
    "MAX_STEPS": max_steps,
    "WHICH_GATES": ["cx"],
    "GRAPH": graph_connectivity,
    "SOFTNESS": 1,
    "P_I": 0.7,
    "LAMBDA": 0.1 * (7. / float(n_qubits_physical)),
    "SEED": 42, # Single random seed controls all experiments
    "LR": 1e-3,
    "NUM_ENVS": 128,
    "NUM_STEPS": max_steps,
    "TOTAL_TIMESTEPS": 2e6, 
    "UPDATE_EPOCHS": 4,
    "NUM_MINIBATCHES": 16,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.2,
    "ENT_COEF": 0.05,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "ACTIVATION": "relu",
    "HIDDEN_DIM": 400,
    "ANNEAL_LR": True,
    "NUM_AGENTS": 8,
    "COMPUTE_METRICS": True,
}

print("n:", config["N"], "k:", config["K"], 'd:', config["D"], "connectivity:", config["GRAPH"], "Bell:", config["BELL"])

results_folder_name = 'results/%d-%d-%d-%s/' % (n_qubits_physical, n_qubits_logical, code_distance, graph_connectivity)

if not os.path.exists(results_folder_name):
    os.makedirs(results_folder_name)

params = None
# config["HIDDEN_DIM"] += 400*(config["GRAPH"] == "All-to-All")

# Start with d = 3, then d=4, then d=5
# This transfer learning has better convergence
for d in range(3,config["D"]+1):
    config["D"] = d
    print("Training with d:%d" % d)

    # config["MAX_STEPS"] -= 40 * (d == 3)
    # config["NUM_STEPS"] -= 40 * (d == 3)

    # config["MAX_STEPS"] += 20 * (d == 4)
    # config["NUM_STEPS"] += 20 * (d == 4)

    # config["MAX_STEPS"] += 20 * (d == 5)
    # config["NUM_STEPS"] += 20 * (d == 5)
    
    config["TOTAL_TIMESTEPS"] += 2e6*(config["D"] == 4)
    config["TOTAL_TIMESTEPS"] += 1e7*(config["D"] == 5)

    config["TOTAL_TIMESTEPS"] += 3e6*(config["GRAPH"] == "NN-1")*(config["D"] == 4)
    config["TOTAL_TIMESTEPS"] += 3e6*(config["GRAPH"] == "All-to-All")*(config["D"] == 4)
    config["TOTAL_TIMESTEPS"] += 2e7*(config["GRAPH"] == "NN-1")*(config["D"] == 5)
    config["TOTAL_TIMESTEPS"] += 1e7*(config["GRAPH"] == "All-to-All")*(config["D"] == 5)

    finder = CodeFinderCSS(config)
    params, metrics = finder.train(params=params)

    print("Evaluating agents:")
    data = finder.evaluate()

    if d < 5:
        for _ in range(config["NUM_AGENTS"]):
            gates = data[_]['gates']
            actions = []

            for g in gates:
                gate_name = g.split("(")[0].split(".")[1]
                qubit_ids = g.split("(")[1].split(")")[0]
                instruction = '.append("%s", [%s])' % (gate_name, qubit_ids)
                    
                actions.append(instruction)

            circ = stim.Circuit()
            for action in actions:
                eval('circ%s' % action)

            qc = convert_stim_to_qiskit_circuit(circ)
            qc.draw('latex_source', filename='%s/%d_%d_%d_%s_%d_%d.tex' % (results_folder_name, n_qubits_physical, n_qubits_logical, d, graph_connectivity, bell, _))
            qc.draw('mpl', filename='%s/%d_%d_%d_%s_%d_%d.png' % (results_folder_name, n_qubits_physical, n_qubits_logical, d, graph_connectivity, bell, _))

            # We save circuits in stim format
            circ.to_file('%s/%d_%d_%d_%s_%d_%d.stim' % (results_folder_name, n_qubits_physical, n_qubits_logical, d, graph_connectivity, bell, _))

        # Halve learning rate for smoother transfer learning
        config["LR"] /= 2.


# print("Evaluating agents:")
# data = finder.evaluate()

returns = metrics["returned_episode_returns"]
lengths = metrics["returned_episode_lengths"]


x = np.linspace(0, config["TOTAL_TIMESTEPS"], len(returns[0]))

fig = plt.figure()
# Each curve is a different agent
for i in range(config["NUM_AGENTS"]):
    plt.plot(x, returns[i])
plt.xlabel("Number of timesteps")
plt.ylabel("Return")
plt.show() 

fig.savefig('%s/return-%d-%d-%d-%s-%d.png' % (results_folder_name, n_qubits_physical, n_qubits_logical, code_distance, graph_connectivity, bell), format='png', dpi=1200, bbox_inches='tight')
plt.close()

fig = plt.figure()
# Each curve is a different agent
for i in range(config["NUM_AGENTS"]):
    plt.plot(x, lengths[i])
plt.xlabel("Number of timesteps")
plt.ylabel("Circuit size")
plt.show() 

fig.savefig('%s/circuit-size-%d-%d-%d-%s-%d.png' % (results_folder_name, n_qubits_physical, n_qubits_logical, code_distance, graph_connectivity, bell), format='png', dpi=1200, bbox_inches='tight')
plt.close()

print("Saving data...")
# We save our data
pickle.dump(returns, open('%s/returns-%s-%d.p' % (results_folder_name, graph_connectivity, bell), 'wb'))
pickle.dump(lengths, open('%s/lengths-%s-%d.p' % (results_folder_name, graph_connectivity, bell), 'wb'))
pickle.dump(params, open('%s/params-%s-%d' % (results_folder_name, graph_connectivity, bell), 'wb'))


json.dump(config, open('%s/config-%s-%d.json' % (results_folder_name, graph_connectivity, bell), 'w'))
json.dump(data, open('%s/data-%s-%d.json' % (results_folder_name, graph_connectivity, bell), 'w'))


for _ in range(config["NUM_AGENTS"]):
    gates = data[_]['gates']
    actions = []

    for g in gates:
        gate_name = g.split("(")[0].split(".")[1]
        qubit_ids = g.split("(")[1].split(")")[0]
        instruction = '.append("%s", [%s])' % (gate_name, qubit_ids)
            
        actions.append(instruction)

    circ = stim.Circuit()
    for action in actions:
        eval('circ%s' % action)

    qc = convert_stim_to_qiskit_circuit(circ)
    qc.draw('latex_source', filename='%s/%d_%d_%d_%s_%d_%d.tex' % (results_folder_name, n_qubits_physical, n_qubits_logical, code_distance, graph_connectivity, bell, _))
    qc.draw('mpl', filename='%s/%d_%d_%d_%s_%d_%d.png' % (results_folder_name, n_qubits_physical, n_qubits_logical, code_distance, graph_connectivity, bell, _))

    # We save circuits in stim format
    circ.to_file('%s/%d_%d_%d_%s_%d_%d.stim' % (results_folder_name, n_qubits_physical, n_qubits_logical, code_distance, graph_connectivity, bell, _))

print("DONE")