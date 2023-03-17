import csv
from math import log2, factorial, ceil

num_hidden_layers = 2
num_hidden_units = 128

def num_qubits(n):
    return 2*n + 2

def num_layers(n, l=1):
    return n*l

def layer_depth(n):
    # Rx, Ry, Rz, 2 f. cyclic CZ
    return 5

def circuit_depth(n, l=1):
    return num_layers(n, l) * layer_depth(n)

def num_inputs(n):
    return 2*n**2 + 2*n

def num_outputs_critic(n):
    return 1

def num_outputs_actor(n):
    return n**2 - n

def num_params_qc(n, l=1):
    return 2*num_qubits(n)*num_layers(n, l)

def num_params_actor_quantum(n, l=1):
    return num_params_qc(n, l) + \
        num_qubits(n)*num_outputs_actor(n) + \
        num_outputs_actor(n)

def num_params_critic_quantum(n, l=1):
    return num_params_qc(n, l) + 2

def num_params_quantum(n, l=1):
    return num_params_actor_quantum(n, l) + num_params_critic_quantum(n, l)

def num_params_quantum_critic(n, l=1):
    return num_params_actor_classical(n) + num_params_critic_quantum(n, l)

def num_params_quantum_actor(n, l=1):
    return num_params_actor_quantum(n, l) + num_params_critic_classical(n)

def num_params_actor_classical(n, l=1):
    layers = [num_hidden_units for _ in range(num_hidden_layers)] + [num_outputs_actor(n)]
    num_params = tmp = num_inputs(n)
    for l in layers:
        num_params += tmp*l + l
        tmp = l
    return num_params

def num_params_critic_classical(n, l=1):
    layers = [num_hidden_units for _ in range(num_hidden_layers)] + [num_outputs_critic(n)]
    num_params = tmp = num_inputs(n)
    for l in layers:
        num_params += tmp*l + l
        tmp = l
    return num_params

def num_params_classical(n, l=1):
    return num_params_actor_classical(n) + num_params_critic_classical(n)

# SIGMOD'23 approach
def get_num_mandatory_variables_original(num_relations):
    num_predicates = num_relations - 1
    num_joins = num_relations - 1
    return num_relations*(2*num_joins+1) + (3*num_predicates)*(num_joins-1)

# VLDB'24 approach
def get_num_mandatory_variables_new(num_relations):
    num_predicates = num_relations - 1
    num_joins = num_relations - 1
    return (num_relations + num_predicates)*(num_joins-1)

# QDSM'23 approach
def get_num_mandatory_variables_bushy(num_relations):
    num_predicates = num_relations - 1
    num_joins = num_relations - 1
    num_vanilla_variables = num_joins*(3*num_relations + 2*num_joins + num_predicates)
    num_ancillary_variables = num_joins*num_joins*(2*num_relations + num_joins)
    return num_vanilla_variables + num_ancillary_variables

# BiDEDE approach Winker et al.
def get_num_qubits_single_step(num_relations):
    dividend = factorial(2*(num_relations-1))
    divisor =  pow(2, num_relations-1) * factorial(num_relations-1)
    l = log2(dividend / divisor)
    return ceil(l)

# Nayak et al. approach (BiDEDE)
def get_num_variables_exponential(num_relations):
    return pow(2, num_relations) - num_relations - 1

if __name__=="__main__":
    fn = "num_qubits.csv"
    rel_range = range(4, 31)
    with open(fn, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "num_relations", "num_qubits"])
    for n in rel_range:
        n_qubits_qrl = num_qubits(n)
        n_qubits_sigmod = get_num_mandatory_variables_original(n)
        n_qubits_vldb = get_num_mandatory_variables_new(n)
        n_qubits_qdsm = get_num_mandatory_variables_bushy(n)
        n_qubits_exponential = get_num_variables_exponential(n)
        n_qubits_single_step = get_num_qubits_single_step(n)
        with open(fn, "a") as f:
            writer = csv.writer(f)
            writer.writerow(["QRL", n, n_qubits_qrl])
            writer.writerow(["SIGMOD'23", n, n_qubits_sigmod])
            writer.writerow(["VLDB'24", n, n_qubits_vldb])
            writer.writerow(["QDSM'23", n, n_qubits_qdsm])
            writer.writerow(["BiDEDE'23", n, n_qubits_exponential])
            writer.writerow(["SingleStep", n, n_qubits_single_step])

