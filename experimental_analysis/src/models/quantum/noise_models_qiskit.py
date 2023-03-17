from qiskit_aer.noise import NoiseModel, depolarizing_error

all_single_qubit_gates = [ 
    'id', 'sx', 'u1', 'u2', 'u3',
    'x', 'y', 'z',
    'rx', 'ry', 'rz',
]

all_multi_qubit_gates = [
    'cx', 'cy', 'cz'
]

# At least all that we need for our circuits
all_gates = all_single_qubit_gates + all_multi_qubit_gates

def get_depolarising_noise_model(prob, basis_gates = all_gates):
    nm = NoiseModel(basis_gates=basis_gates)

    depol_error_single_qubit = depolarizing_error(prob, 1)
    depol_error_two_qubit = depolarizing_error(prob, 2)

    nm.add_all_qubit_quantum_error(depol_error_single_qubit, all_single_qubit_gates)
    nm.add_all_qubit_quantum_error(depol_error_two_qubit, all_multi_qubit_gates)

    return nm

