from copy import copy
import numpy as np
import math
import tensorflow as tf
from tensorflow import keras
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.primitives import BackendEstimator, utils as qu
from qiskit_aer import AerSimulator
from qiskit_algorithms.gradients import ParamShiftEstimatorGradient, LinCombEstimatorGradient
from qiskit_algorithms.gradients.utils import _make_param_shift_parameter_values
from abc import ABC
import tensorflow_quantum as tfq

from src.models.quantum.noise_models_qiskit import get_depolarising_noise_model
from src.models.quantum.vqc_layer_tfq import VQC_Layer as VQC_Layer_TFQ

def layer_ry_rz_cz(num_qubits, idx):
    circuit = QuantumCircuit(num_qubits)
    y_params = [ Parameter(f'param_l{idx}_y_q{i}') for i in range(num_qubits) ]
    z_params = [ Parameter(f'param_l{idx}_z_q{i}') for i in range(num_qubits) ]

    # variational part
    for i in range(num_qubits):
        circuit.ry(y_params[i], i)
        circuit.rz(z_params[i], i)

    # entangling part
    for i in range(num_qubits):
        circuit.cz(i, (i+1) % num_qubits)

    return circuit, y_params + z_params

def layer_ry_rz_czc(num_qubits, idx):
    circuit = QuantumCircuit(num_qubits)
    y_params = [ Parameter(f'param_l{idx}_y_q{i}') for i in range(num_qubits) ]
    z_params = [ Parameter(f'param_l{idx}_z_q{i}') for i in range(num_qubits) ]

    # variational part
    for i in range(num_qubits):
        circuit.ry(y_params[i], i)
        circuit.rz(z_params[i], i)

    # entangling part
    for i in range(num_qubits):
        for j in range(i+1, num_qubits):
            circuit.cz(i, j)

    return circuit, y_params + z_params

def encoding_ops_rx(input, qubit, num_qubits):
    circuit = QuantumCircuit(num_qubits)
    circuit.rx(input, qubit)
    return circuit

def encoding_ops_rx_ry(input, qubit, num_qubits):
    circuit = QuantumCircuit(num_qubits)
    circuit.rx(input, qubit)
    circuit.ry(input, qubit)
    return circuit

def encoding_ops_rx_rz(input, qubit, num_qubits):
    circuit = QuantumCircuit(num_qubits)
    circuit.rx(input, qubit)
    circuit.rz(input, qubit)
    return circuit

def encoding_ops_rx_ry_rz(input, qubit, num_qubits):
    circuit = QuantumCircuit(num_qubits)
    circuit.rx(input, qubit)
    circuit.ry(input, qubit)
    circuit.rz(input, qubit)
    return circuit

class VQC_Layer_Base(keras.layers.Layer, ABC):
    """
    This class represents the trainable VQC Base Class using Qiskit.

    Attributes:

        encoding_ops: Rotation gates applied to inputs
        layertype: VQC-Layer architecture as qiskit circuit
        data_reuploading: whether to apply data re-uploading
        initializer: initializer for weights,
    """
    def __init__(self,
                 encoding_ops = encoding_ops_rx,
                 layer = layer_ry_rz_cz,
                 data_reuploading=False,
                 initializer=keras.initializers.Zeros,
                 depol_error_prob=0.,
    ):
        super(VQC_Layer_Base, self).__init__()

        self.encoding_ops = encoding_ops
        self.data_reuploading = data_reuploading
        self.initializer = initializer
        self.layer = layer
        self.depol_error_prob = depol_error_prob

    def initialize(self):
        self.circuit = QuantumCircuit(self.num_qubits)

        # input part
        self.input_params, input_ops = self._get_input_params()

        if input_ops is not None:
            self.circuit = self.circuit.compose(input_ops)

        var_circuit, param_symbols = self.create_circuit(self.layer)
        self.circuit = self.circuit.compose(var_circuit)

        self.noise_model = get_depolarising_noise_model(self.depol_error_prob)

        ideal_backend = AerSimulator()
        if self.depol_error_prob > 0.:
            noisy_backend = AerSimulator(noise_model=self.noise_model)
        else:
            noisy_backend = AerSimulator()
        if 'GPU' in ideal_backend.available_devices():
            ideal_backend.set_options(device='GPU')
            noisy_backend.set_options(device='GPU')

        self.ideal_estimator = BackendEstimator(ideal_backend)
        self.noisy_estimator = BackendEstimator(noisy_backend)

        # For runtime purposes, we choose an ideal backend for gradient calculation
        self.gradient_estimator = LinCombEstimatorGradient(self.ideal_estimator)

        self.num_weights = len(param_symbols)
        self.param_symbols = param_symbols + self.input_params
        symbols = [symb.name for symb in self.param_symbols]
        self.indices = tf.constant([symbols.index(a) for a in sorted(symbols)])

        self.diff = tfq.differentiators.Adjoint()

    def build(self, input_shape):
        self.params = self.add_weight(
            name='vqc_weights',
            shape=(1, self.num_weights),
            initializer=self.initializer,
            trainable=True
        )

    def _call(self, inputs):
        inputs = tf.math.scalar_mul(math.pi, inputs)
        batch_dim = tf.gather(tf.shape(inputs), 0)

        if not hasattr(inputs, "numpy"):
            return tf.zeros((batch_dim, len(self.observable_str_list)))

        # tile to batch dimension
        tiled_up_params = tf.tile(self.params, multiples=[batch_dim, 1])

        # tile inputs to circuit-layer number
        if self.data_reuploading:
            inputs = tf.tile(inputs, multiples=[1, self.num_uploading_layers]) 

        joined_vars = tf.concat([tiled_up_params, inputs], axis=1)
        joined_vars = tf.gather(joined_vars, self.indices, axis=1)
        x = self._run_circuit_ideal_backward(joined_vars)

        return x

    @tf.custom_gradient
    def _run_circuit(self, parameter_values: tf.Tensor):

        batch_dim = tf.gather(tf.shape(parameter_values), 0).numpy()
        num_params = len(self.param_symbols)
        num_observables = len(self.observable_str_list)

        tiled_up_parameter_values = tf.repeat(parameter_values, num_observables, axis=0)
        tiled_up_parameter_values = tiled_up_parameter_values.numpy()

        circuits = [self.circuit] * batch_dim * num_observables
        observables = [ SparsePauliOp(obs) for obs in np.tile(self.observable_str_list, batch_dim) ]

        jobs = self.noisy_estimator.run(circuits, observables, tiled_up_parameter_values)

        results = tf.convert_to_tensor(jobs.result().values)
        results = tf.reshape(results, (batch_dim, num_observables))

        def circuit_grad(upstream):
            exp_vals = np.zeros((num_observables, batch_dim, num_params))
            for i, obs in enumerate(self.observable_str_list):
                observables = [ SparsePauliOp(obs) ] * batch_dim
                gradient_jobs = self.gradient_estimator.run(circuits, observables, parameter_values.numpy())
                grads = gradient_jobs.result().gradients
                grads = np.array(grads)
                exp_vals[i] = grads
            grads = tf.convert_to_tensor(exp_vals)
            grads = tf.einsum("ij,jik->ik", upstream, grads)
            grads = tf.cast(grads, tf.float32)
            return grads

        return results, circuit_grad

    @tf.custom_gradient
    def _run_circuit_ideal_backward(self, parameter_values: tf.Tensor):
        batch_dim = tf.gather(tf.shape(parameter_values), 0).numpy()
        num_params = len(self.param_symbols)
        num_observables = len(self.observable_str_list)

        tiled_up_parameter_values = tf.repeat(parameter_values, num_observables, axis=0)
        tiled_up_parameter_values = tiled_up_parameter_values.numpy()

        circuits = [self.circuit] * batch_dim * num_observables
        observables = [ SparsePauliOp(obs) for obs in np.tile(self.observable_str_list, batch_dim) ]

        jobs = self.noisy_estimator.run(circuits, observables, tiled_up_parameter_values)

        results = tf.convert_to_tensor(jobs.result().values)
        results = tf.reshape(results, (batch_dim, num_observables))

        def circuit_grad(upstream):
            tfq_circ = tfq.convert_to_tensor([self.tfq_layer.circuit])
            tiled_up_circuits = tf.repeat(tfq_circ, repeats=batch_dim)

            tfq_observable = tfq.convert_to_tensor([self.tfq_layer.observable])
            tiled_up_observable = tf.tile(tfq_observable, multiples=[batch_dim, 1])

            grads = self.diff.differentiate_analytic(
                    tiled_up_circuits,
                    self.tfq_layer.symbols,
                    parameter_values,
                    tiled_up_observable,
                    results,
                    upstream,
            )
            return grads

        return results, circuit_grad



class VQC_Layer(VQC_Layer_Base):
    """
    This class represents a trainable VQC using Qiskit.

    Attributes:

        num_qubits: Number of Qubits
        num_layers: Number of VQC-Layers
        encoding_ops: Rotation gates applied to inputs
        layertype: VQC-Layer architecture as qiskit circuit
        data_reuploading: whether to apply data re-uploading
        initializer: initializer for weights
        num_actions: Number of actions
        incremental_data_uploading: whether to apply incremental data uploading
        num_inputs: number of inputs
    """
    def __init__(self,  num_qubits,
                        num_layers,
                        encoding_ops = encoding_ops_rx,
                        layer = layer_ry_rz_cz,
                        data_reuploading=False,
                        initializer=keras.initializers.Zeros,
                        num_actions=2,
                        incremental_data_uploading=False,
                        num_inputs=None,
                        depol_error_prob=0.,
    ):
        super(VQC_Layer, self).__init__(
                    encoding_ops,
                    layer,
                    data_reuploading,
                    initializer,
                    depol_error_prob,
        )
        self.num_qubits = num_qubits
        self.num_layers = num_layers

        self.num_inputs = num_inputs

        self.incr_uploading = incremental_data_uploading

        if self.data_reuploading:
            self.num_uploading_layers = num_layers

        if incremental_data_uploading:
            assert num_inputs is not None
            assert num_inputs % num_qubits == 0
            self.num_incr_layers = num_inputs // num_qubits
            if self.data_reuploading:
                assert num_layers % self.num_incr_layers == 0
                self.num_uploading_layers = num_layers // self.num_incr_layers
        else:
            self.num_incr_layers = 0

        super().initialize()

        # Observable
        action_qubit_ratio = num_qubits // num_actions
        pauli_str_list = []
        id_pauli_str = ["I"] * num_qubits
        if action_qubit_ratio < 1:
            for i in range(num_qubits-1, -1, -1):
                pauli_str = copy(id_pauli_str)
                pauli_str[i] = "Z"
                pauli_str = "".join(pauli_str)
                pauli_str_list.append(pauli_str)
        elif action_qubit_ratio == num_qubits:
            pauli_str_list.append("Z" * num_qubits)
        else:
            for i in range(num_actions-1):
                pauli_str = copy(id_pauli_str)
                for j in range(action_qubit_ratio):
                    pauli_str[(-i+1)*action_qubit_ratio + j] = "Z"
                pauli_str = "".join(pauli_str)
                pauli_str_list.append(pauli_str)
            remainder = action_qubit_ratio + num_qubits % num_actions
            pauli_str = copy(id_pauli_str)
            for j in range(remainder):
                pauli_str[j] = "Z"
            pauli_str = "".join(pauli_str)
            pauli_str_list.append(pauli_str)

        self.observable_str_list = pauli_str_list

        # This only supports the default circuit with rx encoding and ry_rz_cx layer
        self.tfq_layer = VQC_Layer_TFQ(
                num_qubits,
                num_layers,
                data_reuploading=data_reuploading,
                num_actions=num_actions,
                incremental_data_uploading=incremental_data_uploading,
                num_inputs=num_inputs,
        )

    def call(self, inputs):
        return super()._call(inputs)

    def create_circuit(self, layer_fkt):
        circuit = QuantumCircuit(self.num_qubits)
        param_symbols = []

        for idx in range(self.num_layers):

            # input part
            if self.data_reuploading or self.incr_uploading and idx < self.num_incr_layers:
                for i in range(self.num_qubits):
                    circuit = circuit.compose(self.encoding_ops(self.input_params[idx*self.num_qubits+i], i, self.num_qubits))

            layer, layer_symbols = layer_fkt(self.num_qubits, idx)

            circuit = circuit.compose(layer)
            param_symbols = param_symbols + layer_symbols

        return circuit, param_symbols

    def _get_input_params(self):
        ops = None
        if self.incr_uploading:
            if self.data_reuploading:
                input_params = [ Parameter(f'inputs0{ul}_0{r}_0{q}') for ul in range(self.num_uploading_layers) \
                        for r in range(self.num_inputs // self.num_qubits) for q in range(self.num_qubits) ]
            else:
                input_params = [ Parameter(f'inputs0{r}_0{q}') \
                        for r in range(self.num_inputs // self.num_qubits) for q in range(self.num_qubits) ]
        else:
            if self.data_reuploading:
                input_params = [ Parameter(f'inputs0{ul}_0{q}') \
                        for ul in range(self.num_uploading_layers) for q in range(self.num_qubits) ]
            else:
                input_params = [ Parameter(f'inputs0{q}') for q in range(self.num_qubits) ]
                ops = [self.encoding_ops(input_params[i], i, self.num_qubits) for i in range(self.num_qubits)]
        return input_params, ops


class VQC_Layer_Action_Space(VQC_Layer_Base):
    """
    This class represents a trainable VQC using Qiskit.

    Attributes:

        num_qubits: Number of Qubits
        num_layers: Number of VQC-Layers
        encoding_ops: Rotation gates applied to inputs
        layer: VQC-Layer architecture as qiskit circuit
        data_reuploading: whether to apply data re-uploading
        initializer: initializer for weights,
        critic: Whether the model is a critic part
    """
    def __init__(self,  num_relations,
                        num_uploading_layers,
                        encoding_ops = encoding_ops_rx,
                        layer = layer_ry_rz_cz,
                        data_reuploading = False,
                        initializer = keras.initializers.Zeros,
                        critic = False,
                        depol_error_prob=0.,
    ):
        super(VQC_Layer_Action_Space, self).__init__(
                    encoding_ops,
                    layer,
                    data_reuploading,
                    initializer,
                    depol_error_prob,
        )

        self.num_qubits = self.num_actions = num_relations*(num_relations-1)
        self.num_inputs = 2*(num_relations**2 + num_relations)
        self.num_layers_per_block = math.ceil(self.num_inputs / self.num_qubits)
        self.num_layers = self.num_layers_per_block * num_uploading_layers
        self.num_uploading_layers = num_uploading_layers

        super().initialize()

        # Observable
        pauli_str_list = []
        id_pauli_str = ["I"] * self.num_qubits
        if critic:
            pauli_str_list.append("Z" * self.num_qubits)
        else:
            for i in range(self.num_qubits-1, -1, -1):
                pauli_str = copy(id_pauli_str)
                pauli_str[i] = "Z"
                pauli_str = "".join(pauli_str)
                pauli_str_list.append(pauli_str)

        self.observable_str_list = pauli_str_list

    def call(self, inputs):
        x = super()._call(inputs)

        # scale outputs from [-1, 1] -> [0, 1]
        x = (x + 1) / 2

        return x

    def create_circuit(self, layer_fkt):
        circuit = QuantumCircuit(self.num_qubits)
        param_symbols = []

        for idx in range(self.num_uploading_layers):
            input_ctr = 0

            for l in range(self.num_layers_per_block):
                # input part
                if self.data_reuploading or idx < 1:
                    for i in range(self.num_qubits):
                        if input_ctr >= self.num_inputs:
                            continue
                        circuit = circuit.compose(self.encoding_ops(self.input_params[idx*self.num_inputs + input_ctr], i, self.num_qubits))
                        input_ctr += 1

                layer, layer_symbols = layer_fkt(self.num_qubits, idx*self.num_layers_per_block + l)

                circuit = circuit.compose(layer)
                param_symbols = param_symbols + layer_symbols

        return circuit, param_symbols

    def _get_input_params(self):
        if self.data_reuploading:
            input_params = [ Parameter(f'inputs0{ul}_0{q}') \
                    for ul in range(self.num_uploading_layers) for q in range(self.num_qubits) ]
        else:
            input_params = [ Parameter(f'inputs0{q}') for q in range(self.num_qubits) ]
        return input_params, None

