import sympy
import tensorflow as tf
from tensorflow import keras
import tensorflow_quantum as tfq
import cirq
import numpy as np
import math

def layer_ry_rz_cz(num_qubits, idx):
    circuit = cirq.Circuit()
    qubits = cirq.GridQubit.rect(1, num_qubits)
    y_params = sympy.symbols(f'param_l{idx}_y_q0:{num_qubits}')
    z_params = sympy.symbols(f'param_l{idx}_z_q0:{num_qubits}')

    # variational part
    for i in range(num_qubits):
        circuit += [
            cirq.ry(y_params[i]).on(qubits[i]),
            cirq.rz(z_params[i]).on(qubits[i])
        ]

    # entangling part
    for i in range(num_qubits):
        circuit += cirq.CZ.on(qubits[i], qubits[(i+1) % num_qubits])

    return circuit, np.concatenate([y_params, z_params])

def layer_ry_rz_czc(num_qubits, idx):
    circuit = cirq.Circuit()
    qubits = cirq.GridQubit.rect(1, num_qubits)
    y_params = sympy.symbols(f'param_l{idx}_y_q0:{num_qubits}')
    z_params = sympy.symbols(f'param_l{idx}_z_q0:{num_qubits}')

    # variational part
    for i in range(num_qubits):
        circuit += [
            cirq.ry(y_params[i]).on(qubits[i]),
            cirq.rz(z_params[i]).on(qubits[i])
        ]

    # entangling part
    for i in range(num_qubits):
        for j in range(i+1, num_qubits):
            circuit += cirq.CZ.on(qubits[i], qubits[j])

    return circuit, np.concatenate([y_params, z_params])

def encoding_ops_rx(input, qubit):
    return cirq.rx(input).on(qubit)

def encoding_ops_rx_ry(input, qubit):
    return [cirq.rx(input).on(qubit), cirq.ry(input).on(qubit)]

def encoding_ops_rx_rz(input, qubit):
    return [cirq.rx(input).on(qubit), cirq.ry(input).on(qubit)]

def encoding_ops_rx_ry_rz(input, qubit):
    return [cirq.rx(input).on(qubit), cirq.ry(input).on(qubit), cirq.rz(input).on(qubit)]

class VQC_Layer(keras.layers.Layer):
    """
    This class represents a trainable VQC using TFQ.

    Attributes:

        num_qubits: Number of Qubits
        num_layers: Number of VQC-Layers
        encoding_ops: Rotation gates applied to inputs
        layertype: VQC-Layer architecture as cirq circuit
        data_reuploading: whether to apply data re-uploading
        initializer: initializer for weights,
        num_actions: Number of actions
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
        super(VQC_Layer, self).__init__()

        self.circuit = cirq.Circuit()

        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.num_inputs = num_inputs
        self.qubits = cirq.GridQubit.rect(1, self.num_qubits)
        self.encoding_ops = encoding_ops
        self.data_reuploading = data_reuploading
        self.incr_uploading = incremental_data_uploading

        if data_reuploading:
            self.num_uploading_layers = num_layers

        if incremental_data_uploading:
            assert num_inputs is not None
            assert num_inputs % num_qubits == 0
            self.num_incr_layers = num_inputs // num_qubits
            if data_reuploading:
                assert num_layers % self.num_incr_layers == 0
                self.num_uploading_layers = num_layers // self.num_incr_layers
        else:
            self.num_incr_layers = 0

        self.initializer = initializer

        # input part
        self.input_params, input_ops = self._get_input_params()

        if input_ops is not None:
            self.circuit += input_ops

        var_circuit, param_symbols = self.create_circuit(layer)
        self.circuit += var_circuit

        # Observable
        action_qubit_ratio = int(num_qubits/num_actions)
        self.observable = []
        if action_qubit_ratio < 1:
            self.observable += [
                cirq.PauliString(cirq.Z(qubit)) for qubit in self.qubits
            ]
        elif action_qubit_ratio == num_qubits:
            self.observable += [
                cirq.PauliString(cirq.Z(qubit) for qubit in self.qubits)
            ]
        else:
            for i in range(1, num_actions):
                self.observable += [
                    cirq.PauliString(cirq.Z(qubit) for qubit in self.qubits[(i-1)*action_qubit_ratio:i*action_qubit_ratio])
                ]
            self.observable += [
                cirq.PauliString(cirq.Z(qubit) for qubit in self.qubits[i*action_qubit_ratio:])
            ]

        if depol_error_prob > 0:
            self.circuit = self.circuit.with_noise(cirq.depolarize(p=depol_error_prob))
            self.vqc = tfq.layers.NoisyControlledPQC(self.circuit, self.observable, repetitions=256, sample_based=True)
        else:
            self.vqc = tfq.layers.ControlledPQC(self.circuit, self.observable)

        self.num_weights = len(param_symbols)
        self.symbols = [str(symb) for symb in np.concatenate([param_symbols, self.input_params])]
        self.indices = tf.constant([self.symbols.index(a) for a in sorted(self.symbols)])
        self.empty_circuit = tfq.convert_to_tensor([cirq.Circuit()])

    def build(self, input_shape):
        self.params = self.add_weight(
            name='vqc_weights',
            shape=(1, self.num_weights),
            initializer=self.initializer,
            trainable=True
        )

    def call(self, inputs):
        inputs = tf.math.scalar_mul(math.pi, inputs)
        batch_dim = tf.gather(tf.shape(inputs), 0)

        # tile to batch dimension
        tiled_up_circuits = tf.repeat(self.empty_circuit, repeats=batch_dim)
        tiled_up_params = tf.tile(self.params, multiples=[batch_dim, 1])

        # tile inputs to circuit-layer number
        if self.data_reuploading:
            inputs = tf.tile(inputs, multiples=[1, self.num_uploading_layers]) 

        joined_vars = tf.concat([tiled_up_params, inputs], axis=1)
        joined_vars = tf.gather(joined_vars, self.indices, axis=1)
        x = self.vqc([tiled_up_circuits, joined_vars])

        return x

    def create_circuit(self, layer_fkt):
        circuit = cirq.Circuit()
        param_symbols = []

        for idx in range(self.num_layers):

            # input part
            if self.data_reuploading or self.incr_uploading and idx < self.num_incr_layers:
                circuit.append([self.encoding_ops(self.input_params[idx*self.num_qubits+i], qubit) for i, qubit in enumerate(self.qubits)])

            layer, layer_symbols = layer_fkt(self.num_qubits, idx)

            circuit += layer
            param_symbols = np.concatenate([param_symbols, layer_symbols])

        return circuit, param_symbols

    def _get_input_params(self):
        ops = None
        if self.incr_uploading:
            if self.data_reuploading:
                input_params = sympy.symbols(f'inputs0:{self.num_uploading_layers}_0:{self.num_inputs // self.num_qubits}_0:{self.num_qubits}')
            else:
                input_params = sympy.symbols(f'inputs0:{self.num_inputs // self.num_qubits}_0:{self.num_qubits}')
        else:
            if self.data_reuploading:
                input_params = sympy.symbols(f'inputs0:{self.num_uploading_layers}_0:{self.num_qubits}')
            else:
                input_params = sympy.symbols(f'inputs0:{self.num_qubits}')
                ops = [self.encoding_ops(input_params[i], qubit) for i, qubit in enumerate(self.qubits)]
        return input_params, ops


class VQC_Layer_Action_Space(keras.layers.Layer):
    """
    This class represents a trainable VQC using TFQ.

    Attributes:

        num_qubits: Number of Qubits
        num_layers: Number of VQC-Layers
        encoding_ops: Rotation gates applied to inputs
        layertype: VQC-Layer architecture as cirq circuit
        data_reuploading: whether to apply data re-uploading
        initializer: initializer for weights,
        num_actions: Number of actions
    """
    def __init__(self,  num_relations,
                        num_uploading_layers,
                        encoding_ops = encoding_ops_rx,
                        layer = layer_ry_rz_cz,
                        data_reuploading = False,
                        initializer = keras.initializers.Zeros,
                        critic = False,
                        depol_error_prob = 0.,
        ):
        super(VQC_Layer_Action_Space, self).__init__()

        self.num_qubits = self.num_actions = num_relations*(num_relations-1)
        self.num_inputs = 2*(num_relations**2 + num_relations)
        self.num_layers_per_block = math.ceil(self.num_inputs / self.num_qubits)
        self.num_layers = self.num_layers_per_block * num_uploading_layers
        self.num_uploading_layers = num_uploading_layers

        self.qubits = cirq.GridQubit.rect(1, self.num_qubits)
        self.encoding_ops = encoding_ops
        self.data_reuploading = data_reuploading

        self.initializer = initializer

        # input part
        self.input_params = self._get_input_params()

        self.circuit, param_symbols = self.create_circuit(layer)

        # Observable
        if critic:
            self.observable = [
                cirq.PauliString(cirq.Z(qubit) for qubit in self.qubits)
            ]
        else:
            self.observable = [
                cirq.PauliString(cirq.Z(qubit)) for qubit in self.qubits
            ]

        if depol_error_prob > 0:
            self.circuit = self.circuit.with_noise(cirq.depolarize(p=depol_error_prob))
            self.vqc = tfq.layers.NoisyControlledPQC(self.circuit, self.observable, repetitions=256, sample_based=True)
        else:
            self.vqc = tfq.layers.ControlledPQC(self.circuit, self.observable)

        self.num_weights = len(param_symbols)
        symbols = [str(symb) for symb in np.concatenate([param_symbols, self.input_params])]
        self.indices = tf.constant([symbols.index(a) for a in sorted(symbols)])
        self.empty_circuit = tfq.convert_to_tensor([cirq.Circuit()])

    def build(self, input_shape):
        self.params = self.add_weight(
            name='vqc_weights',
            shape=(1, self.num_weights),
            initializer=self.initializer,
            trainable=True
        )

    def call(self, inputs):
        inputs = tf.math.scalar_mul(math.pi, inputs)
        batch_dim = tf.gather(tf.shape(inputs), 0)

        # tile to batch dimension
        tiled_up_circuits = tf.repeat(self.empty_circuit, repeats=batch_dim)
        tiled_up_params = tf.tile(self.params, multiples=[batch_dim, 1])

        # tile inputs to circuit-layer number
        if self.data_reuploading:
            inputs = tf.tile(inputs, multiples=[1, self.num_uploading_layers])

        joined_vars = tf.concat([tiled_up_params, inputs], axis=1)
        joined_vars = tf.gather(joined_vars, self.indices, axis=1)
        x = self.vqc([tiled_up_circuits, joined_vars])

        # scale outputs from [-1, 1] -> [0, 1]
        x = (x + 1) / 2

        return x

    def create_circuit(self, layer_fkt):
        circuit = cirq.Circuit()
        param_symbols = []

        for idx in range(self.num_uploading_layers):
            input_ctr = 0

            for l in range(self.num_layers_per_block):
                # input part
                if self.data_reuploading or idx < 1:
                    for i, qubit in enumerate(self.qubits):
                        if input_ctr >= self.num_inputs:
                            continue
                        circuit.append(self.encoding_ops(self.input_params[idx*self.num_inputs + input_ctr], qubit))
                        input_ctr += 1

                layer, layer_symbols = layer_fkt(self.num_qubits, idx*self.num_layers_per_block + l)

                circuit += layer
                param_symbols = np.concatenate([param_symbols, layer_symbols])

        return circuit, param_symbols

    def _get_input_params(self):
        if self.data_reuploading:
            input_params = sympy.symbols(f'inputs0:{self.num_uploading_layers}_0:{self.num_inputs}')
        else:
            input_params = sympy.symbols(f'inputs0:{self.num_inputs}')
        return input_params

