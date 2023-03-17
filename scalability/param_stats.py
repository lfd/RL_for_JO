import csv
from tensorflow.keras.optimizers import Adam
import time
import tensorflow as tf
import numpy as np
import random

from num_qubits_stats import num_params_quantum_actor, num_params_quantum_critic, num_params_quantum, num_params_classical, num_qubits, circuit_depth

def get_adam_time(p):

    optimizer = Adam()
    weights = tf.Variable(np.random.rand(p), dtype=tf.float32)
    grads = tf.constant(np.random.rand(p), dtype=tf.float32)

    start = time.time_ns()
    optimizer.apply_gradients(zip([grads], [weights]))
    end = time.time_ns()
    elapsed = (end - start) * 1e-6
    return elapsed

if __name__=="__main__":
    fn = "num_params.csv"
    rel_range = list(range(4, 31))
    random.shuffle(rel_range)
    num_measurements = 1000
    num_repetitions = [1, 5, 10]
    with open(fn, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "num_relations", "num_params", "circuit_depth", "num_qubits", "t_adam", "num_repetitions", "run"])
    for i in range(num_measurements):
        for n in rel_range:
            for l in num_repetitions:
                n_q = num_params_quantum(n, l)
                n_qubits = num_qubits(n)
                c_depth = circuit_depth(n, l)
                t_adam_q = get_adam_time(n_q)
                with open(fn, "a") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Quantum", n, n_q, c_depth, n_qubits, t_adam_q, l, i])
        for n in rel_range:
            for l in num_repetitions:
                n_q = num_params_quantum_critic(n, l)
                n_qubits = num_qubits(n)
                c_depth = circuit_depth(n, l)
                t_adam_q = get_adam_time(n_q)
                with open(fn, "a") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Q-Critic", n, n_q, c_depth, n_qubits, t_adam_q, l, i])
        for n in rel_range:
            for l in num_repetitions:
                n_q = num_params_quantum_actor(n, l)
                n_qubits = num_qubits(n)
                c_depth = circuit_depth(n, l)
                t_adam_q = get_adam_time(n_q)
                with open(fn, "a") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Q-Actor", n, n_q, c_depth, n_qubits, t_adam_q, l, i])
        for n in rel_range:
            n_c = num_params_classical(n)
            t_adam_c = get_adam_time(n_c)
            with open(fn, "a") as f:
                writer = csv.writer(f)
                writer.writerow(["Classical", n, n_c, 0, 0, t_adam_c, 0, i])

