import pennylane as qml
from jax import numpy as jnp
from matplotlib import pyplot as plt
import jax

jax.config.update("jax_platform_name", "cpu")
jax.config.update('jax_enable_x64', True)

# set the random seed
key = jax.random.PRNGKey(42)


# create a device to execute the circuit on
dev = qml.device("default.qubit", wires=3)


def CNOT_ring(wires):
    """Apply CNOTs in a ring pattern"""
    n_wires = len(wires)

    for w in wires:
        qml.CNOT([w % n_wires, (w + 1) % n_wires])


@qml.qnode(dev, diff_method="parameter-shift")
def circuit(params):
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.RZ(params[2], wires=2)

    CNOT_ring(wires=[0, 1, 2])

    qml.RX(params[3], wires=0)
    qml.RY(params[4], wires=1)
    qml.RZ(params[5], wires=2)

    CNOT_ring(wires=[0, 1, 2])
    return qml.expval(qml.PauliY(0) @ qml.PauliZ(2))


# initial parameters
params = jax.random.normal(key, [6])


print("Parameters:", params)
print("Expectation value:", circuit(params))



def parameter_shift_term(qnode, params, i):
    shifted = params.copy()
    shifted = shifted.at[i].add(jnp.pi/2)
    forward = qnode(shifted)  # forward evaluation

    shifted = shifted.at[i].add(-jnp.pi)
    backward = qnode(shifted) # backward evaluation

    return 0.5 * (forward - backward)

# gradient with respect to the first parameter
print(parameter_shift_term(circuit, params, 0))

