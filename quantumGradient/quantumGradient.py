import pennylane as qml
import jax
from jax import numpy as jnp
from matplotlib import pyplot as plt

# Library for array oriented numerical computation. It support automatic differentiation and Just In Time compilation.
jax.config.update("jax_platform_name", "cpu")
jax.config.update('jax_enable_x64', True)


# set the random seed
key = jax.random.PRNGKey(42)

#Define a device: 3 qubits circuit
dev = qml.device("default.qubit", wires=3)

# Definition of CNOT ring, it applies a CNOT on every consecutive pair of qubits
def CNOT_ring(wires):
    nWires = len(wires)

    for i in wires:
        qml.CNOT(wires=[i % nWires, (i + 1) % nWires])
#Decorator to create a qnode:
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

    # Returning the expected value:
    return qml.expval(qml.PauliY(0) @ qml.PauliZ(2))

'''
    Now that we have define the CNOT ring function, and we have built our differentiable qnode.
    1) We are goint to implement a function that computes the gradient (using parameter shift rule) of the function circuit with respecto to a single parameter
    2) We'll use the gradeint function to implement an iterative function that updates the parameters of the circuit. (backpropagation like)

'''





if __name__ == "__main__":
    
    # Define the initial parameters
    params = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    print("Parameters:", params)
    print("Expectation value:", circuit(params))

    fig, ax = qml.draw_mpl(circuit, decimals=2)(params)
    plt.show()


