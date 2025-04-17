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
'''
    Function we want to implement: 0.5 * (f(theta + pi/2) - f(theta - pi/2)) 
    where f(theta) is a quantum circuit.
'''
def parameter_shift_term(qnode, params, i): # qnode is the parametric circuit, params is the circuit's parameters and i i s the index of parameter respectto which we want to compute the gradient
    shifted = params.copy() # Don't want to modify the original parameters
    shifted = shifted.at[i].add(jnp.pi/2) # JAX function to not modify in a distributed way the parameter in i-th position
    forward = qnode(shifted) # f(theta + pi/2)

    shifted = shifted.at[i].add(-jnp.pi) 
    backward = qnode(shifted) # f(theta - pi/2)

    return 0.5 * (forward - backward)


def parameter_shift(qnode, params):

    gradients = jnp.zeros([len(params)])
    for i in range(len(params)):
        gradients = gradients.at[i].set(parameter_shift_term(qnode, params, i))

    return gradients






if __name__ == "__main__":
    
    
    # Define the initial parameters
    params = jax.random.normal(key, [6])

    print("Parameters:", params)
    print("Expectation value:", circuit(params))

    fig, ax = qml.draw_mpl(circuit, decimals=2)(params)
    plt.show()

    # Compute the gradient
    print("Derivate of the function circuit respect to the first parameter:")
    print(parameter_shift_term(circuit, params, 0))
    grads = parameter_shift(circuit, params)

    for i in range(len(grads)):
        print("Derivate of the function circuit respect to the", i+1, "parameter:")
        print(grads[i])



    


