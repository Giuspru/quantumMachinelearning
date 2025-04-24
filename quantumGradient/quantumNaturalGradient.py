import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from matplotlib import pyplot as plt

dev = qml.device("lightning.qubit", wires=3)

'''
    Statring with the implementation of a parametric quantum circuit with 4 trainable parameters, the angles of the 4 rotations.
    In pennyLane a circuit is a function for that we can compute the gradient of the circuit and the gradient descending algorithm,
    which will provide the minimum outcome of the circuit.

'''
@qml.qnode(dev, interface="autograd", diff_method="parameter-shift")
def circuit(params):
    # |psi_0>: state preparation
    qml.RY(np.pi / 4, wires=0)
    qml.RY(np.pi / 3, wires=1)
    qml.RY(np.pi / 7, wires=2)

    # V0(theta0, theta1): Parametrized layer 0
    qml.RZ(params[0], wires=0)
    qml.RZ(params[1], wires=1)

    # W1: non-parametrized gates
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])

    # V_1(theta2, theta3): Parametrized layer 1
    qml.RY(params[2], wires=1)
    qml.RX(params[3], wires=2)

    # W2: non-parametrized gates
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])

    return qml.expval(qml.PauliY(0))

# Use pennylane.numpy for trainable parameters
params = pnp.array([0.432, -0.123, 0.543, 0.233])

'''
Let's make a comparison between the optimization of QuantumNaturalGradient and the standard gradient descent.
GradientDescendOptimizer.

'''

steps = 200
init_params = pnp.array([0.432, -0.123, 0.543, 0.233], requires_grad=True)


# Let's make 200 steps to optimize the circuit with the traditional gradient descent
gd_cost = []
opt = qml.GradientDescentOptimizer(0.01)

theta = init_params
for _ in range(steps):
    theta = opt.step(circuit, theta)
    gd_cost.append(circuit(theta))

# And now with the Natural Gradient Descent
qng_cost = []
opt = qml.QNGOptimizer(0.01)

theta = init_params
for _ in range(steps):
    theta = opt.step(circuit, theta)
    qng_cost.append(circuit(theta))


plt.style.use("seaborn-v0_8-whitegrid")
plt.plot(gd_cost, "b", label="Vanilla gradient descent")
plt.plot(qng_cost, "g", label="Quantum natural gradient descent")

plt.ylabel("Cost function value")
plt.xlabel("Optimization steps")
plt.legend()
plt.show()