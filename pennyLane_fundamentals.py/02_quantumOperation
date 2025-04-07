import pennylane as qml
from pennylane import numpy as np

#In this section we'll introduce the quantum state prapartion: remember that by deafault a qbits is setted to |0> ---> [1. ,0.]

dev = qml.device("default.qubit", wires=2)
@qml.qnode(dev)

#This is a circuit where we have a state preparation, so that when we call the circuit, we can can provide how we want the state
def circuit(state = None):
    qml.StatePrep(state, wires=range(2), normalize=True)
    return qml.state()

state1 = circuit([0.5, 0.5, 0.5, 0.5])
print(state1)

# fuction BasiState: suppose we want the |100> state.
base1 = qml.BasisState(np.array([1,0,0]) , wires=range(3))
print(base1)






dev2 = qml.device("default.qubit", wires=3)
@qml.qnode(dev2)
#Let's start with preparing a state in this way: alpha|001> + beta|010> + gamma|100>

def prep_circuit(alpha, beta, gamma, state = None):
    qml.StatePrep([0, alpha, beta,0,gamma,0,0,0 ], wires = range(3) , normalize = True) # ---> 8 possible states
    return qml.state()

alpha, beta, gamma = 1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3),
print("############################################################\n")
print("The prepared state is", prep_circuit(alpha, beta, gamma))
print(qml.draw(prep_circuit)(alpha, beta, gamma))



#Gates under control: qml.ctrl(qml.gate, control_values, control, target) it permits to apply a gate to a wire if is respected a particular condition.
dev3 = qml.device("default.qubit", wires = 3)

@qml.qnode(dev3)
def ctrl_circuit(theta,phi):
    """Implements the circuit shown in the Codercise statement
    Args:
        theta (float): Rotation angle for RX
        phi (float): Rotation angle for RY
    Returns:
        (numpy.array): The output state of the QNode
    """

    qml.RY(phi, wires = 0)
    qml.Hadamard(wires = 1)
    qml.RX(theta, wires  =2)
    qml.ctrl(qml.S, control = (0) , control_values = (1))(wires = 1)
    qml.ctrl(qml.T, control = 1, control_values = 0)(wires = 2)
    qml.ctrl(qml.Hadamard, control = 2 , control_values = 1)(wires = 0)

    return qml.state()

print("############################################################\n")
print(ctrl_circuit(np.pi/4, np.pi/4))
print(qml.draw(ctrl_circuit)(np.pi/4, np.pi/4))


# Same as before, but this time  we use the ControlledQubitUnitary gate, that uses something that is not a qml.function, but a matrix.
dev4 = qml.device("default.qubit", wires = 2)

@qml.qnode(dev4)
def phase_kickback(matrix):
    """Applies phase kickback to a single-qubit operator whose matrix is known
    Args:S
    - matrix (numpy.array): A 2x2 matrix
    Returns:
    - (numpy.array): The output state after applying phase kickback
    """

    qml.Hadamard(wires  =0 )
    qml.ControlledQubitUnitary(matrix, control_wires = 0, control_values = 1, wires= 1)
    qml.Hadamard(wires = 0)
    return qml.state()

matrix = np.array([[-0.69165024-0.50339329j,  0.28335369-0.43350413j],
    [ 0.1525734 -0.4949106j , -0.82910055-0.2106588j ]])

print("############################################################\n")
print("The state after phase kickback is: \n" , phase_kickback(matrix))
print(qml.draw(phase_kickback)(matrix))



# To create a circuit that presents a inverted gate, we can use the adjoint function. 
dev5 = qml.device("default.qubit")

def do(k):

    qml.StatePrep([1,k], wires = [0], normalize = True)

def apply(theta):

    qml.IsingXX(theta, wires = [1,2])

@qml.qnode(dev5)
def do_apply_undo(k,theta):
    """
    Applies V, controlled-U, and the inverse of V
    Args: 
    - k, theta: The parameters for do and apply (V and U) respectively
    Returns:
    - (np.array): The output state of the routine
    """

    do(k)
    qml.ctrl(apply, control = 0, control_values = 1)(theta)
    qml.adjoint(do)(k)

    return qml.state()

k, theta = 0.5, 0.8

print("############################################################\n")
print("The output state is: \n", do_apply_undo(k, theta))
print(qml.draw(do_apply_undo)(k, theta))



