import pennylane as qml
from pennylane import numpy as np

#Definition of a device: dive deep later:
dev = qml.device("default.qubit", wires=2)
dev2 = qml.device("lightning.qubit", wires=2)

#Now we have to pairing the device with our quantum circuit. This procedure is necessaary to run the circuit.Otherwise, we won't be able to obtain any otput
#A pair (device , quantum circuit) is called a quantum node.
@qml.qnode(dev) #<--- Technically this is a decorator.

#definition of a quantum circuit: a quantum circuit is a function
def firstQuantumFunction(theta):
    qml.RX(theta, wires=0)
    qml.Hadamard(wires=0)
    qml.PauliY(wires=1)
    qml.Hadamard(wires=1)
    
    return qml.state() # <--- complex np.array

print(firstQuantumFunction(np.pi/4), "\n")
print(type(firstQuantumFunction(np.pi/4)))

qArray = firstQuantumFunction(np.pi/4)

for i in range(len(qArray)):
    if i == 0:
        print("\nAmplitude of |00> is: ", qArray[i])
    elif i == 1:
        print("\nAmplitude of |01> is: ", qArray[i])
    elif i == 2:
        print("\nAmplitude of |10> is: ", qArray[i])
    elif i == 3:
        print("\nAmplitude of |11> is: ", qArray[i])


# We can also define a variable that store a quantum node (circuit + device)
qNode1 = qml.QNode(firstQuantumFunction, dev)
qNode2 = qml.QNode(firstQuantumFunction, dev2)
print("\nThis is qNode1, with \"default.qubit\": ", qNode1(np.pi/4), "\n")
print("This is qNode2, with \"lightning.qubit\": ", qNode2(np.pi/4), "\n")
print("############################################################\n")


for i in range(len(qNode2(np.pi/4))):
    if i == 0:
        print("\nAmplitude of |00> is: ", qNode2(np.pi/4)[i])
    elif i == 1:
        print("\nAmplitude of |01> is: ", qNode2(np.pi/4)[i])
    elif i == 2:
        print("\nAmplitude of |10> is: ", qNode2(np.pi/4)[i])
    elif i == 3:
        print("\nAmplitude of |11> is: ", qNode2(np.pi/4)[i])



#Now we use a different approache to create our quantum circuit: 
#First we build smaller pieces of circuit, named subCircuit, then we'll use this pieces to create the full circuit.

def subCircuit1(theta):
    qml.RX(theta, wires=0)
    qml.PauliY(wires=1)

    

def subCircuit2():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0,1])

    

def fullCircuit(theta, phi):
    subCircuit1(theta)
    subCircuit2()
    qml.RX(phi, wires=1)
    qml.PauliY(wires=0)

    return qml.state()

    


theta = np.pi/4
phi = np.pi/4

qNode4 = qml.QNode(fullCircuit, dev)

print("############################################################\n")
print("\n", qml.draw(qNode4)(theta,phi), "\n")
print("\n", qNode4(theta,phi), "\n")
        
qNode3 = qml.QNode(fullCircuit, dev2)
print("############################################################\n")
print("\n", qml.draw(qNode3)(theta,phi), "\n")
print("\n", qNode3(theta,phi), "\n")

print(dev.wires)



def subcircuit_1(angle, wire_list):
    """
    Implements the first subcircuit as a function of the RX gate angle
    and the list of wires wire_list on which the gates are applied
    """
    qml.RX(angle, wire_list[0])
    qml.PauliY(wire_list[1])

def subcircuit_2(wire_list):
    """
    Implements the second subcircuit as a function of the list of wires 
    wire_list on which the gates are applied
    """

    qml.Hadamard(wire_list[0])
    qml.CNOT(wire_list)

dev = qml.device("default.qubit", wires = [0,1])

@qml.qnode(dev)
def full_circuit2(theta, phi):
    """
    Builds the full quantum circuit given the input parameters
    """

    subcircuit_1(theta, dev.wires)
    subcircuit_2(dev.wires)
    qml.PauliY(wires = 0)
    qml.RX(phi, wires = 1)

    return qml.state()

final = full_circuit2(np.pi/4, np.pi/4)
print("############################################################\n")
print(final)






