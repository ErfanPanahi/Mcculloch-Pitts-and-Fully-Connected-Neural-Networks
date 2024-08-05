# Mcculloch-Pitts Neural Networks

In this question, we aim to first become familiar with **deterministic finite automaton (DFA)** and then design a **neural network** for it.

## Deterministic Finite Automaton (DFA)

In simple terms, a deterministic finite automaton (DFA) can be thought of as a black box that receives input and announces it in the output if it detects a specific pattern in the inputs. It uses a set of states to store the observed patterns.

**Example.** Consider a deterministic finite automaton that can recognize the pattern "100" at least once in the alphabet {0,1}. After observing the first "100" it remains in the accepting state. The state diagram of the deterministic finite automaton is shown in Figure 1.

* The number inside each circle represents the state number. 
* The numbers on the edges are the inputs that cause the current state to transition to the next state.
* The process starts from state number zero.
* If the inputs are exhausted and we are in a state that is double-lined (state three), the desired input pattern has been detected by the machine (it is accepted).

<p align="center">
  <img src="https://github.com/user-attachments/assets/1035b193-c13c-4383-8772-ffe1ea614029" width="600" height="250" >
</p>

Consider the input 011001.
1. Initially, in state zero with input 0 (**0**11001), we stay in state zero.
2. Then, in state zero with the next input 1 (0**1**1001), we move to state one.
3. Now, in state one with input 1 (01**1**001), we return to state one.
4. In state one with input 0 (011**0**01), we move to state two.
5. In state two with input 0 (0110**0**1), we move to state three.
6. In state three with input 1 (01100**1**), we return to state three.

Since the inputs are exhausted and we have remained in state three, the input string is accepted, and the pattern has been recognized by the deterministic finite automaton.

Now we can draw the state transition table for the deterministic finite automaton as Table 1.

Now we are going to simulate the given DFA using an extended McCulloch-Pitts neuron. The present state and input of the DFA will be considered as the input to the neural network, while the next state and whether the state is accepting (acceptance = 1, non-acceptance = 0) will be considered as the output of the neural network. (Three input neurons and three output neurons.)
Note that the state numbers, inputs, and whether the states are accepting or not are all binary. Also, the timing order of operations in this question is not important, so there is no need to consider delays for operations.

<p align="center">
  <img src="https://github.com/user-attachments/assets/4d0aec04-5a5b-40cf-b590-54826792ad0d" width="600" height="250" >
</p>

Now we simplify the state transition table by converting the inputs and outputs to binary form, as shown in Table 2.

<p align="center">
  <img src="https://github.com/user-attachments/assets/4d8c359f-963b-4046-a5e1-aec9ce0b3cd0" width="900" height="250" >
</p>


To facilitate the design of the desired network using McCulloch-Pitts networks, we will derive the logical equation for each of the outputs in terms of the inputs.

<p align="center">
  <img src="https://github.com/user-attachments/assets/915a05cf-27c5-4088-b3eb-b25aa1ea452b" width="400" height="100" >
</p>


Figure 2 shows the McCulloch-Pitts neural network corresponding to the logical gates we need. In these networks, the threshold is set to 2 ($\theta=2$).

<p align="center">
  <img src="https://github.com/user-attachments/assets/e556be50-9fa0-4507-ad70-e891ed06a195" width="800" height="400" >
</p>

The neural network designed using the McCulloch-Pitts network with threshold=θ = 2 is shown in Figure 3.

<p align="center">
  <img src="https://github.com/user-attachments/assets/2dfe264b-c0fd-4e88-99b7-f04d343303f9" width="800" height="450" >
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/b01b992d-7e3b-4633-af8a-7f5e82180229" width="800" height="470" >
</p>


Using Python, we will implement the designed (Feed-Forward) network and display the output for all states for all inputs as shown in Figure 4. (File Name: [DFA.py](https://github.com/ErfanPanahi/Mcculloch-Pitts-and-Fully-Connected-Neural-Networks/blob/main/Q1/DFA.py))

```python
def dfa(ps, dfa_input):
    # Defining Weights
    W1 = np.array([[1, 2, 2, 0],
                   [1, 0, 0, 2], 
                   [0, -1, 2, -1]])

    W2 = np.array([[2, 0, 2], 
                   [2, 0, 2], 
                   [0, 2, 0], 
                   [2, 0, 0]])
    
    # Input Layer
    x = np.append(DtoB(ps), dfa_input)

    # Hidden Layer
    z1 = np.matmul(W1.T, x)
    a1 = MandP(z1, 2) 

    # Output Layer
    z2 = np.matmul(W2.T, a1)
    a2 = MandP(z2, 2)

    # Mapping the Outputs
    acc = a2[2]
    ns = BtoD(a2[:2])
    return ns, acc

def MandP(h, thr):
    return np.array(h >= thr).astype(int)

def DtoB(ps):
    psb = f'0b{ps:02b}'
    return np.array([int(psb[2]), int(psb[3])])

def BtoD(nsb):
    return int(str(nsb[0]) + str(nsb[1]), 2)
```

```python
df = pd.DataFrame(columns = ['Present State', 'DFA Input', 'Next State', 'Acceptance'])
for i in range(4):
    ns, acc = dfa(i, 0)
    df.loc[2*i] = [i, 0, ns, acc]
    ns, acc = dfa(i, 1)
    df.loc[2*i + 1] = [i, 1, ns, acc]
df.style.set_properties(**{'text-align': 'center'})
```


<p align="center">
  <img src="https://github.com/user-attachments/assets/529dfb64-e493-48b6-83a2-a592aa9b88ea" width="650" height="300" >
</p>
