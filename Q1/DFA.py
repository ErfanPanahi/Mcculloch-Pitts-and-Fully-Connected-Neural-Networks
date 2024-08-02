import numpy as np
import pandas as pd

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

# df = pd.DataFrame(columns = ['Present State', 'DFA Input', 'Next State', 'Acceptance'])
# for i in range(4):
#     ns, acc = dfa(i, 0)
#     df.loc[2*i] = [i, 0, ns, acc]
#     ns, acc = dfa(i, 1)
#     df.loc[2*i + 1] = [i, 1, ns, acc]
# print(df)