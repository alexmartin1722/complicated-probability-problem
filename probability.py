import torch 
import numpy as np 
import pandas as pd
import argparse
import os

def makeS(k):
    s = torch.ones((k,k))
    s = torch.triu(s, diagonal=0)
    return s

def makeX(k):
    x = torch.empty((k,1))
    for i in range(k):
        x[i] = 1/(i+1)
    return x

def makeM(k):
    m = torch.zeros((k,k))
    for i in range(1,k):
        m[i-1,i] = 1/i
    return m

def calcualte(S, M, X, k):
    prob_matrix = torch.empty((k,0))
    P = torch.matmul(S, M)
    x = torch.matmul(S,X)
    prob_matrix = torch.cat((prob_matrix, x), 1)
    for i in range(1, k):
        x = torch.matmul(P, x)
        prob_matrix = torch.cat((prob_matrix, x), 1)
    return prob_matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('k', type=int, default=1000)
    parser.add_argument('--out-dir', type=str, default='')
    parser.add_argument('--all_out', action='store_true')
    args = parser.parse_args()

    k = args.k
    out_dir = args.out_dir
    all_out = args.all_out

    S = makeS(k)
    X = makeX(k)
    M = makeM(k)

    s_np = S.numpy()

    x_np = X.numpy()
    m_np = M.numpy()


    prob_matrix = calcualte(S,M,X,k)
    prob_np = prob_matrix.numpy()


    C = prob_matrix[0]
    one = torch.ones((1))
    C = torch.cat((one, C), 0)
    c_np = C.numpy()


    C = torch.mul(C, (1/(k+1)))
    c_np = C.numpy()

    sum = torch.sum(C)
    print(sum)

    c_df = pd.DataFrame(c_np)
    #write to csv
    #if out_dir doesn't exist, create it
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    c_df.to_csv(out_dir + '/C.csv')

    if all_out:
        s_df = pd.DataFrame(s_np)
        x_df = pd.DataFrame(x_np)
        m_df = pd.DataFrame(m_np)
        prob_df = pd.DataFrame(prob_np)

        s_df.to_csv(out_dir + '/S.csv')
        x_df.to_csv(out_dir + '/X.csv')
        m_df.to_csv(out_dir + '/M.csv')
        prob_df.to_csv(out_dir + '/prob_matrix.csv')

if __name__ == '__main__':
    main()
