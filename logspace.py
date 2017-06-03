import sys
import math
import numpy as np
import random

LOGZERO = None

def eexp(x):
    if x == LOGZERO:
        return 0
    else:
        return math.exp(x)

def eln(x):
    if x == 0:
        return LOGZERO
    elif x > 0:
        return math.log(x)
    else:
        raise ValueError("Negative input to eln(x)!")

def elnsum(elnx, elny):
    if elnx == LOGZERO or elny == LOGZERO:
        if elnx == LOGZERO:
            return elny
        else:
            return elnx
    else:
        if elnx > elny:
            return elnx + eln(1 + math.exp(elny - elnx))
        else:
            return elny + eln(1 + math.exp(elnx - elny))

def elnproduct(elnx, elny):
    if elnx == LOGZERO or elny == LOGZERO:
        return LOGZERO
    else:
        return elnx + elny

#STATES = 10
#OBS = 5

#PI = [ (1/STATES) x STATES ]
#beta = [ [(1/5) x OBS] x STATES ]
#T = 20
#Aij = [ [ (1/STATES) x STATES ] x STATES ]

def bitstring(ar, max_int):
    bits = [0] * max_int

    for elem in ar:
        bits[int(elem)] = 1

    return ''.join([ str(x) for x in bits])

def get_obs(filename, map_to_obs, obs_list):
    fp = open(filename, "r")
    train_obs = list()

    celltype = ""
    chrom = ""
    headers = list()

    for line in fp:
        obs_at_this_time = np.array(line.rstrip().split('\t'))

        if obs_at_this_time[0] != '0' and\
                obs_at_this_time[0] != '1':
            if obs_at_this_time[-1].startswith('chr'):
                celltype = obs_at_this_time[0]
                chrom = obs_at_this_time[1]
            else:
                headers = obs_at_this_time

            continue


        index_1 = np.where(obs_at_this_time == '1')[0]

        key = bitstring(index_1, len(obs_at_this_time))     

        if key not in map_to_obs:
            curr_index = len(obs_list)
            map_to_obs[key] = curr_index
            obs_list.append(curr_index)
        
        train_obs.append(map_to_obs[key])

    fp.close()

    #print(len(obs_list))

    return train_obs

def conv_to_obs(bin_dat, known_obs, map_dict):
    observations = list()
    for t in range(len(bin_dat)):
        obs_1 = np.where(bin_dat[i] == '1')[0]
        key = bitstring(obs_1, len(bin_dat[i]))
        
        if key not in map_dict:
            curr_index = len(known_obs)
            observations.append(curr_index)
            map_dict[key] = curr_index
            known_obs.append(curr_index)
        else:
            observations.append(map_dict[key])

    return observations

def get_b(obs, state, b_ar):
    '''takes care of case where we see an obs not in training data'''

    if obs >= len(b_ar):
        return 0
    else:
        return b_ar[obs][state]


def eln_at(o, pi, A, B, T, STATES):
    eln_a = np.zeros(shape=(T, STATES))

    for i in range(STATES):
        logpi = eln(pi[i])
        logb = eln(get_b(o[0], i, B))

        #eln_a[0,i] = elnproduct(eln(pi[i]), eln(get_b(o[0], i, B)))
        eln_a[0,i] = elnproduct(logpi, logb)

    for t in range(1, T):
        for j in range(STATES):
            logalpha = LOGZERO
            for i in range(STATES):
                logalpha = elnsum(logalpha, elnproduct(eln_a[t - 1][i], eln(A[i][j])))
            #eln_a[t][j] = elnproduct(logalpha, eln(beta[j][o[t]]))
            eln_a[t][j] = elnproduct(logalpha, eln(get_b(o[t], j, B)))

    return eln_a

def eln_bt(o, pi, A, B, T, STATES):
    eln_b = np.zeros(shape=(T, STATES))

    for i in range(STATES):
        eln_b[T - 1][i] = 0

    for t in range(T - 2, -1, -1):
        for i in range(STATES):
            logbeta = LOGZERO
            for j in range(STATES):
                #logbeta = elnsum( logbeta, elnproduct(eln(aij[i][j]), elnproduct(beta[j][o[t+1]], eln_b[t + 1][j])) )
                elna = eln(A[i][j])
                b = eln(get_b(o[t+1], j, B))
                prev_beta = eln_b[t + 1][j]

#                print("elna:", elna, "b:", b, "prev_beta:", prev_beta)
                logbeta = elnsum( logbeta, elnproduct(elna, elnproduct(b, prev_beta)) )

            #print("BETA:", logbeta)
            eln_b[t][i] = logbeta
    return eln_b

def eln_gammat(eln_at, eln_bt, T, STATES):
    eln_gamma = np.zeros(shape=(T, STATES))

    for t in range(T):
        normalizer = LOGZERO
        for i in range(STATES):
#            if t == 0:
#                print("in eln_gamma", eln_at[t][i], eln_bt[t][i])
            eln_gamma[t][i] = elnproduct(eln_at[t][i], eln_bt[t][i])
            normalizer = elnsum(normalizer, eln_gamma[t][i])
        
        for i in range(STATES):
            eln_gamma[t][i] = elnproduct(eln_gamma[t][i], 0 - normalizer)

#            if t == 0:
#                print("exp elngamma", eexp(eln_gamma[0][i]))

    return eln_gamma

def eln_xit(eln_at, eln_bt, o, A, B, T, STATES):
    eln_xi = np.zeros(shape=(T, STATES, STATES))

    for t in range(T-1):
        normalizer = LOGZERO
        for i in range(STATES):
            for j in range(STATES):
#                eln_xi = elnproduct( eln_at[t][i], elnproduct(eln(A[i][j]), elnproduct(eln(beta[j][o[t+1]]), eln_xi[t][i][j] ) ) )
                eln_xi[t][i][j] = elnproduct( eln_at[t][i], elnproduct(eln(A[i][j]), elnproduct(eln(get_b(o[t+1], j, B)), eln_xi[t][i][j] ) ) )

                normalizer = elnsum(normalizer, eln_xi[t][i][j])

        for i in range(STATES):
            for j in range(STATES):
                eln_xi[t][i][j] = elnproduct(eln_xi[t][i][j], 0 - normalizer)

    return eln_xi

def update_pi(eln_gammat, i):
    #print(eln_gammat[0][i])
    return eexp(eln_gammat[0][i])

def update_aij(eln_gammat, eln_xit, i, j, T):
    numerator = LOGZERO
    denominator = LOGZERO

    for t in range(T - 1):
        numerator = elnsum(numerator, eln_xit[t][i][j])
        denominator = elnsum(denominator, eln_gammat[t][i])

#        print("Update aij: num", numerator)
#        print("Update aij: denom", denominator)

    return eexp(elnproduct(numerator, 0 - denominator))

def update_beta(eln_gammat, eln_xit, o, vk, j, T):
    numerator = LOGZERO
    denominator = LOGZERO

    for t in range(T):
        if o[t] == vk:
            numerator = elnsum(numerator, eln_gammat[t][j])

        denominator = elnsum(denominator, eln_gammat[t][j])

    return eexp(elnproduct(numerator, 0 - denominator))

def main():

    
    if len(sys.argv) < 3:
        print("usage: <num states> <file: training data>", file = sys.stderr)
        exit(2)

    ALPHA_INIT = 0.02
    num_states = int(sys.argv[1])
    train_dat_file = sys.argv[2]

    obs_list = list()
    map_ems = dict()
    train_obs = get_obs(train_dat_file, map_ems, obs_list)

    num_em = len(obs_list)
    tot_time = len(train_obs)

    print("Num observations:", num_em)


'''

    num_states = 2
    train_obs = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
    tot_time = len(train_obs)
    num_em = 2
    obs_list = [0, 1]
'''
    pi = np.zeros(shape=(num_states))
#    pi.fill((1/num_states))

    for i in range(num_states):
        pi[i] = random.random()

    B = np.zeros(shape=(num_em, num_states))
#    B.fill(1/num_em)

#    B[0][0] = 0.8
#    B[1][0] = 0.2
#    B[0][1] = 0.3
#    B[1][1] = 0.7


    #print(B)
    A = np.zeros(shape=(num_states, num_states))
#    A.fill(1/num_states)
    
    for i in range(num_states):
        for j in range(num_states):
            A[i][j] = random.random()


    stop = False
    prev_likelihood = None

    print("Done initializing")

    while not stop:
        # forward-backward algorithm
        alphas = eln_at(train_obs, pi, A, B, tot_time, num_states)
        print("Got alphas")

        #print(alphas)
        #exit(2)

        betas = eln_bt(train_obs, pi, A, B, tot_time, num_states)
        print("Got betas")

#        print(betas)

        xis = eln_xit(alphas, betas, train_obs, A, B, tot_time, num_states)
        print("Got xis")
        gammas = eln_gammat(alphas, betas, tot_time, num_states)
        print("Got gammas")

        # update pi
        for s in range(num_states):
#            print("old pi:", pi[s])
            pi[s] = update_pi(gammas, s)
#            print("new pi:", pi[s])

        print("Updated pi")

        # update A
        for i in range(num_states):
            for j in range(num_states):
#                print("old aij:", A[i][j])
                A[i][j] = update_aij(gammas, xis, i, j, tot_time)
#                print("new aij:", A[i][j])

        print("Updated A")

        # update B
        for v in range(num_em):
            for j in range(num_states):
                B[v][j] = update_beta(gammas, xis, train_obs, v, j, tot_time)
        
        print("Updated B")
        #print (B)

        # calculate likelihood -- just the sum of alphas since we're
        # already in log space

        curr_likelihood = 0
        for s in range(num_states):
            curr_likelihood = elnsum(curr_likelihood, alphas[tot_time - 1][s])


        print("Calculated log likelihood")

        if prev_likelihood == None:
            prev_likelihood = curr_likelihood
            continue
        else:
            diff = curr_likelihood - prev_likelihood
            print("diff:", diff)

            if diff <= 0.0001: 
                stop = True

            prev_likelihood = curr_likelihood


main()
