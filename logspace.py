import sys
import math
import numpy as np
import random

LOGZERO = float('nan')
np.random.seed(4)
random.seed(4)

def eexp(x):
#    if x == LOGZERO:
    if math.isnan(x):
        return 0
    else:
        return math.exp(x)

def eln(x):
    if x == 0.0:
        return LOGZERO
    elif x > 0:
        return math.log(x)
    else:
        raise ValueError("Negative input to eln(x)!:", x)

def elnsum(elnx, elny):
#    print("elnx:", elnx, "elny", elny)
#    if elnx == LOGZERO or elny == LOGZERO:
    if math.isnan(elnx) or math.isnan(elny):
        if math.isnan(elnx):
#            print("IF IF")
            return elny
        else:
#            print("IF ELSE")
            return elnx
    else:
        if elnx > elny:
#            print("ELSE IF")
            return elnx + eln(1 + math.exp(elny - elnx))
        else:
#            print("ELSE ELSE")
            return elny + eln(1 + math.exp(elnx - elny))

def elnproduct(elnx, elny):
    if math.isnan(elnx) or math.isnan(elny):
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

def get_obs(filename, map_to_obs, obs_list, num_states, obs_to_mark):
#def get_obs(filename, num_states):
    fp = open(filename, "r")
    train_obs = list()

    celltype = ""
    chrom = ""
    headers = list()
    train_keys = list()

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
            obs_to_mark[curr_index] = index_1
        
        train_keys.append(map_to_obs[key])
        #train_obs.append(map_to_obs[key])
        train_obs.append([int(x) for x in obs_at_this_time])

    fp.close()

    train_obs = np.array(train_obs)

    ALPHA = 0.02

    p = np.zeros(shape=(num_states, len(headers)))

    for m in range(len(headers)):
        p[0][m] = ALPHA/2

    d_dat = np.zeros(shape=(len(train_obs), len(headers)))
    d_dat = train_obs[1:] * np.roll(train_obs, 1, axis=1)[1:]

    s_bins = np.ones(shape=(len(train_obs) - 1), dtype=np.int)

    for i in range(2, num_states + 1):
        curr_max_entropy = -1
        curr_max_group = -1
        curr_max_mark = -1

        for group in range(1, i):
            for mark in range(len(headers)):
                h_1 = 0
                h_0 = 0

                for ct_index in range(len(train_obs) - 1):
                    if s_bins[ct_index] == group:
                        if d_dat[ct_index][mark] == 1:
                            h_1 += 1
                        else:
                            h_0 += 1

                h_0 = abs(h_0)
                h_1 = abs(h_1)
                h_0 /= len(train_obs - 1)
                h_1 /= len(train_obs - 1)

                entropy = 0
                if h_0 <= 0 or h_1 <= 0:
                    entropy = 0
                else:
                    entropy = (h_0 + h_1) * math.log2(h_0 + h_1)\
                            - h_0 * math.log2(h_0)\
                            - h_1 * math.log2(h_1)

                if entropy > curr_max_entropy:
                    curr_max_entropy = entropy
                    curr_max_group = group
                    curr_max_mark = mark

        tot_bins = 0
        for ct_index in range(len(s_bins)):
            if s_bins[ct_index] == curr_max_group and\
                    d_dat[ct_index][curr_max_mark] == 1:
                s_bins[ct_index] = i - 1
                tot_bins += 1

                for mark in range(len(headers)):
                    if d_dat[ct_index][mark] == 1:
                        p[i - 1][mark] += 1

        for mark in range(len(headers)):
            p[i - 1][mark] = p[i - 1][mark] / tot_bins * (1 - ALPHA) + (ALPHA / 2)

    
    # transition probabilities
    A_ij = np.zeros(shape=(num_states, num_states))
    i_count = np.zeros(shape=(num_states))

    key_counts = [None] * num_states
        #index_1 = np.where(obs_at_this_time == '1')[0]

        #key = bitstring(index_1, len(obs_at_this_time))     

    for ct_index in range(len(s_bins) - 1):
        A_ij[s_bins[ct_index]][s_bins[ct_index + 1]] += 1
        i_count[s_bins[ct_index]] += 1

        group = s_bins[ct_index]
        if key_counts[group] == None: 
            key_counts[group] = dict()

        if train_keys[ct_index] not in key_counts[group]:
            key_counts[group][train_keys[ct_index]] = 1
        else:
            key_counts[group][train_keys[ct_index]]+= 1

    last_bin = len(s_bins) - 1
    i_count[s_bins[last_bin]] += 1

    group = s_bins[last_bin]
    if key_counts[group] == None: 
        key_counts[group] = dict()

    if train_keys[last_bin] not in key_counts[group]:
        key_counts[group][train_keys[last_bin]] = 1
    else:
        key_counts[group][train_keys[last_bin]]+= 1

#    print(i_count)

    emission_probs = np.zeros(shape=(num_states, len(obs_list)))
    #emission_probs.fill(1/len())

    for i in range(num_states):
        if key_counts[i] != None:
            tot_obs_seen = len(key_counts[i])
        else:
            tot_obs_seen = 0
        not_seen = len(obs_list) - tot_obs_seen
        for o in range(len(obs_list)):
            if i_count[i] != 0 and o in key_counts[i]:
                emission_probs[i][o] = key_counts[i][o] / i_count[i] 
            elif i_count[i] != 0:
                emission_probs[i][o] = 1/i_count[i]

#                print("i", i, "o", o, emission_probs[i][o])

        for j in range(num_states):
            if i_count[i] == 0:
                A_ij[i][j] = ALPHA / num_states
            else:
                A_ij[i][j] = (1 - ALPHA) * A_ij[i][j] / i_count[i] + ALPHA / num_states

    #for i in range(len(sbins) - 1):
    #    for j in range(len(sbins)):

    # pis are going to be random
    #print(len(obs_list))

    return {
            'train_obs' : train_keys, 
#            'B' : p,
            'B' : emission_probs,
            'A' : A_ij,
            'headers' : headers, 
            'cell': celltype
            }

#def conv_to_obs(bin_dat, known_obs, map_dict):
#    observations = list()
#    for t in range(len(bin_dat)):
#        obs_1 = np.where(bin_dat[i] == '1')[0]
#        key = bitstring(obs_1, len(bin_dat[i]))
#        
#        if key not in map_dict:
#            curr_index = len(known_obs)
#            observations.append(curr_index)
#            map_dict[key] = curr_index
#            known_obs.append(curr_index)
#        else:
#            observations.append(map_dict[key])
#
#    return observations

def get_b(obs, state, b_ar):
    '''takes care of case where we see an obs not in training data'''

#    print("b_ar", len(b_ar[state]))

    if obs >= len(b_ar[state]) or b_ar[state][obs] == 0:
        return 0.0000000000001
    else:
        return b_ar[state][obs]

    #return np.sum(b_ar[state] * obs)

#    prob = 0
#    for o in obs:
#        print("state:", state, "o:", o)
#        if o == 0:
#            prob += (1 - b_ar[state][o])
#        else:
#        prob += b_ar[state][o]

#    return prob / len(b_ar[0])


def eln_at(o, pi, A, B, T, STATES):
    eln_a = np.zeros(shape=(T, STATES))

    for i in range(STATES):
        logpi = eln(pi[i])
        logb = eln(get_b(o[0], i, B))

        #eln_a[0,i] = elnproduct(eln(pi[i]), eln(get_b(o[0], i, B)))
#        print("pi:", pi[i])
#        print("logpi:", logpi, "logb:", logb)
        eln_a[0,i] = elnproduct(logpi, logb)
#        print("eln_a at t=0, i=", i, eln_a[0,i])

    for t in range(1, T):
        for j in range(STATES):
            logalpha = LOGZERO
            for i in range(STATES):
#                print("==============================")
#                print("eln_at: t", t, "j", j, "i", i)
#                print("eln_at[t-1][i]", eln_a[t-1][i])
#                print("Aij:", A[i][j])
#                print("eln(Aij):", eln(A[i][j]))
                logalpha = elnsum(logalpha, elnproduct(eln_a[t - 1][i], eln(A[i][j])))
#                print("i:", i, "logalpha:", logalpha)
            #eln_a[t][j] = elnproduct(logalpha, eln(beta[j][o[t]]))
#            print("j:", j, "logalpha:", logalpha)
#            print("b:", get_b(o[t], j, B))
#            print("eln(b):", eln(get_b(o[t], j, B)))
            eln_a[t][j] = elnproduct(logalpha, eln(get_b(o[t], j, B)))
#            if t == T - 1:
#                print ("logalpha", logalpha)
#                print("b", get_b(o[t], j, B))
#                print("eln t", t, "j", j, eln_a[t][j])
#            print("eln_a[t][j]", eln_a[t][j])

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

def printem(B, obs_to_mark, marks):
    fp = open("emissions.txt", "w")
    fp.write("\t".join(marks) + "\n")

    for i in range(len(B)):
        string = ""
        mark_probs = np.zeros(len(marks))
        counts = np.zeros(len(marks))
        for j in range(len(B[i])):

            curr_obs_prob = B[i][j]
            curr_marks = obs_to_mark[j]

            for c in curr_marks:
                mark_probs[c] += curr_obs_prob
                counts[c] += 1
            #string += str(B[i][j]) 

        for m in range(len(mark_probs)):
            if m != 0:
                string += "\t"
            string += str(mark_probs[m] / counts[m])


        fp.write(string + "\n")
    fp.close()
            
def printtrans(A):
    fp =open("transitions.txt", "w")

    for i in range(len(A)):
        string = ""
        for j in range(len(A[i])):
            if j != 0:
                string += "\t"
            string += str(A[i][j])
        fp.write(string + "\n")

    fp.close()

def main():
    
    if len(sys.argv) < 3:
        print("usage: <num states> <file: training data>", file = sys.stderr)
        exit(2)

    #ALPHA_INIT = 0.02
    num_states = int(sys.argv[1])
    train_dat_file = sys.argv[2]

    obs_list = list()
    map_ems = dict()
    obs_to_mark = dict()
    #return {
    #        'train_obs' : train_obs, 
    #        'B' : p,
    #        'A' : A_ij,
    #        'headers' : headers, 
    #        'cell': celltype
    #        }
    #result = get_obs(train_dat_file, map_ems, obs_list)
    result = get_obs(train_dat_file, map_ems, obs_list, num_states, obs_to_mark)

    train_obs = result['train_obs']
    tot_time = len(train_obs)
    num_em = len(obs_list)
    #tot_time = len(train_obs)
    B = result['B']   
    A = result['A']

    #    print(B[0])
    #    sys.exit(2)

    #print(A)

    marks = result['headers']
    cell = result['cell']
    #num_em = len(marks)
    print("Num observations:", num_em)

    '''
    num_states = 3
    train_obs = [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0]
    tot_time = len(train_obs)
    obs_list = [0, 1, 2]
    num_em = len(obs_list) 
    A = np.zeros(shape=(num_states, num_states))
    B = np.zeros(shape=(num_states, num_em))
    '''
    
    pi = np.zeros(shape=(num_states))
    #    pi.fill((1/num_states))

    
    for i in range(num_states):
        pi[i] = random.random()
#        for j in range(num_states):
#            A[i][j] = 1/num_states

    '''
    B[0][0] = 0.0 
    B[0][1] = 0.0
    B[0][2] = 0.0
    B[1][0] = 0.0
    B[1][1] = 0.0
    B[1][2] = 0.0
    B[2][0] = 0.0
    B[2][1] = 0.0
    B[2][2] = 0.0
    '''
#    for i in range(num_states):
#        for j in range(num_em):
#            B[i][j] = 1/num_em
    

    #print(B)
    #    A = np.zeros(shape=(num_states, num_states))
    #    A.fill(1/num_states)
    
    #    for i in range(num_states):
    #        for j in range(num_states):
    #            A[i][j] = random.random()


    stop = False
    prev_likelihood = None

    print("Done initializing")

#    print("A")
#    print(A)
#    print("B")
#    print(B)
#    print("pi")
#    print(pi)

    while not stop:
        # forward-backward algorithm
        alphas = eln_at(train_obs, pi, A, B, tot_time, num_states)
        print("Got alphas")

#        print(alphas[tot_time - 1])
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
#        print(pi)
        # update A
        for i in range(num_states):
            for j in range(num_states):
#                print("old aij:", A[i][j])
                A[i][j] = update_aij(gammas, xis, i, j, tot_time)
#                print("new aij:", A[i][j])

        print("Updated A")
#        print(A)

        # update B
        for j in range(num_states):
            for v in range(num_em):
                B[j][v] = update_beta(gammas, xis, train_obs, v, j, tot_time)
        
        print("Updated B")
#        print (B)

        # calculate likelihood -- just the sum of alphas since we're
        # already in log space

        curr_likelihood = None
        for s in range(num_states):
            #print("curr_likelihood", curr_likelihood)
            #print("alphas[t-1][s]", alphas[tot_time - 1][s])
            if curr_likelihood == None:
                curr_likelihood = alphas[tot_time - 1][s]
            else:
                curr_likelihood = elnsum(curr_likelihood, alphas[tot_time - 1][s])


        print("Calculated log likelihood")

        if prev_likelihood == None:
            prev_likelihood = curr_likelihood
            continue
        else:
            diff = curr_likelihood - prev_likelihood
            print("diff:", diff)

            if diff <= 1: 
                stop = True

            prev_likelihood = curr_likelihood


    printem(B, obs_to_mark, marks)
    printtrans(A)
main()
