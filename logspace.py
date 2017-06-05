# Author: Aparna Rajpurkar
# GENE245 Final Project: Aparna Rajpurkar and Matt Buckley
# Log-space calculation idea based off Mann 2006 paper. 
# Overall HMM algorithm based off Rabiner 1989 HMM tutorial.

## imports ##
import sys
import math
import numpy as np
import random

# Constants and setting random seeds for reproducibility
LOGZERO = float('nan')
np.random.seed(4)
random.seed(4)

# Helper functions
def eexp(x):
    '''
    Function to return the exponential of the input. Handles case where
    input is 0.
    '''
    if math.isnan(x):
        return 0
    else:
        return math.exp(x)

def eln(x):
    '''
    Function to return the natural logarithm of the input. Handles case where
    input is 0.
    '''
    if x == 0.0:
        return LOGZERO
    elif x > 0:
        return math.log(x)
    else:
        # negative values not allowed for ln function
        raise ValueError("Negative input to eln(x)!:", x)

def elnsum(elnx, elny):
    '''
    Add two numbers in log space 
    '''
    # check if numbers are invalid (LOGZERO)
    if math.isnan(elnx) or math.isnan(elny):
        if math.isnan(elnx):
            return elny
        else:
            return elnx
    else:
        if elnx > elny:
            return elnx + eln(1 + math.exp(elny - elnx))
        else:
            return elny + eln(1 + math.exp(elnx - elny))

def elnproduct(elnx, elny):
    '''
    Multiply two numbers in log space
    '''
    if math.isnan(elnx) or math.isnan(elny):
        return LOGZERO
    else:
        return elnx + elny

def bitstring(ar, max_int):
    '''
    Given an array of indicies (marks) at which this sample is 1, 
    output a hashable, unique bitstring
    This allows a unique key for every possible combination of marks,
    which will later be used to hash into a list of observations
    '''
    bits = [0] * max_int

    for elem in ar:
        bits[int(elem)] = 1

    return ''.join([ str(x) for x in bits])

def get_obs(filename, map_to_obs, obs_list, num_states, obs_to_mark):
    '''
    process input from a file of binarized data. This function handles
    getting observations, storing in data structures, building a table
    of mark combination observations, and initializing parameters using
    the segmentation algorithm from Ernst et al. 2010
    '''
    # open file
    fp = open(filename, "r")

    # initialize data structures
    train_obs = list()
    celltype = ""
    chrom = ""
    headers = list()
    train_keys = list()

    # iterate over every line in the file
    for line in fp:
        # get current line, split into array of fields
        obs_at_this_time = np.array(line.rstrip().split('\t'))

        # handle header lines
        if obs_at_this_time[0] != '0' and\
                obs_at_this_time[0] != '1':
            if obs_at_this_time[-1].startswith('chr'):
                celltype = obs_at_this_time[0]
                chrom = obs_at_this_time[1]
            else:
                headers = obs_at_this_time

            continue

        # get only indicies where marks are 1 at this bin
        index_1 = np.where(obs_at_this_time == '1')[0]
        # convert this combination of ON marks into a unique bitstring
        key = bitstring(index_1, len(obs_at_this_time))     

        # check if we have seen this combination before
        if key not in map_to_obs:
            # if not, create a new entry for this observation in 
            # appropriate data structures
            curr_index = len(obs_list)
            # map key to index in observation list
            map_to_obs[key] = curr_index
            # add to list of possible observations
            obs_list.append(curr_index)
            # add to reverse dictionary to get back the list of "ON"
            # marks for this observation
            obs_to_mark[curr_index] = index_1
        
        # append to a list that just keeps track of which observation exists
        # for each bin
        train_keys.append(map_to_obs[key])
        train_obs.append([int(x) for x in obs_at_this_time])

    fp.close()

    # convert to np array
    train_obs = np.array(train_obs)

    # ALPHA is the minimum value for any emission probability. Probabilities
    # cannot be 0 due to log space calculations. This number is taken 
    # from Ernst et al. 2010 supplement. 
    ALPHA = 0.02

    # The following segmentation algorithm is described in Erst et al. 2010
    # initialize p matrix (emission matrix)
    p = np.ones(shape=(num_states, len(headers)))

    for m in range(len(headers)):
        p[0][m] = ALPHA/2

    # initialize d_dat to the binary observation of whether a bin contains
    # a 1 for 2 sequential bins for all bins.
    d_dat = np.zeros(shape=(len(train_obs), len(headers)))
    d_dat = train_obs[1:] * np.roll(train_obs, 1, axis=1)[1:]

    # initialize group array to 0: all bins in same group (or state)
    s_bins = np.zeros(shape=(len(train_obs) - 1), dtype=np.int)

    # iterate over all states + 1
    for i in range(2, num_states + 1):
        # initialize entropy measures
        curr_max_entropy = -1
        curr_max_group = -1
        curr_max_mark = -1

        # iterate over all groups in current i
        for group in range(1, i):
            # iterate over all marks
            for mark in range(len(headers)):
                # set entropy measures of obs = 1 and 0 to 0
                h_1 = 0
                h_0 = 0

                # iterate over each bin in the training data
                for ct_index in range(len(train_obs) - 1):
                    # check current group
                    if s_bins[ct_index] == group:
                        if d_dat[ct_index][mark] == 1:
                            h_1 += 1
                        else:
                            h_0 += 1

                # calculate h values after iterating over all bins
                h_0 = abs(h_0)
                h_1 = abs(h_1)
                h_0 /= len(train_obs - 1)
                h_1 /= len(train_obs - 1)

                entropy = 0
                # calculate current entropy
                if h_0 <= 0 or h_1 <= 0:
                    entropy = 0
                else:
                    entropy = (h_0 + h_1) * math.log2(h_0 + h_1)\
                            - h_0 * math.log2(h_0)\
                            - h_1 * math.log2(h_1)

                # check if we have achieved a higher entropy level
                # if yes, update current variables storing arguments
                if entropy > curr_max_entropy:
                    curr_max_entropy = entropy
                    curr_max_group = group
                    curr_max_mark = mark

        tot_bins = 0
        # iterate over all bins
        for ct_index in range(len(s_bins)):
            # check if each bin is in the current max-entropy group
            # and has 1 for the maximum mark
            # if yes, create a new group and reassign
            if s_bins[ct_index] == curr_max_group and\
                    d_dat[ct_index][curr_max_mark] == 1:
                s_bins[ct_index] = i - 1
                tot_bins += 1

                # update emission matrix
                for mark in range(len(headers)):
                    if d_dat[ct_index][mark] == 1:
                        p[i - 1][mark] += 1

        # update emission matrix after 
        for mark in range(len(headers)):
            if tot_bins != 0:
                p[i - 1][mark] = p[i - 1][mark] / tot_bins * (1 - ALPHA) + (ALPHA / 2)
            else:
                p[i - 1][mark] = ALPHA/2

    
    # get transition probabilities after final group assignments
    A_ij = np.zeros(shape=(num_states, num_states))
    i_count = np.zeros(shape=(num_states))
    key_counts = [None] * num_states

    # iterate over all bins
    for ct_index in range(len(s_bins) - 1):
        # get counts of all groups
        A_ij[s_bins[ct_index]][s_bins[ct_index + 1]] += 1
        i_count[s_bins[ct_index]] += 1

        group = s_bins[ct_index]
        if key_counts[group] == None: 
            key_counts[group] = dict()

        if train_keys[ct_index] not in key_counts[group]:
            key_counts[group][train_keys[ct_index]] = 1
        else:
            key_counts[group][train_keys[ct_index]]+= 1

    # handle the last bin
    last_bin = len(s_bins) - 1
    i_count[s_bins[last_bin]] += 1
    group = s_bins[last_bin]

    if key_counts[group] == None: 
        key_counts[group] = dict()

    if train_keys[last_bin] not in key_counts[group]:
        key_counts[group][train_keys[last_bin]] = 1
    else:
        key_counts[group][train_keys[last_bin]]+= 1

    # initialize emission matrix: combinatorial marks
    emission_probs = np.zeros(shape=(num_states, len(obs_list)))

    # calculate probabilities of emission matrix by proportion
    # of bins that it is present in by initial state
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

        # calculate initial transition probabilities
        for j in range(num_states):
            if i_count[i] == 0:
                A_ij[i][j] = ALPHA / num_states
            else:
                A_ij[i][j] = (1 - ALPHA) * A_ij[i][j] / i_count[i] + ALPHA / num_states

    # return result values: estimated initial values for all parameters 
    # and inputs
    return {
            'train_obs' : train_keys, 
            'B_seg' : p,
            'B' : emission_probs,
            'A' : A_ij,
            'headers' : headers, 
            'cell': celltype
            }

def get_b(obs, state, b_ar):
    '''takes care of case where we see an obs not in training data'''

    if obs >= len(b_ar[state]) or b_ar[state][obs] == 0:
        # return an arbitrary extremely small number instead of 0
        return 0.0000000000001
    else:
        return b_ar[state][obs]

def eln_at(o, pi, A, B, T, STATES):
    '''
    calculate the alpha variables: forward algorithm 
    '''

    eln_a = np.zeros(shape=(T, STATES))

    for i in range(STATES):
        logpi = eln(pi[i])
        logb = eln(get_b(o[0], i, B))

        eln_a[0,i] = elnproduct(logpi, logb)

    for t in range(1, T):
        for j in range(STATES):
            logalpha = LOGZERO
            for i in range(STATES):
                logalpha = elnsum(logalpha, elnproduct(eln_a[t - 1][i], eln(A[i][j])))
            eln_a[t][j] = elnproduct(logalpha, eln(get_b(o[t], j, B)))

    return eln_a

def eln_bt(o, pi, A, B, T, STATES):
    '''
    calculate the beta variables: backwards algorithm
    '''
    eln_b = np.zeros(shape=(T, STATES))

    for i in range(STATES):
        eln_b[T - 1][i] = 0

    for t in range(T - 2, -1, -1):
        for i in range(STATES):
            logbeta = LOGZERO
            for j in range(STATES):
                elna = eln(A[i][j])
                b = eln(get_b(o[t+1], j, B))
                prev_beta = eln_b[t + 1][j]

                logbeta = elnsum( logbeta, elnproduct(elna, elnproduct(b, prev_beta)) )

            eln_b[t][i] = logbeta
    return eln_b

def eln_gammat(eln_at, eln_bt, T, STATES):
    '''
    calculate the gamma variable in log space
    '''

    eln_gamma = np.zeros(shape=(T, STATES))

    for t in range(T):
        normalizer = LOGZERO
        for i in range(STATES):
            eln_gamma[t][i] = elnproduct(eln_at[t][i], eln_bt[t][i])
            normalizer = elnsum(normalizer, eln_gamma[t][i])
        
        for i in range(STATES):
            eln_gamma[t][i] = elnproduct(eln_gamma[t][i], 0 - normalizer)

    return eln_gamma

def eln_xit(eln_at, eln_bt, o, A, B, T, STATES):
    ''' 
    calculate the xi variable in log space
    '''
    eln_xi = np.zeros(shape=(T, STATES, STATES))

    for t in range(T-1):
        normalizer = LOGZERO
        for i in range(STATES):
            for j in range(STATES):
                eln_xi[t][i][j] = elnproduct( eln_at[t][i], elnproduct(eln(A[i][j]), elnproduct(eln(get_b(o[t+1], j, B)), eln_xi[t][i][j] ) ) )

                normalizer = elnsum(normalizer, eln_xi[t][i][j])

        for i in range(STATES):
            for j in range(STATES):
                eln_xi[t][i][j] = elnproduct(eln_xi[t][i][j], 0 - normalizer)

    return eln_xi

def update_pi(eln_gammat, i):
    ''' 
    update the pi vector with re-estimated values
    '''
    return eexp(eln_gammat[0][i])

def update_aij(eln_gammat, eln_xit, i, j, T):
    '''
    update the A matrix with re-estimated values
    '''
    numerator = LOGZERO
    denominator = LOGZERO

    for t in range(T - 1):
        numerator = elnsum(numerator, eln_xit[t][i][j])
        denominator = elnsum(denominator, eln_gammat[t][i])

    return eexp(elnproduct(numerator, 0 - denominator))

def update_beta(eln_gammat, eln_xit, o, vk, j, T):
    ''' 
    update the B matrix with re-estimated values
    '''
    numerator = LOGZERO
    denominator = LOGZERO

    for t in range(T):
        if o[t] == vk:
            numerator = elnsum(numerator, eln_gammat[t][j])

        denominator = elnsum(denominator, eln_gammat[t][j])

    return eexp(elnproduct(numerator, 0 - denominator))

def printem(B, obs_to_mark, marks):
    '''
    print the final emission matrix, assuming we are handling
    combinations of marks as emissions. Must calculate probability
    of each individual mark from combination observations
    to align to ChromHMM emission matrix
    '''
    fp = open("emissions.txt", "w")
    fp.write("\t".join(marks) + "\n")

    # iterate over all states in B matrix
    for i in range(len(B)):
        string = ""
        # initialize probabilities
        mark_probs = np.zeros(len(marks))
        counts = np.zeros(len(marks))
        for j in range(len(B[i])):
            # get the observation probability
            curr_obs_prob = B[i][j]
            # get the combination of marks this probability corresponds to
            curr_marks = obs_to_mark[j]

            # iterate over all of the marks in this combo, aggregate 
            # probabilities
            for c in curr_marks:
                mark_probs[c] += curr_obs_prob
                counts[c] += 1
            #string += str(B[i][j]) 

        # print out normalized probabilities per mark
        for m in range(len(mark_probs)):
            if m != 0:
                string += "\t"
            string += str(mark_probs[m] / counts[m])


        fp.write(string + "\n")
    fp.close()
            
def printtrans(A):
    '''
    print the final transition matrix
    '''
    fp =open("transitions.txt", "w")

    for i in range(len(A)):
        string = ""
        for j in range(len(A[i])):
            if j != 0:
                string += "\t"
            string += str(A[i][j])
        fp.write(string + "\n")

    fp.close()

## MAIN FUNCTION ##
def main():
    # check number of inputs
    if len(sys.argv) < 3:
        print("usage: <num states> <file: training data>", file = sys.stderr)
        exit(2)

    # get inputs from command line arguments
    num_states = int(sys.argv[1])
    train_dat_file = sys.argv[2]

    # initialize emission dictionaries and lists
    obs_list = list()
    map_ems = dict()
    obs_to_mark = dict()

    # process binarized data input and calculate initial values
    # of A, B based on segmentation algorithm from 
    # Erst et al. 2010. Additionally, gives the emission probabilities 
    # in combinatorial emissions, rather than the original method which
    # assumes total independence, though that is calculated 
    # and also returned as result['B_seg']
    result = get_obs(train_dat_file, map_ems, obs_list, num_states, obs_to_mark)

    # assign important variables from output
    train_obs = result['train_obs']

    tot_time = len(train_obs)
    num_em = len(obs_list)

    B = result['B']   
    A = result['A']

    marks = result['headers']
    cell = result['cell']

    # initialize pi vector randomly
    pi = np.zeros(shape=(num_states))
    
    for i in range(num_states):
        pi[i] = random.random()

    # set stopping boolean variable
    stop = False
    # initialize likelihood variable
    prev_likelihood = None

    print("Done initializing")

    # repeat Baum-Welch algorithm until convergence
    while not stop:
        ## E Step ##

        # forward-backward algorithm
        alphas = eln_at(train_obs, pi, A, B, tot_time, num_states)
        print("Got alphas")

        betas = eln_bt(train_obs, pi, A, B, tot_time, num_states)
        print("Got betas")

        # get xi variable using the forward and backward variables
        xis = eln_xit(alphas, betas, train_obs, A, B, tot_time, num_states)
        print("Got xis")

        # get gammas using forward and backward variables
        gammas = eln_gammat(alphas, betas, tot_time, num_states)
        print("Got gammas")
        
        ## M Step ##

        # update pi
        for s in range(num_states):
            pi[s] = update_pi(gammas, s)

        print("Updated pi")

        # update A
        for i in range(num_states):
            for j in range(num_states):
                A[i][j] = update_aij(gammas, xis, i, j, tot_time)

        print("Updated A")

        # update B
        for j in range(num_states):
            for v in range(num_em):
                B[j][v] = update_beta(gammas, xis, train_obs, v, j, tot_time)
        
        print("Updated B")

        # calculate log likelihood -- just the sum of alphas since we're
        # already in log space
        curr_likelihood = None
        for s in range(num_states):
            if curr_likelihood == None:
                curr_likelihood = alphas[tot_time - 1][s]
            else:
                curr_likelihood = elnsum(curr_likelihood, alphas[tot_time - 1][s])

        print("Calculated log likelihood")

        # go to next loop if this is the first run of Baum-Welch
        if prev_likelihood == None:
            prev_likelihood = curr_likelihood
            continue
        else:
            # calculate difference of log likelihoods between 
            # this and previous run
            diff = curr_likelihood - prev_likelihood
            print("diff:", diff)

            # if we meet the convergence criteria, stop training
            if diff <= .001: 
                stop = True

            prev_likelihood = curr_likelihood


    # print emissions and transition results
    printem(B, obs_to_mark, marks)
    printtrans(A)

## Call Main Function ##    
main()
