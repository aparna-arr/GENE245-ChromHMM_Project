import sys
import numpy as np
import math
if len(sys.argv) < 2:
    print("usage: python3 init_params.py <binarized ChromHMM file>", file=sys.stderr)
    sys.exit(2)

binfile = sys.argv[1]

fp = open(binfile, "r")
celltype = ""
chrom = ""
headers = []
bin_dat = []
for line in fp:
    line_ar = line.rstrip().split('\t')

    if line_ar[-1].startswith('chr'):
        # first line of file
        celltype = line_ar[0]
        chrom = line_ar[1]
        continue
    elif line_ar[0] != '0' and line_ar[0] != '1':
        headers = line_ar
        continue

    bin_dat.append([ int(x) for x in line_ar ])

fp.close()

bin_dat = np.array(bin_dat)
#print(bin_mat)

ALPHA = 0.02
NUM_STATES = 50

p = np.zeros(shape=(NUM_STATES - 1, len(headers)))
p[0] = p[0] + ALPHA/2


# check that this actually works
d_dat = np.zeros(shape=(len(bin_dat), len(bin_dat[0])))
d_dat = bin_dat[1:] * np.roll(bin_dat, 1, axis=1)[1:]

s_bins = np.ones(shape=(len(bin_dat)-1), dtype=np.int)

#u_entropy = np.zeros(shape(len(headers), len(NUM_STATES-2)))

for i in range(2,NUM_STATES):
    print("i is",i)
    curr_max_entropy = -1
    curr_max_group = -1
    curr_max_mark = -1

    for group in range(1,i):
        for mark in range(len(headers)):
            h_1 = 0
            h_0 = 0

            for ct_index in range(len(bin_dat)-1):
                if s_bins[ct_index] == group:
                    if d_dat[ct_index][mark] == 1:
                        h_1 += 1
                    else:
                        h_0 += 1

            h_0 = abs(h_0)
            h_1 = abs(h_1)
            h_0 /= len(bin_dat-1)
            h_1 /= len(bin_dat-1)

            #print(h_0, h_1)

            entropy = 0
            if h_0 <= 0 or h_1 <= 0:
                entropy = 0
            else:
                entropy = (h_0 + h_1) * math.log2(h_0 + h_1) \
                    - h_0 * math.log2(h_0) \
                    - h_1 * math.log2(h_1)

            #print("Entropy",entropy)

            if entropy > curr_max_entropy:
                curr_max_entropy = entropy
                curr_max_group = group
                curr_max_mark = mark
    
    print("curr_max_entropy is", curr_max_entropy, "group:", curr_max_group, "mark:", curr_max_mark)
    tot_bins = 0
    for ct_index in range(len(s_bins)):
        if s_bins[ct_index] == curr_max_group and \
                d_dat[ct_index][curr_max_mark] == 1:
            s_bins[ct_index] = i
            tot_bins += 1
            for mark in range(len(headers)):
                if d_dat[ct_index][mark] == 1: 
                    p[i - 1][mark] += 1

    p[i-1] = p[i - 1] / tot_bins * (1 - ALPHA) + (ALPHA / 2)

    
