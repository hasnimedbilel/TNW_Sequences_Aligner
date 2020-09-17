import pandas as pd
import TNW_aligner
import string
import itertools

data = pd.read_csv("results/example_data.csv")

aligner = TNW_aligner.Aligner()

# Now we will define the scoring scheme : match, mismatch, alphabet ...
# We initialized the values with those defined in the TNW paper, we can change these values based on the use case 
# we can use grid search for example to find the best combination of Parameters : 
# N.B : Grid Search might be very time consuming !!!

match = 1.
mismatch = -1.1
s = {'11': match}
alphabet = string.ascii_uppercase
# Preparing all the possible combinations of Sequences couples : 
comb = list(itertools.product(alphabet, repeat=2))

for pairs in comb:
    if pairs[0] == pairs[1]:
        s[pairs[0] + pairs[1]] = match
    else:
        s[pairs[0] + pairs[1]] = mismatch

align_result = aligner.TNW(t_encoded_data=data, gap_penalty=1, t_penalty=0.5, scoring_scheme=s)
print(align_result.shape)
print(align_result)

# align_result.to_csv("results/TNW_align_results.csv", index=False)