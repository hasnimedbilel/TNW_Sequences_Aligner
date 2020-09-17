import itertools
import pandas as pd
import numpy as np

# This Class will return a DataFrame containing Similarity scores, Distances, original
# sequences and aligned sequences.

class Aligner:

    def __init__(self):
        pass

    # Initialisation of the score matrix
    @staticmethod
    def score_initialisation(rows, cols, gap):

        score = np.zeros((rows, cols), float)

        for i in range(rows):
            score[i][0] = -i * gap
        for j in range(cols):
            score[0][j] = -j * gap

        return score

    # Initialisation of the traceback matrix : this matrix will be used to
    # get the optimal alignement of the sequences.
    @staticmethod
    def traceback_initialisation(rows, cols):

        traceback = np.zeros((rows, cols))

        # end of path top left corner
        traceback[0][0] = -1

        # going up
        for i in range(1, rows):
            traceback[i][0] = 1

            # going left
        for j in range(1, cols):
            traceback[0][j] = 2

        return traceback

    # Initialisation of the TR matrix
    @staticmethod
    def TR_initialisation(rows, cols, seq2):

        TR = np.zeros((rows, cols))

        # end of path top left corner
        TR[0][0] = 0

        # going up
        for i in range(1, rows):
            TR[i][0] = TR[i - 1][0] + float(seq2[i - 1][0])

            # going left
        for j in range(1, cols):
            TR[0][j] = 0

        return TR

    # Initialisation of the TC matrix
    @staticmethod
    def TC_initialisation(rows, cols, seq1):

        TC = np.zeros((rows, cols))

        # end of path top left corner
        TC[0][0] = 0

        # going up
        for i in range(1, rows):
            TC[i][0] = 0

        # going left
        for j in range(1, cols):
            TC[0][j] = TC[0][j - 1] + float(seq1[j - 1][0])

        return TC

    # calculation of the scores and filling the traceback matrix
    @staticmethod
    def calculate_scores(score, traceback, rows, cols, seq1, seq2, TR, TC, gap, T, s):

        for i in range(1, rows):
            for j in range(1, cols):
                if i - 1 == 0 and j - 1 == 0:
                    tp = 0
                else:
                    num = abs(float(seq2[i - 1][0]) + TR[i - 1][j - 1] - float(seq1[j - 1][0]) - TC[i - 1][j - 1])
                    den = max(float(seq2[i - 1][0]) + TR[i - 1][j - 1], float(seq1[j - 1][0]) + TC[i - 1][j - 1])
                    # temporal penalty function
                    tp = T * (num / (den + 1e-8))

                choice1 = score[i - 1][j - 1] + s[(seq1[j - 1][1] + seq2[i - 1][1])] - tp  # diagonal
                choice2 = score[i - 1][j] - gap  # up
                choice3 = score[i][j - 1] - gap  # left
                choices = [choice1, choice2, choice3]
                score[i][j] = max(choices)

                # update traceback matrix
                traceback[i][j] = choices.index(max(choices))

                if traceback[i][j] == 0:
                    TR[i][j] = 0
                    TC[i][j] = 0

                elif traceback[i][j] == 1:
                    TR[i][j] = TR[i - 1][j] + float(seq2[i - 1][0])
                    TC[i][j] = TC[i - 1][j]

                elif traceback[i][j] == 2:
                    TR[i][j] = TR[i][j - 1]
                    TC[i][j] = TC[i][j - 1] + float(seq1[j - 1][0])

    # deducing the alignment from the traceback matrix
    @staticmethod
    def alignment(traceback, rows, cols, seq1, seq2):

        aseq1 = ''
        aseq2 = ''

        # We reconstruct the alignment into aseq1 and aseq2,
        j = cols - 1
        i = rows - 1
        while i > 0 and j > 0:

            # going diagonal
            if traceback[i][j] == 0:
                aseq1 = seq1[j - 1][1] + aseq1
                aseq2 = seq2[i - 1][1] + aseq2
                i -= 1
                j -= 1

            # going up -gap in sequence1 (top one)
            elif traceback[i][j] == 1:
                aseq1 = '_' + aseq1
                aseq2 = seq2[i - 1][1] + aseq2
                i -= 1
            # going left -gap in sequence2 (left one)
            elif traceback[i][j] == 2:
                aseq1 = seq1[j - 1][1] + aseq1
                aseq2 = '_' + aseq2
                j -= 1
            else:
                # we should never get here !!!
                print('ERROR')
                i = 0
                j = 0
                aseq1 = 'ERROR'
                aseq2 = 'ERROR'
                seq1 = 'ERROR'
                seq2 = 'ERROR'

        while i > 0:
            # If we hit j==0 before i==0 we keep going in i (up).
            aseq1 = '_' + aseq1
            aseq2 = seq2[i - 1][1] + aseq2
            i -= 1

        while j > 0:
            # If we hit i==0 before j==0 we keep going in j (left).
            aseq1 = seq1[j - 1][1] + aseq1
            aseq2 = '_' + aseq2
            j -= 1

        aligned = [aseq1, aseq2]

        return aligned

    @staticmethod
    def convert_similarity_to_distance_matrix(similarity_matrix):
        # This method is super important since Clustering methods only accepts distance matrices.
        # That means that the similarities are considered distances which is completely the
        # the opposite.
        # So we need to do this transformation before moving to the Clustering part !!!

        # we do this "negation" so that sequences with maximum similarities will
        # have Zero distance.
        distance_matrix = -similarity_matrix
        distance_matrix = distance_matrix + abs(distance_matrix[distance_matrix.idxmin()])

        return distance_matrix

    def TNW(self, t_encoded_data, gap_penalty, t_penalty, scoring_scheme):
        # t_encoded_data :  the encoded data containing one entry per patient and the shape :
        # ('Id_Patient' & 'treatment_encoded')

        # gap_penalty: As presented in the TNW Paper, it's the gap penalty : penalty on this "_" .
        # Should be fixed positive, it will be considered negative penalty later !!!

        # t_penalty: It's the Temporal Penalty imposed. The higher the T the higher the penalty
        # on the difference in time between sequences.
        # Should be fixed positive, it will be considered negative penalty later !!!

        # scoring_scheme: As presented in the TNW Paper : it's the predifined scoring scheme.

        # get all the possible combinations between the patients to perform alignment
        patient_comb = list(itertools.combinations(t_encoded_data['Id_Patient'], 2))

        t_encoded_data.set_index('Id_Patient', inplace=True)

        # Results is the data frame containing the scores and the alignments :
        results = pd.DataFrame(patient_comb, columns=['patient1', 'patient2'])
        list_sequences_aligned = []
        list_scores = []
        list_sequences = []

        # analyze every possible combination between patients
        for patient_pair in patient_comb:
            # get the sequences to be aligned
            seq1_encoded = t_encoded_data.loc[patient_pair[0], 'treatment_encoded']
            seq2_encoded = t_encoded_data.loc[patient_pair[1], 'treatment_encoded']

            list_sequences.append([seq1_encoded, seq2_encoded])
            # split the sequences
            aux1 = seq1_encoded.split(",")
            aux2 = seq2_encoded.split(",")

            seq1 = []
            seq2 = []

            for seq in aux1:
                seq1.append(seq.rsplit(".", 1))

            for seq in aux2:
                seq2.append(seq.rsplit(".", 1))

            cols = len(seq1) + 1
            rows = len(seq2) + 1

            score = self.score_initialisation(rows, cols, gap_penalty)
            traceback = self.traceback_initialisation(rows, cols)
            TR = self.TR_initialisation(rows, cols, seq2)
            TC = self.TC_initialisation(rows, cols, seq1)

            self.calculate_scores(score, traceback, rows, cols, seq1, seq2, TR, TC, gap_penalty, t_penalty,
                                  scoring_scheme)
            sequences_aligned = self.alignment(traceback, rows, cols, seq1, seq2)

            list_sequences_aligned.append(sequences_aligned)

            list_scores.append(score[rows - 1][cols - 1])

        results['sequences'] = pd.Series(list_sequences)
        results['sequences_aligned'] = pd.Series(list_sequences_aligned)
        results['score'] = pd.Series(list_scores)
        results["distance"] = self.convert_similarity_to_distance_matrix(results["score"])
        return results
