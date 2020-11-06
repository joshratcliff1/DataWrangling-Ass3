# ============================================================================
# Record linkage software for the COMP3430/COMP8430 Data Wrangling course, 
# 2020.
# Version 1.0
#
# Copyright (C) 2020 the Australian National University and
# others. All Rights Reserved.
#
# =============================================================================

"""Main module for linking records from two files.

   This module calls the necessary modules to perform the functionalities of
   the record linkage process.
"""

# =============================================================================
# Import necessary modules (Python standard modules first, then other modules)

import time

import loadDataset
import blocking
import comparison
import classification
import evaluation
import saveLinkResult



# =============================================================================
#
'''Step 1: Load the two datasets from CSV files'''


def load(datasetA_name, datasetB_name, truthfile_name, rec_id_col, attr_list,header_line):
    start_time = time.time()

    rec_idA_col = rec_id_col
    rec_idB_col = rec_id_col
    headerA_line = header_line
    headerB_line = header_line
    attrA_list = attr_list
    attrB_list = attr_list

    recA_dict = loadDataset.load_data_set(datasetA_name, rec_idA_col, attrA_list, headerA_line)
    recB_dict = loadDataset.load_data_set(datasetB_name, rec_idB_col, attrB_list, headerB_line)

    # Load data set of true matching pairs
    true_match_set = loadDataset.load_truth_data(truthfile_name)

    # Get the number of total record pairs to compared if no blocking used
    all_comparisons = len(recA_dict) * len(recB_dict)

    loading_time = time.time() - start_time

    # Return record dictionaries
    return recA_dict, recB_dict, true_match_set, all_comparisons, loading_time


'''Step 2: Block the datasets'''


def block(recA_dict,recB_dict, block_method, block_attribute):
    # The list of attributes to use for blocking (all must occur in the above attribute lists)
    blocking_attrA_list = block_attribute
    blocking_attrB_list = block_attribute

    start_time = time.time()

    # Select one blocking technique
    if block_method == 'none':
        # No blocking (all records in one block)
        blockA_dict = blocking.noBlocking(recA_dict)
        blockB_dict = blocking.noBlocking(recB_dict)

    elif block_method == 'simple':
        # Simple attribute-based blocking
        blockA_dict = blocking.simpleBlocking(recA_dict, blocking_attrA_list)
        blockB_dict = blocking.simpleBlocking(recB_dict, blocking_attrB_list)

    elif block_method == 'phonetic':
        # Phonetic (Soundex) based blocking
        blockA_dict = blocking.phoneticBlocking(recA_dict, blocking_attrA_list)
        blockB_dict = blocking.phoneticBlocking(recB_dict, blocking_attrB_list)

    elif block_method == 'SLK':
        # Statistical linkage key (SLK-581) based blocking
        fam_name_attr_ind = 3
        giv_name_attr_ind = 1
        dob_attr_ind = 6
        gender_attr_ind = 4

        blockA_dict = blocking.slkBlocking(recA_dict, fam_name_attr_ind, giv_name_attr_ind,
                                           dob_attr_ind, gender_attr_ind)
        blockB_dict = blocking.slkBlocking(recB_dict, fam_name_attr_ind, giv_name_attr_ind,
                                           dob_attr_ind, gender_attr_ind)
    else:
        print("Blocking Error")

    blocking_time = time.time() - start_time
    print("blocking time:", blocking_time)

    # Print blocking statistics
    blocking.printBlockStatistics(blockA_dict, blockB_dict)

    # Return Block dictionaries
    return blockA_dict, blockB_dict, blocking_time


'''Step 3: Compare the candidate pairs'''


def compare(blockA_dict, blockB_dict, recA_dict, recB_dict, approx_comp_funct_list):
    # The list of tuples (comparison function, attribute number in record A,
    # attribute number in record B)
    #


    start_time = time.time()

    sim_vec_dict = comparison.compareBlocks(blockA_dict, blockB_dict, recA_dict, recB_dict, approx_comp_funct_list)

    comparison_time = time.time() - start_time
    print("comparison time:" ,comparison_time)

    # Return similarity vector dictionary and time
    return sim_vec_dict, comparison_time


'''Step 4: Classify the candidate pairs'''


def classify(sim_vec_dict, classification_mode, threshold, approx_comp_funct_list, weights):
    start_time = time.time()

    if classification_mode == 'exact':
        # Exact matching based classification
        class_match_set, class_nonmatch_set = classification.exactClassify(sim_vec_dict)

    elif classification_mode == 'similarity':
        # Similarity threshold based classification
        sim_threshold = threshold
        class_match_set, class_nonmatch_set = classification.thresholdClassify(sim_vec_dict, sim_threshold)

    elif classification_mode == 'min_sim':
        # Minimum similarity threshold based classification
        min_sim_threshold = threshold
        class_match_set, class_nonmatch_set = classification.minThresholdClassify(sim_vec_dict, min_sim_threshold)

    elif classification_mode == 'weighted':
        # Weighted similarity classification
        sim_threshold = threshold
        class_match_set, class_nonmatch_set = classification.weightedSimilarityClassify(sim_vec_dict,
                                                                                        weights, sim_threshold)
    elif classification_mode == 'tree':
        # A supervised decision tree classifier
        class_match_set, class_nonmatch_set = classification.supervisedMLClassify(sim_vec_dict, true_match_set)
    else:
        print("classification error")

    classification_time = time.time() - start_time
    print("classification time:", classification_time)

    # Return matching and non matching sets and time taken
    return class_match_set, class_nonmatch_set, classification_time


''' Step 5: Evaluate the Blocking'''


def evaluate_block(sim_vec_dict, recA_dict, recB_dict, true_match_set, all_comparisons):
    # Get the number of record pairs compared
    num_comparisons = len(sim_vec_dict)

    # Get the list of identifiers of the compared record pairs
    cand_rec_id_pair_list = sim_vec_dict.keys()

    # Blocking evaluation
    rr = evaluation.reduction_ratio(num_comparisons, all_comparisons)
    pc = evaluation.pairs_completeness(cand_rec_id_pair_list, true_match_set)
    pq = evaluation.pairs_quality(cand_rec_id_pair_list, true_match_set)

    print('Blocking evaluation:')
    print('  Reduction ratio:    %.6f' % rr)
    print('  Pairs completeness: %.6f' % pc)
    print('  Pairs quality:      %.6f' % pq)
    print('')


''' Step 5: Evaluate the Linking'''


def evaluate_link(class_match_set, class_nonmatch_set, true_match_set, all_comparisons):
    # Linkage evaluation
    linkage_result = evaluation.confusion_matrix(class_match_set, class_nonmatch_set, true_match_set, all_comparisons)

    accuracy = evaluation.accuracy(linkage_result)
    precision = evaluation.precision(linkage_result)
    recall = evaluation.recall(linkage_result)
    fmeasure = evaluation.fmeasure(linkage_result)

    print('Linkage evaluation:')
    print('  Accuracy:    %.6f' % (accuracy))
    print('  Precision:   %.6f' % (precision))
    print('  Recall:      %.6f' % (recall))
    print('  F-measure:   %.6f' % (fmeasure))
    print('')


''' Step 6: Calculate the time taken'''

def time_taken(loading_time, blocking_time, comparison_time, classification_time):
    linkage_time = loading_time + blocking_time + comparison_time + classification_time
    print('Total runtime required for linkage: %.3f sec' % linkage_time)



''' Run the actual program'''
# Variable names for loading datasets
dataset_A_list = ['datasets_test/clean/A-1000.csv','datasets_test/little/A-1000.csv',
                  'datasets_test/very/A-1000.csv', 'datasets_test/very2/A-10000.csv',
                  'Dataset_generation/data_wrangling_rl1_2020_u7199704.csv']
dataset_B_list = ['datasets_test/clean/B-1000.csv','datasets_test/little/B-1000.csv',
                  'datasets_test/very/B-1000.csv', 'datasets_test/very2/B-10000.csv',
                  'Dataset_generation/data_wrangling_rl2_2020_u7199704.csv']
truth_list = ['datasets_test/clean/true-1000.csv','datasets_test/little/true-1000.csv',
              'datasets_test/very/true-1000.csv', 'datasets_test/very2/true-10000.csv',
              'Dataset_generation/data_wrangling_rlgt_2020_u7199704.csv']

# Default Time values
loading_time, blocking_time, comparison_time, classification_time = 0,0,0,0




# The list of attributes to be used either for blocking or linking
#  0: rec_id
#  1: first_name
#  2: middle_name
#  3: last_name
#  4: gender
#  5: current_age
#  6: birth_date
#  7: street_address
#  8: suburb
#  9: postcode
# 10: state
# 11: phone
# 12: email



'''LOADING'''
# Loading Parameters
attr_list = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12]
header_line = True
rec_id_col = 0

# Assignment Datasets and Truth File
dataset = 2
datasetA_name = dataset_A_list[dataset]
datasetB_name = dataset_B_list[dataset]
truthfile_name = truth_list[dataset]

# Create the record dictionaries
recA_dict, recB_dict, true_match_set, all_comparisons, loading_time = load(datasetA_name, datasetB_name, truthfile_name,
                                                                           rec_id_col, attr_list, header_line)

'''BLOCKING'''
# Blocking Parameters
# blocking_options: ['none','simple','phonetic','SLK']
block_method = 'phonetic'
block_attribute = [7]

# Create the blocking dictionaries
blockA_dict, blockB_dict, blocking_time = block(recA_dict,recB_dict, block_method, block_attribute)

'''COMPARISON'''
# Comparison parameters
approx_comp_funct_list = [(comparison.dice_comp, 1, 1),  # First name
                            (comparison.dice_comp, 2, 2),  # Middle name
                            (comparison.jaro_winkler_comp, 3, 3),  # Last name
                            #(comparison.jaccard_comp, 7, 7),  # Address
                            (comparison.edit_dist_sim_comp, 8, 8),  # Suburb
                            (comparison.dice_comp, 6, 6)]  # Birthdate
                            #(comparison.dice_comp, 11, 11),  # Phone
                            #(comparison.jaccard_comp, 12, 12)]  # Email
                            #(comparison.jaccard_comp, 10, 10)]  # State

'[jaccard_comp, dice_comp, jaro_winkler_comp, bag_dist_sim_comp, edit_dist_sim_comp, exact_comp]'

# Create the similarity vector test typing here
sim_vec_dict, comparison_time = compare(blockA_dict, blockB_dict, recA_dict, recB_dict,
                                        approx_comp_funct_list)

'''BLOCKING EVALUATION'''
# Perform blocking evaluation
evaluate_block(sim_vec_dict, recA_dict, recB_dict, true_match_set, all_comparisons)

'''CLASSIFICATION'''
# Parameters
# classification_options: ['exact', 'similarity', 'min_sim', 'weighted', 'tree']
threshold = 1
classification_mode = 'similarity'
weight_vec = [1, 1, 1, 1, 1, 1] # First name, middle_name, last_name, Address, suburb, state

# Create the classified list of candidate pairs
class_match_set, class_nonmatch_set, classification_time = classify(sim_vec_dict, classification_mode,
                                                                    threshold, approx_comp_funct_list,weight_vec)

'''CLASSIFICATION EVALUATION'''
# Perform linking evaluation
evaluate_link(class_match_set, class_nonmatch_set, true_match_set, all_comparisons)

'''CALCULATE TIME'''
# Calculate time taken
time_taken(loading_time, blocking_time, comparison_time, classification_time)


# '''PRINT RESULTS TO CSV'''
# file_name = 'savedResults/similarity.csv'
#
# saveLinkResult.save_linkage_set(file_name,class_match_set)