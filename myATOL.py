#!/usr/bin/python
# 
# -> baseline script <-
#
# This program takes as input tokenized text of onion sites, the
# labeling of onion sites with categories (both train and test data),
# a seed list of keywords for each category, and an index file having
# the titles for each onion site. It outputs weights associated with
# new keywords for each category (higher weight => keyword is more
# important for category). It also computes probabilities of
# categories for the test set, and computes accuracy of label
# prediction based on those probabilities.
#
# The program takes 7 arguments: (a) a train label file, which has a
# list of onion sites for a particular category (one site per line),
# (b) a word_group directory having multiple files, one per onion --
# each file has the count of words in that onion site, (c) a seed list
# of keywords for each category, (d) an index file having title of
# onions and existing onion labelings, (e) test label file.
# (f) stop word list, (g) baseline labels (for discovery).
#
# Usage: python enhance_keywords.py -l train_label_file \
#                                   -d wordgrp_dir \
#                                   -k keywords_file \
#                                   -i index_file \
#                                   -t test_label_file \
#                                   -s stopwords_file \
#                                   -b baseline_label_file \
#                                   -m mode
# 
# Example: python enhance_keywords.py -l train.labels -d WORD_GRP2 -k KeywordGroups.txt -i MASTER.Onion.Index.csv -t test.labels -s stopwords.txt -b weapons_outDead.txt
#
# Get accuracy results on Feb 19 data: python enhance_keywords.py -l train.labels -d WORD_GRP2 -k KeywordGroups.txt -i MASTER.Onion.Index.csv -t test.labels -s stopwords.txt -b weapons_outDead.txt -m "accuracy" > accuracy_20160219.output
#
# Get filtering results on Feb 19 data: python enhance_keywords.py -l train.labels -d WORD_GRP2 -k KeywordGroups.txt -i MASTER.Onion.Index.csv -t test.labels -s stopwords.txt -b weapons_outDead.txt -m "filtering" > filtering_20160219.output
#
# Get discovery results on Mar 2 data: python enhance_keywords.py -l train.labels -d WORD_GRP3 -k KeywordGroups_03022016.txt -i MASTER.Onion.Index_03022016.csv -t test.labels -s stopwords.txt -b wordGrp3_Results_03022016.dat -m "discovery" > discovery_20160302.output
#
# Format of each line of train_label_file or test_label_file:
# Onion,Category
#
# Format of each line of each wordgrp file in the wordgrp_dir:
# Word,Count,NumPages,Ratio
#
# Format of each line in the keywords_file:
# Category, <comma-separated list of keywords>
#
# Format of each line of index_file:
# Onion,Real Port,Title,Link Count,Language,First Seen,Last Seen,Last Status,Probe Status,Contains CP,Keys
#
# Format of stopwords file:
# Word
#
# Format of baseline label file:
# Onion, Category[<comma-separated list of matching keywords>] 
#


import getopt, glob, math, os, sys
from collections import defaultdict

from time import time

kEpsilon = 0.0000000001  # Small number to add to denominator, to prevent /0.

kMinKeywordLength = 3  # Prune out keywords with less than this lengh.
kMinDocSize = 10  # Onions with less than this #unique words are ignored.
kMaxVecSize = 50  # Max size of keyword vector

# Multiplier weights, used to accumulate score in 'count' variable in
# ProcessFilesInCategory function
#
kTitleMultiplier = 10  # Scaling up factor for title keywords.
kKeywordMultiplier = 100  # Scaling up factor for existing keywords.
kPageMultiplier = 1  # Scaling up factor for pages
kRatersMultiplier = 1.5  # Scaling up factor for human labels

# Main driver function.
def main(argv):

    t0 = time()
    train_label_file, wordgrp_dir, keywords_file, index_file, test_label_file, stopwords_file, baseline_label_file, mode, dedup, practical_dir = ProcessArguments(argv)
    (L, K, T, M, Mt, data, test, H, B) = CreateHashes(train_label_file,
                                                      wordgrp_dir, 
                                                      keywords_file,
                                                      index_file,
                                                      test_label_file,
                                                      stopwords_file,
                                                      baseline_label_file
    )
    if practical_dir != None:
        practical_data = CreateData(practical_dir)

    # Deduplicate if necessary.
    if dedup:
        print('\nDeduplicating data and test.')
        print('Data size before deduplication: ' + str(len(list(data.keys()))))
        data = DedupData(data, T)
        print('Data size after deduplication: ' + str(len(list(data.keys()))))
        print('Test size before deduplication: ' + str(len(list(test.keys()))))
        test = DedupTest(test, T)
        print('Test size after deduplication: ' + str(len(list(test.keys()))))

    # Unique categories in labeled data.
    categories = list(set([item for sublist in list(L.values()) for item in sublist]))

    ##### Original data #####

    ### Phase 1
    print('\n\n==== Running PHASE 1 (Keywords) ====')
    print('\n\n==== PHASE 1: New keyword list ====')
    # Compute baseline and TFICF keywords.
    K1 = TransformHashFormat(K)
    print(("\n Pre-processing & Dataset loading done in %0.3fs." % (time() - t0)))
    t0 = time()
    keywords = ComputeTFICF(M, Mt, K, categories)
    print(("\n TFICF Computation done in %0.3fs." % (time() - t0)))

    ### Phase 2
    if mode == 'accuracy':
        print('\n\n==== Running PHASE 2 (Accuracy) ====')
        # Results with baseline keywords.
        print('\n\n==== PHASE 2: Probability estimates using baseline keyword list ====')
        t0 = time()
        probs = RunInference(data, test, K1, list(M.keys()), T, 'baseline: ')
        print(("\n Baseline running time: %0.3fs." % (time() - t0)))
        # Results with TFICF keywords.
        print('\n\n==== PHASE 2: Probability estimates using ATOL keyword list ====')
        t0 = time()
        probs = RunInference(data, test, keywords, categories, T, 'atol: ')
        print(("\n ATOL running time: %0.3fs." % (time() - t0)))

    ### Phase 3
    if mode == 'filtering':
        print('\n\n==== Running PHASE 3 (Filtering) ====')
        # Results on subset of data labeled as DRUGS by original keywords.
        print('\n\n==== PHASE 3: Results on data subset labeled as DRUGS by baseline algorithm ====')
        probsD = RunInferenceOnLabel(data, H, keywords, categories, T, 'DRUGS')
        # Results on subset of data labeled as HACKER by original keywords.
        print('\n\n==== PHASE 3: Results on data subset labeled as HACKER by baseline algorithm ====')
        probsH = RunInferenceOnLabel(data, H, keywords, categories, T, 'HACKER')
        # Results on subset of data labeled as Weapons by original keywords.
        print('\n\n==== PHASE 3: Results on data subset labeled as Weapons by baseline algorithm ====')
        probsW = RunInferenceOnLabel(data, H, keywords, categories, T, 'Weapons')

    ### Phase 4
    if mode == 'discovery':
        print('\n\n==== Running PHASE 4 (Discovery) ====')
        print('\n\n==== PHASE 4: From full data we find onions where Weapons has high probability ====')
        probsW_all = RunInferenceDiff(data, B, keywords, categories, T, test, 'Weapons', 0.5)

    if mode == 'practical':
        print('\n\n==== Running PHASE 5 (Practical) ====')
        print('\n\n==== PHASE 5: Categorizing from new onion website ====')
        RunPracticalDiff(practical_data, keywords, categories)



# Dedup data defaultdict.
def DedupData(data, T):
    dedup_data = defaultdict(lambda:defaultdict(int))
    # Create transpose of title hash T.
    Tt = {}
    for (onion, title) in list(T.items()):
        title_str = ' '.join(title)
        if title_str in Tt:
            Tt[title_str].append(onion)
        else:
            Tt[title_str] = [onion]
    # Create map of title to onion with largest word list.
    Tmax = {}
    for (title_str, onions) in list(Tt.items()):
        selected_onion = 0
        max_value = 0
        for onion in onions:
            if len(data[onion]) > max_value:
                max_value = len(data[onion])
                selected_onion = onion
        Tmax[selected_onion] = title_str
    for (onion, val) in list(data.items()):
        if onion in Tmax:
            dedup_data[onion] = val
    return dedup_data


# Dedup test hash.
def DedupTest(test, T):
    dedup_test = {}
    testTt = {}
    for onion in list(test.keys()):
        add_onion = True
        if onion in T:
            title_str = ' '.join(T[onion])
            if title_str in testTt:
                add_onion = False
            else:
                testTt[title_str] = onion
        if add_onion:
            dedup_test[onion] = test[onion]
    return dedup_test


# Add 1.0 weights to the list in the hash val, for format compatibility.
def TransformHashFormat(K):
    K1 = {}
    for key in list(K.keys()):
        lst = []
        for val in K[key]:
            lst.append((val, 1.0))
        K1[key] = lst
    # print 'K = ' + str(K)
    # print 'K1 = ' + str(K1)
    return K1


# This function returns 9 hashes:
#   1) L, mapping onion -> category list. (From train_label_file)
#   2) T, mapping onion -> list of title words. (From index_file, limited to onions in train_label_file)
#   3) K, mapping category -> list of keywords. (From keywords_file, limited to categories in train_label_file)
#   4) M, mapping category x keyword -> count. (From words, train_label_file, T and K).
#   5) Mt, Mt is the transpose of M. 
#   6) data, mapping onion x keyword -> count. (From words)
#   7) test, mapping onion -> category list. (From test_label_file)
#   8) H mapping onion -> category list. (From index_file)
#   9) B mapping onion -> category list. (From baseline label file)
def CreateHashes(train_label_file, wordgrp_dir, keywords_file,
                 index_file, test_label_file, stopwords_file,
                 baseline_label_file):
    L = CreateL(train_label_file)
    test = CreateL(test_label_file)
    K = CreateK(keywords_file, L)
    data = CreateData(wordgrp_dir)
    S = CreateS(stopwords_file)
    T = CreateT(index_file, S)
    H = CreateH(index_file)
    B = CreateB(baseline_label_file)
    (M, Mt) = CreateM(wordgrp_dir, L, T, K, S, H, test)
    return (L, K, T, M, Mt, data, test, H, B)


# Create Hash S mapping word -> 1.
def CreateS(stopwords_file):
    S = {}
    index = os.getcwd() + '/' + stopwords_file
    #print 'Processing stopwords file: ' + stopwords_file
    f = open(index, 'r')
    lines = f.readlines()
    for line in lines:
        # Ignore comments.
        if line.startswith('#'):
            continue
        stopword = line.rstrip('\n')
        # print '--> Line: ' + stopword
        if stopword not in S:
            S[stopword] = 1
#    print 'S hash is:'
#    for key in S.keys():
#        print 'Stopword: ' + key
    return S


# Create Hash L mapping onion -> category list.
def CreateL(label_file):
    L = {}
    index = os.getcwd() + '/' + label_file
    #print 'Processing label file: ' + label_file
    f = open(index, 'r')
    lines = f.readlines()
    for line in lines:
        # Ignore comments.
        if line.startswith('#'):
            continue
        stripped_line = line.rstrip('\n')
        # print '--> Line: ' + stripped_line
        tokens = stripped_line.split(',')
        onion = tokens[0]
        cat_label = tokens[1]
        if len(cat_label) <= 1:
            # print 'Ignoring onion ' + onion + ' with empty categories.'
            continue
        else:
            if len(cat_label) > 1:
                # print 'Adding label: ' + cat_label + ' for onion: ' + onion
                if onion in L:
                    L[onion].append(cat_label)
                else:
                    L[onion] = [cat_label]
            # print 'Onion: ' + onion + ', category list: ' + str(L[onion])
#    print 'L hash is:'
#    for (key, val) in L.items():
#        print 'Onion: ' + key + ', categories: ' + str(val)
    return L


# Create Hash T mapping onion -> list of title words, where onion is in L.
def CreateT(index_file, S):
    T = {}
    index = os.getcwd() + '/' + index_file
    #print 'Processing index file for titles: ' + index_file
    f = open(index, 'r')
    lines = f.readlines()
    for line in lines:
        # Ignore comments.
        if line.startswith('#'):
            continue
        stripped_line = line.rstrip('\n')
        # print '--> Line: ' + stripped_line
        tokens = stripped_line.split(',')
        onion = tokens[0]
        title_string = tokens[2]
        if len(title_string) <= 1:
            # print 'Ignoring onion ' + onion + ' with empty title.'
            continue
        else:
            title_words = title_string.split(' ')
            for title_word in title_words:
                # Ignore stop words in title.
                if title_word in S:
                    continue
                if len(title_word) > 1:
                    if onion in T:
                        T[onion].append(title_word)
                    else:
                        T[onion] = [title_word]
            # print 'Onion: ' + onion + ', title words: ' + str(T[onion])
#    print 'T hash is:'
#    for (key, val) in T.items():
#        print 'Onion: ' + key + ', title words: ' + str(val)
    return T


# Create Hash H mapping onion -> category list.
def CreateH(index_file):
    # From index_file, create hash H mapping onion -> category list.
    H = {}
    index = os.getcwd() + "/" + index_file
    #print "Processing index file for original categories: " + index_file
    f = open(index, 'r')
    lines = f.readlines()
    for line in lines:
        # Ignore comments.
        if line.startswith("#"):
            continue
        stripped_line = line.rstrip("\n")
        # print "--> Line: " + stripped_line
        tokens = line.split(",")
        onion = tokens[0]
        category_string = tokens[10]
        if len(category_string) <= 1:
            # print "Ignoring onion with empty categories."
            continue;
        else:
            categories = category_string.split(";")
            for cat in categories:
                cat_tokens = cat.split("[")
                cat_label = cat_tokens[0]
                if len(cat_label) > 1:
                    # print "Adding label: " + cat_label + " for onion: " + onion
                    if onion in H:
                        H[onion].append(cat_label)
                        # print "Val: " + str(H[onion])
                    else:
                        H[onion] = [cat_label]
                        # print "Val: " + str(H[onion])
            # print "Onion: " + onion + ", category list: " + str(H[onion])
    return H


# Create Hash B mapping onion -> category list.
def CreateB(baseline_label_file):
    # From baseline_label_file, create hash B mapping onion -> category list.
    B = {}
    baseline = os.getcwd() + "/" + baseline_label_file
    #print "Processing baseline label file for baseline categories: " + baseline_label_file
    f = open(baseline, 'r')
    lines = f.readlines()
    for line in lines:
        # Ignore comments.
        if line.startswith("#"):
            continue
        stripped_line = line.rstrip("\n")
        # print "--> Line: " + stripped_line
        #tokens = line.split(", ")
        tokens = line.split("; ")
        #print 'Tokens: ' + str(tokens)
        onion_tokens = tokens[0].split(".")
        onion = onion_tokens[0]
        category_string = tokens[1]
        if len(category_string) <= 1:
            # print "Ignoring onion with empty categories."
            continue;
        else:
            categories = category_string.split(";")
            for cat in categories:
                cat_tokens = cat.split("[")
                cat_label = cat_tokens[0]
                if len(cat_label) > 1:
                    # print "Adding label: " + cat_label + " for onion: " + onion
                    if onion in B:
                        B[onion].append(cat_label)
                        # print "Val: " + str(H[onion])
                    else:
                        B[onion] = [cat_label]
                        # print "Val: " + str(H[onion])
            # print "Onion: " + onion + ", category list: " + str(B[onion])
    return B


# Create Hash K mapping category -> list of keywords.
def CreateK(keywords_file, L):
    K = {}
    keywords = os.getcwd() + '/' + keywords_file
    #print 'Processing keywords file: ' + keywords_file
    f = open(keywords, 'r')
    lines = f.readlines()
    for line in lines:
        # Ignore comments.
        if line.startswith('#'):
            continue
        stripped_line = line.rstrip('\n')
        # print '--> Line: ' + stripped_line
        tokens = stripped_line.split(',')
        cat_label = tokens[0]
        # Check if cat_label is in categories of labeled set, ignore if not.
        # Flatten list, make items unique.
        L_cats = list(set([item for sublist in list(L.values()) for item in sublist]))
        # print 'Categories in L = ' + str(L_cats)
        if not cat_label in L_cats:
            # print 'Not processing category ' + cat_label + ', not in labeled data.'
            continue
        # else:
            # print 'Processing keyword of category with labels: ' + cat_label
        keywords = tokens[1:]
        kw_tokens = [x.strip() for x in keywords]
        # print 'Category: ' + cat_label + ', keywords: ' + str(kw_tokens)
        for kw in kw_tokens:
            if cat_label in K:
                K[cat_label].append(kw)
            else:
                K[cat_label] = [kw]
#    print 'K hash is:'
#    for (key, val) in K.items():
#        print 'Category: ' + key + ', kw list: ' + str(val)
    return K


# Create Hash data mapping onion x keyword -> count.
def CreateData(wordgrp_dir):
    # Create 2d hash data.
    data = defaultdict(lambda:defaultdict(int))
    data = ProcessFilesInDir(wordgrp_dir, data)
    return data


# Process files in directory to create dataset hash 'data'.
def ProcessFilesInDir(directory, data):
    # Read all filenames in directory.
    path = os.getcwd() + '/' + directory + '/*'
    filenames = glob.glob(path)
    counter = 0
    #print 'Processing files for word lookup in dir: ' + str(path)
    for filename in filenames:
        counter += 1
        # Get onion from filename
        onion = filename[filename.rfind('/')+1:filename.find('.')]
        # print 'Extracted onion name: ' + onion
        f = open(filename, 'r')
        lines = f.readlines()
        for line in lines:
            # Ignore comments.
            if line.startswith('#'):
                continue
            stripped_line = line.rstrip('\n')
            tokens = stripped_line.split(',')
            if '' in tokens:
                continue
            word = tokens[0]
            count = int(tokens[1]) + kPageMultiplier * math.sqrt(int(tokens[2]))
            # print 'Line: ' + stripped_line + ' --> word: ' + word + ', count: ' + count + ', category: ' + str(cat_list)
            data[onion][word] += count
    return data


# Create Hash M mapping category x keyword -> count.
def CreateM(wordgrp_dir, L, T, K, S, H, tst):
    # Create 2d hashes M and Mt (Mt is the transpose of M).
    M = defaultdict(lambda:defaultdict(int))
    Mt = defaultdict(lambda:defaultdict(int))
    (M, Mt) = ProcessFilesInCategory(wordgrp_dir, L, T, K, M, Mt, S, H, tst)
    return (M, Mt)


# Process files in directory to create 2d hash M. Algorithm:
# 1. For onion O with category C:
#       1a. Add count of each word W to M[C][W].
#       1b. If W is a keyword for C, multiply count M[C][W] by kKeywordMultiplier.
# 2. For each word W in title of O with category C:
#       2a. Add kTitleMultiplier to existing count of M[C][W].
def ProcessFilesInCategory(directory, L, T, K, M, Mt, S, H, test):
    # Read all filenames in directory.
    path = os.getcwd() + '/' + directory + '/*'
    filenames = glob.glob(path)
    counter = 0
    #print 'Processing files for category lookup in dir: ' + str(path)
    commonOnions = []
    for filename in filenames:
        counter += 1
        # Get onion from filename
        onion = filename[filename.rfind('/')+1:filename.rfind('.')]
        # print 'Extracted onion name: ' + onion
        
        # If this onion is not in Label set, ignore if not.
        cat_list = []
        cat_from_raters = False
        if onion not in L:
            # Getting original category list from index file.
            if onion in H:
                cat_list = H[onion]
        else:
            if onion in test:
                # print 'Onion ' + onion + ' is in intersection of training and test set.'
                commonOnions.append(onion)
                continue
            cat_list = L[onion]
            cat_from_raters = True
#            print 'Reading filename #' + str(counter) + ': '  + filename 
#            print 'Processing onion: ' + onion + ', in labeled set with categories: ' + str(cat_list)

        f = open(filename, 'r')
        lines = f.readlines()
        if len(lines) < kMinDocSize:
            continue
        else: 
            for line in lines:
                # Ignore comments.
                if line.startswith('#'):
                    continue
                stripped_line = line.rstrip('\n')
                tokens = stripped_line.split(',')
                if '' in tokens:
                    continue
                word = tokens[0]
                # Ignore stop words from kw list.
                if word in S:
                    continue
                count = int(tokens[1])
                # print 'Line: ' + stripped_line + ' --> word: ' + word + ', count: ' + count + ', category: ' + str(cat_list)

                # Step 1.
                if len(cat_list) > 0:
                    count = float(count) / len(cat_list)
                for cat in cat_list:
                    if cat in K and word in K[cat]:
                        count  *= kKeywordMultiplier
                    if cat_from_raters:
                        count *= kRatersMultiplier
                    M[cat][word]  += count
                    Mt[word][cat] += count

        # Step 2.
        if onion not in T:
#            print 'Onion: ' + onion + ' not a key in T_hash'
            continue
        for title_word in T[onion]:
            for cat in cat_list:
                M[cat][title_word]  += kTitleMultiplier
                Mt[title_word][cat] += kTitleMultiplier            

    commonCatCount = {}
    for onion in commonOnions:
        cat = L[onion]
        if cat[0] in commonCatCount:
            commonCatCount[cat[0]] += 1
        else:
            commonCatCount[cat[0]] = 1
#    print 'commonCatCount = ' + str(commonCatCount)
#    print '\n===== Hash M =====\n'
    # print str(M)
    return (M, Mt)


# This function computes the TFICF of the keywords in each category.
# For each category i, it computes the term frequency (TF) of the
# keywords in category i. It also computes the inverse class frequency
# (ICF) of each keyword j in category i. For each category i, it sorts
# the keywords using TF*ICF and outputs the resulting vector of
# category keywords with TFICF weights.
def ComputeTFICF(M, Mt, K, categories):
    ICF = {}
    all_cat = len(list(M.keys()))
    for key in list(Mt.keys()):
        key_cat = len(Mt[key])
        key_tficf = (all_cat + kEpsilon)/(key_cat + kEpsilon)
        ICF[key] = key_tficf
    cat_tficf = {}
    for cat in categories:
        kw_hash = M[cat]
        lst = []
        for kw in list(kw_hash.keys()):
            icf = ICF[kw]
            tf  = kw_hash[kw]
            # print 'tf = ' + str(tf) + ', icf = ' + str(icf)
            tficf = math.sqrt(tf * icf)
            lst.append((kw, tficf))
        sorted_lst = sorted(lst, key=lambda x: x[1], reverse=True)
        # Remove keywords having string length<= kMinKeywordLength.
        pruned_lst = [x for x in sorted_lst if len(x[0]) > kMinKeywordLength]
        cat_tficf[cat] = pruned_lst[0:kMaxVecSize]
    # cat_tficf = PostProcessHash(cat_tficf)
    PrintFinalHash(cat_tficf)
    PrintTopNewKeywords(cat_tficf, K, M, categories)
    return cat_tficf


# Post-process to remove words that occur in more than 1 category.
def PostProcessHash(cat_tficf):
    word_cat_count = {}
    for (cat, lst) in list(cat_tficf.items()):
        for x in lst:
            if x[0] in word_cat_count:
                word_cat_count[x[0]] += 1
            else:
                word_cat_count[x[0]] = 1
    for (cat, lst) in list(cat_tficf.items()):
        pruned_lst = [x for x in lst if word_cat_count[x[0]] == 1]
        cat_tficf[cat] = pruned_lst
    return cat_tficf


# Runs inference on the test set using the following:
#   data: onion x word -> count
#   test: onion -> category
#   keywords: category -> list of (word, weight) tuples5C
#   categories: category list
#
# Method:
#   For each onion in test
#    1. For each category, traverse the list of (word, weights).
#       If a word is in data[onion], accumulate score += weights * count.
#    2. Normalize the scores across categories to sum to 1, using softmax.
#
# Returns a hash 'probs' mapping onion -> list of (category, probability)
def RunInference(data, test, keywords, categories, T, txt):
    probs = {}
    correct_count = 0
    total_count = 0
    for (onion, label) in list(test.items()):
        #if T.has_key(onion):
            #print '\nonion = ' + str(onion) + ', label = ' + str(label) + ', title words = ' + str(T[onion])
        #else:
            #print '\nonion = ' + str(onion) + ', label = ' + str(label) + ', title words UNKNOWN'
        total_score = 0
        lst = []
        for category in categories:
            # print 'category = ' + category
            score = 0
            if category in keywords:
                for (x, w) in keywords[category]:
                    if x in data[onion]:
                        count = data[onion][x]
                        score += count * w
                        # print 'x = ' + str(x) + ', w = ' + str(w) + ', count = ' + str(count) + ', score = ' + str(score)
                lst.append((category, score))
                # print 'lst = ' + str(lst)
                total_score += score
        if total_score > 0:
            lst = [(x[0], x[1]/total_score) for x in lst]
        probs[onion] = lst
        sorted_lst = sorted(lst, key=lambda x: x[1], reverse=True)
        #print '\tProbs = ' + str(sorted_lst)
        # Compute accuracy
        if onion in data and len(data[onion]) > 1:
            total_count += 1
            if sorted_lst[0][0] == test[onion][0]:
                correct_count += 1
        print('\tCorrect_Count = ' + str(correct_count) + ', Total_Count = ' + str(total_count))
    print(str(sorted_lst[0][0]) + '::' + str(test[onion][0]))
    accuracy = (correct_count * 100.0) / (total_count * 1.0)
    print(str(txt) + ' Accuracy (percentage) = ' + str(accuracy) + '\tCorrect_Count = ' + str(correct_count) + ', Total_Count = ' + str(total_count))
    return probs


def RunInferenceOnLabel(data, test, keywords, categories, T, target):
    probs = {}
    correct_count = 0
    total_count = 0
    correct_probs = []
    list_80 = []
    list_90 = []
    for (onion, label) in list(test.items()):
        if not target in label:
            continue;
        if onion in T:
            print('\nonion = ' + str(onion) + ', label = ' + str(label) + ', title words = ' + str(T[onion]))
        else:
            print('\nonion = ' + str(onion) + ', label = ' + str(label) + ', title words UNKNOWN')
        total_score = 0
        lst = []
        for category in categories:
            # print 'category = ' + category
            score = 0
            if category in keywords:
                for (x, w) in keywords[category]:
                    if x in data[onion]:
                        count = data[onion][x]
                        score += count * w
                        # print 'x = ' + str(x) + ', w = ' + str(w) + ', count = ' + str(count) + ', score = ' + str(score)
                lst.append((category, score))
                # print 'lst = ' + str(lst)
                total_score += score
        if total_score > 0:
            lst = [(x[0], x[1]/total_score) for x in lst]
        probs[onion] = lst
        sorted_lst = sorted(lst, key=lambda x: x[1], reverse=True)
        print('\tProbs = ' + str(sorted_lst))
        # Compute accuracy
        if onion in data and len(data[onion]) > 1:
            total_count += 1
            if sorted_lst[0][0] == test[onion][0]:
                correct_count += 1
                correct_probs.append(sorted_lst[0][1])
                if sorted_lst[0][1] > 0.8:
                    list_80.append(onion)
                if sorted_lst[0][1] > 0.9:
                    list_90.append(onion)
        print('\tCorrect_Count = ' + str(correct_count) + ', Total_Count = ' + str(total_count))
    print('\nTotal found = ' + str(total_count))
    print('Number of predictions: ')
    print('\t > 0.5 = ' + str(len([x for x in correct_probs if x > 0.5])))
    print('\t > 0.6 = ' + str(len([x for x in correct_probs if x > 0.6])))
    print('\t > 0.7 = ' + str(len([x for x in correct_probs if x > 0.7])))
    print('\t > 0.8 = ' + str(len([x for x in correct_probs if x > 0.8])))
    print('\t > 0.9 = ' + str(len([x for x in correct_probs if x > 0.9])))

    print('\n==== List for > 0.9, size = ' + str(len([x for x in correct_probs if x > 0.9])))
    for onion in list_90:
        if onion in T:
            print('\nonion = ' + str(onion) + ', title words = ' + str(' '.join(T[onion])))
        else:
            print('\nonion = ' + str(onion) + ', title words UNKNOWN')
    print('\n==== List for > 0.8, size = ' + str(len([x for x in correct_probs if x > 0.8])))
    for onion in list_80:
        if onion in T:
            print('\nonion = ' + str(onion) + ', title words = ' + str(' '.join(T[onion])))
        else:
            print('\nonion = ' + str(onion) + ', title words UNKNOWN')


def RunInferenceDiff(data, B, keywords, categories, T, test, target, threshold):
    probs = {}
    numTargetOnions = 0
    numDiffOnions = 0
    numAllOnions = 0
    for onion in list(data.keys()):
        numAllOnions += 1
        total_score = 0
        lst = []
        for category in categories:
            # print 'category = ' + category
            score = 0
            if category in keywords:
                for (x, w) in keywords[category]:
                    if x in data[onion]:
                        count = data[onion][x]
                        score += count * w
                        # print 'x = ' + str(x) + ', w = ' + str(w) + ', count = ' + str(count) + ', score = ' + str(score)
                lst.append((category, score))
                # print 'lst = ' + str(lst)
                total_score += score
        if total_score > 0:
            lst = [(x[0], x[1]/total_score) for x in lst]
        probs[onion] = lst
        sorted_lst = sorted(lst, key=lambda x: x[1], reverse=True)
        # Print onions that have > threshold probability of being of category 'target'
        if sorted_lst[0][0] == target and sorted_lst[0][1] > threshold and sorted_lst[0][1] < 1.0:
            numTargetOnions += 1
            if onion not in B or target not in B[onion]:
                numDiffOnions += 1
                if onion in T:
                    print('\nonion = ' + str(onion) + ', title words = ' + str(' '.join(T[onion])))
                else:
                    print('\nonion = ' + str(onion) + ', title words UNKNOWN')
                print('\tProbs = ' + str(sorted_lst))
                if onion in B:
                    print('\tBaseline labels = ' + str(B[onion]))
    print('NumAllOnions = ' + str(numAllOnions))
    print('NumTargetOnions = ' + str(numTargetOnions))
    print('NumDiffOnions = ' + str(numDiffOnions) + ', at threshold= ' + str(threshold))

def RunPracticalDiff(data, keywords, categories):
    probs = {}
    numTargetOnions = 0
    numDiffOnions = 0
    numAllOnions = 0
    for onion in list(data.keys()):
        numAllOnions += 1
        total_score = 0
        lst = []
        for category in categories:
            # print 'category = ' + category
            score = 0
            if category in keywords:
                for (x, w) in keywords[category]:
                    if x in data[onion]:
                        count = data[onion][x]
                        score += count * w
                        # print 'x = ' + str(x) + ', w = ' + str(w) + ', count = ' + str(count) + ', score = ' + str(score)
                lst.append((category, score))
                # print 'lst = ' + str(lst)
                total_score += score
        if total_score > 0:
            lst = [(x[0], x[1]/total_score) for x in lst]
        probs[onion] = lst
        sorted_lst = sorted(lst, key=lambda x: x[1], reverse=True)
        # Print onions that have > threshold probability of being of category 'target'
        print('onion = {}'.format(onion))
        print('Probs = {}'.format(sorted_lst))


# Print final tficf_hash with weights.
def PrintFinalHash(cat_tficf):
    num = kMaxVecSize
    print('\n\n==== Printing final weighted keyword list, pruned to top ' + str(num) + ' ====\n\n')
    for (key, val) in list(cat_tficf.items()):
        print('Category: ' + key + ', kw list: ' + str(val[0:num]) + '\n')


# Prints top keywords for the category, sorted in decreasing order of weight.
def PrintTopNewKeywords(cat_tficf, K, M, categories):
    print('\n==== Printing original and new keyword lists ====')
    final_kw_hash = {}
    all_kw = set([])
    for cat in categories:
        all_kw = all_kw.union(set(K[cat]))        
    for cat in categories:
        old_lst = K[cat]
        new_lst = [x[0] for x in cat_tficf[cat] if (x[0] not in all_kw)]
        sub_new_lst = new_lst[0:200]
        print('\n\nExisting keywords for category: ' + cat + ' = ' + str(old_lst))
        print('\nNew keywords for category: ' + cat + ' = ' + str(sub_new_lst))


# Function that processes the input arguments.
def ProcessArguments(argv):
    found_l = False
    found_d = False
    found_k = False
    found_i = False
    found_t = False
    found_s = False
    found_b = False
    found_m = False
    abort = False
    stopwords_file = None
    dedup = False
    practical_dir = None

    try:
        options, args = getopt.getopt(sys.argv[1:],'hl:d:k:i:t:s:b:m:up:',['help','label=','dir=','keywords=','index=','test=','stopwords=','baseline=','mode=','unique','practical='])
    except getopt.GetoptError as error:
        # Print error and usage
        print(str(error))
        PrintUsage()
        sys.exit(2)
    # Process arguments
    for opt, arg in options:
        if opt in ('-h', '--help'):
            # Help message
            PrintUsage()
            sys.exit()
        elif opt in ('-l', '--label'):
            # Training label file
            train_label_file = arg
            found_l = True
            # print 'Training label file is:', train_label_file
        elif opt in ('-d', '--dir'):
            # Wordgrp_dir
            wordgrp_dir = arg
            found_d = True
            # print 'Wordgrp directory is:', wordgrp_dir
        elif opt in ('-k', '--keywords'):
            # Keywords file
            keywords_file = arg
            found_k = True
            # print 'Keywords file is:', keywords_file
        elif opt in ('-i', '--index'):
            # Index file
            index_file = arg
            found_i = True
            # print 'Index file is:', index_file
        elif opt in ('-t', '--test'):
            # Test label file
            test_label_file = arg
            found_t = True
            # print 'Test label file is:', test_label_file
        elif opt in ('-s', '--stopwords'):
            # Stopwords file
            stopwords_file = arg
            found_s = True
            # print 'Stopwords file is:', stopwords_file
        elif opt in ('-b', '--baseline'):
            # Baseline label file
            baseline_label_file = arg
            found_b = True
            # print 'Baseline label file is:', baseline_label_file
        elif opt in ('-m', '--mode'):
            # Mode
            mode = arg
            found_m = True
            # print 'Mode is:', mode
        elif opt in ('-u', '--unique'):
            # Dedup data
            dedup = True
        elif opt in ('-p', '--practical'):
            practical_dir = arg


    # Check if arguments are given
    if not found_l:
        print('Required option -l not given')
        abort = True
    if not found_d:
        print('Required option -d not given')
        abort = True
    if not found_k:
        print('Required option -k not given')
        abort = True
    if not found_i:
        print('Required option -i not given')
        abort = True
    if not found_t:
        print('Required option -t not given')
        abort = True
    if not found_s:
        print('Required option -s not given')
        abort = True
    if not found_b:
        print('Required option -b not given')
        abort = True
    if not found_m:
        print('Required option -m not given')
        abort = True
    if abort:
        PrintUsage()
        sys.exit(2)
    else:
        return train_label_file, wordgrp_dir, keywords_file, index_file, test_label_file, stopwords_file, baseline_label_file, mode, dedup, practical_dir


# Function for printing the usage of the program.
def PrintUsage():
    print('Usage:')
    print('\tGet accuracy results on Feb 19 data: python enhance_keywords.py -l train.labels -d WORD_GRP2 -k KeywordGroups.txt -i MASTER.Onion.Index.csv -t test.labels -s stopwords.txt -b weapons_outDead.txt -m "accuracy"')
    print('\tGet filtering results on Feb 19 data: python enhance_keywords.py -l train.labels -d WORD_GRP2 -k KeywordGroups.txt -i MASTER.Onion.Index.csv -t test.labels -s stopwords.txt -b weapons_outDead.txt -m "filtering"')
    print('\tGet discovery results on Mar 2 data: python enhance_keywords.py -l train.labels -d WORD_GRP3 -k KeywordGroups_03022016.txt -i MASTER.Onion.Index_03022016.csv -t test.labels -s stopwords.txt -b wordGrp3_Results_03022016.dat -m "discovery"')

if __name__ == '__main__':
    main(sys.argv[1:])
