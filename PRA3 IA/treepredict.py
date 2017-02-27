#!/usr/bin/env python
#Author: Jordi Blanco Lopez
# NIF: 20998

import argparse
import collections
import itertools
import math
import random
import sys
import time


# ---- t3 ----

def read_file(file_path, data_sep=' ', ignore_first_line=False):
    with open(file_path, 'r') as f:
        return read_stream(f, data_sep, ignore_first_line)


def read_stream(stream, data_sep=' ', ignore_first_line=False):
    strip_reader = (l.strip()  for l in stream)
    filtered_reader = (l for l in strip_reader if l) 
    start_at = 1 if ignore_first_line else 0

    prototypes = []
    for line in itertools.islice(filtered_reader, start_at, None):
        tokens = itertools.imap(str.strip, line.split(data_sep))
        prototypes.append(map(filter_token, tokens))

    return prototypes


def filter_token(token):
    try:
        return int(token)
    except ValueError:
        try:
            return float(token)
        except ValueError:
            return token


# ---- t4 ----

def unique_counts(part):
    # return collections.Counter(row[-1] for row in part)
    counts = {}
    for row in part:
        counts[row[-1]] = counts.get(row[-1], 0) + 1
    return counts


# ---- t5 ----

def gini_impurity(part):
    total = float(len(part))
    counts = unique_counts(part)

    probs = (v / total for v in counts.itervalues())
    return 1 - sum((p * p for p in probs))

# ---- t6 ----

def entropy(rows):
   from math import log
   log2=lambda x:log(x)/log(2)
   results=unique_counts(rows)
   # Now calculate the entropy
   ent=0.0
   for r in results.keys():
      p=float(results[r])/len(rows)
      ent=ent-p*log2(p)
   return ent

# ---- t7 ----

def divideset(part, column, value):
    def split_num(prot): return prot[column] >= value
    def split_str(prot): return prot[column] == value

    split_fn = split_num if isinstance(value, (int, float)) else split_str
  
    set1, set2 = [], []
    for prot in part:
        s = set1 if split_fn(prot) else set2
        s.append(prot)

    return set1, set2


# ---- t8 ----

class decisionnode:
  def __init__(self,col=-1,value=None,results=None,tb=None,fb=None):
    self.col=col
    self.value=value
    self.results=results
    self.tb=tb
    self.fb=fb

# ---- t9 ----

def buildtree(rows, scoref=entropy, beta=0):
    if len(rows) == 0: return decisionnode()
    current_score = scoref(rows)

    # Set up some variables to track the best criteria
    best_gain = 0.0
    best_criteria = None
    best_sets = None

    column_count = len(rows[0]) - 1
    for col in range(0, column_count):
        # Generate the list of different values in
        # this column
        column_values = {}
        for row in rows:
            column_values[row[col]] = 1
        # Now try dividing the rows up for each value
        # in this column
        for value in column_values.keys():
            (set1, set2) = divideset(rows, col, value)

            # Information gain
            p = float(len(set1)) / len(rows)
            gain = current_score - p * scoref(set1) - (1 - p) * scoref(set2)
            if gain > best_gain and len(set1) > 0 and len(set2) > 0:
                best_gain = gain
                best_criteria = (col, value)
                best_sets = (set1, set2)
    # Create the sub branches
    if best_gain > beta:
        trueBranch = buildtree(best_sets[0])
        falseBranch = buildtree(best_sets[1])
        return decisionnode(col=best_criteria[0], value=best_criteria[1],
                            tb=trueBranch, fb=falseBranch)
    else:
        return decisionnode(results=unique_counts(rows))

# ---- t10 ----

def buildtreeIt(part, scoref=entropy, beta=0):
    root = decisionnode()
    stack = [[part, root]]

    while len(stack) > 0:

        prototip, parent = stack.pop()
        current_score = scoref(prototip)

        best_gain = 0
        best_criteria = None
        best_sets = None
        best_col = -1

        for column in range(len(prototip[0]) - 1):
            divide_criterials = []
            for rows in part:
                if rows[column] not in divide_criterials:
                    divide_criterials.append(rows[column])
            for criteria in divide_criterials:
                sets = divideset(prototip, column, criteria)
                gain = current_score - (len(sets[0]) / len(prototip)) * scoref(sets[0]) - (len(sets[1]) / len(
                    prototip)) * scoref(sets[1])
                if gain > best_gain:
                    best_gain = gain
                    best_criteria = criteria
                    best_sets = sets
                    best_col = column

        if best_gain > beta:
            parent.tb = decisionnode()
            parent.fb = decisionnode()
            parent.value = best_criteria
            parent.col = best_col

            stack.append([best_sets[0], parent.tb])
            stack.append([best_sets[1], parent.fb])

        else:
            parent.results = unique_counts(prototip)

    return root


# ---- t11 ----

def printtree(tree,indent=''):
   # Is this a leaf node?
   if tree.results!=None:
      print str(tree.results)
   else:
      # Print the criteria
      print str(tree.col)+':'+str(tree.value)+'? '

      # Print the branches
      print indent+'T->',
      printtree(tree.tb,indent+'  ')
      print indent+'F->',
      printtree(tree.fb,indent+'  ')

#  ---- t12 ----

def classify(observation, tree):
       if tree.results != None:
           return tree.results
       else:
           v = observation[tree.col]
           branch = None
           if isinstance(v, int) or isinstance(v, float):
               if v >= tree.value:
                   branch = tree.tb
               else:
                   branch = tree.fb
           else:
               if v == tree.value:
                   branch = tree.tb
               else:
                   branch = tree.fb
           return classify(observation, branch)

# ---- t13 ----
def test_performance(testset, trainingset):
    trained = buildtree(trainingset)
    good = 0.0
    for x in testset:
        result = classify(x, trained)
        if result.get(str(x[-1])) is not None:
            good += 1.0
    return str(good/len(testset)*100) + "%"

def divideset_test(part, porcent):
    divide = int((len(part) / float(porcent)) * len(part))
    return (part[:divide], part[divide:])

def notfound_imput(part, none_element):
    random.seed()
    for column in range(len(part[0]) - 1):
        none_rows = []
        parts = []
        for rows in part:
            if rows[column] == none_element:
                none_rows.append(rows)
            elif rows[column] not in parts:
                parts.append(rows[column])
        for rows in none_rows:
            rows[column] = parts[random.randint(0, len(parts) - 1)]
# ---- t15 ----

# 'missing data classify'
def mdclassify(observation,tree):
  if tree.results!=None:
    return tree.results
  else:
    v=observation[tree.col]
    if v==None:
      tr,fr=mdclassify(observation,tree.tb),mdclassify(observation,tree.fb)
      tcount=sum(tr.values())
      fcount=sum(fr.values())
      tw=float(tcount)/(tcount+fcount)
      fw=float(fcount)/(tcount+fcount)
      result={}
      for k,v in tr.items(): result[k]=v*tw
      for k,v in fr.items(): result[k]=v*fw
      return result
    else:
      if isinstance(v,int) or isinstance(v,float):
        if v>=tree.value: branch=tree.tb
        else: branch=tree.fb
      else:
        if v==tree.value: branch=tree.tb
        else: branch=tree.fb
      return mdclassify(observation,branch)


# ---- t16 ----

def prune(tree, mingain):
    # If the branches aren't leaves, then prune them
    if tree.tb.results == None:
        prune(tree.tb, mingain)
    if tree.fb.results == None:
        prune(tree.fb, mingain)

    # If both the subbranches are now leaves, see if they
    # should merged
    if tree.tb.results != None and tree.fb.results != None:
        # Build a combined dataset
        tb, fb = [], []
        for v, c in tree.tb.results.items():
            tb += [[v]] * c
        for v, c in tree.fb.results.items():
            fb += [[v]] * c

        # Test the reduction in entropy
        delta = entropy(tb + fb) - (entropy(tb) + entropy(fb) / 2)

        if delta < mingain:
            # Merge the branches
            tree.tb, tree.fb = None, None
            tree.results = unique_counts(tb + fb)

# ------------------------ #
#        Entry point       #
# ------------------------ #

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="---- DESCRIPTION HERE ----",
        epilog="---- EPILOG HERE ----")

    parser.add_argument('prototypes_file', type=argparse.FileType('r'), help="File filled with prototypes (one per line)")

    parser.add_argument('-ifl', '--ignore_first_line', action='store_true', help="Ignores the first line of the prototypes file")

    parser.add_argument('-ds', '--data_sep', required=False, default=',', help="Prototypes data fields separation mark")

    parser.add_argument('-s', '--seed', default=int(time.time()), type=int, help="Random number generator seed.")

    options = parser.parse_args()

    protos = read_stream(options.prototypes_file, options.data_sep, options.ignore_first_line)

    print unique_counts(protos)

    tree = buildtree(protos)
    tree2 = buildtreeIt(protos)
	
    print (" ")
    printtree(tree)
    print(" ")
    printtree(tree2)
