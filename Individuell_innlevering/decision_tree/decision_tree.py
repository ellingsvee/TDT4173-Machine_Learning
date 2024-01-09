import numpy as np 
import pandas as pd 
import random

class DecisionTree:
    def __init__(self, target_attribute = 'Play Tennis', predict_mtd = "most_similar_branch", alphas = [0.001, 0.01, 0.1, 1.0, 10.0]):
        # Initializing the hyperparameters
        self.tree = None
        self.target_attribute = target_attribute
        self.alphas = alphas
        self.predict_mtd = predict_mtd
        self.tree_hierachy = []

    def fit(self, X, y):
        """
        Generates a decision tree for classification
        
        Args:
            X (pd.DataFrame): a matrix with discrete value where
                each row is a sample and the columns correspond
                to the features.
            y (pd.Series): a vector of discrete ground-truth labels
        """
        self.number_of_columns = X.shape[1]
        self.booleans = y.unique().tolist() # A list containg the names of the boolean-values in the training-data. F.ex. ["Yes", "No"], ["success", "failure"] etc.
        self.tree = self.build_tree(X, y, X.columns.tolist()) # Building the tree

    def build_tree(self, X, y, attributes):
        """
        Recursively building a tree based on the training data.

        Args:
            X (pd.DataFrame): a matrix with discrete value where
                each row is a sample and the columns correspond
                to the features.
            y (pd.Series): a vector of discrete ground-truth labels
            attributes (np.array): list of the column-names in X
        """
        root = [] 

        # Checking if there are only one more unique value in y.
        unique_y_vals = y.unique()
        if len(unique_y_vals) == 1:
            if unique_y_vals[0] in self.booleans:
                root.append(([], unique_y_vals[0]))
            return root
        # Checking if there are no more columns in X that has been taken into account. If so, returning the most common value in y.
        if len(attributes) == 0:
            most_common = self.most_common_value(y)
            root.append(([], most_common))
            return root

        attr = self.find_best_attribute(X, y, attributes) # Finding the remaining attribute with the highest gain.
        if len(self.tree_hierachy) < self.number_of_columns:
            self.tree_hierachy.append(attr)

        for val in X[attr].unique().tolist(): # Iterating throught the elements in the column related to the attribute
            condition_mask = X[attr] == val
            if X[condition_mask].empty:
                leaf_label = self.most_common_value(y)
                root.append(([(attr, val)], leaf_label))
            else: 
                new_attributes = np.delete(attributes, np.argwhere(attributes == attr))
                sub_tree = self.build_tree(X[condition_mask], y[condition_mask], new_attributes) # Recursion
                for sub_case in sub_tree:
                    root.append(([(attr, val)] + sub_case[0], sub_case[1]))
        return root

    def most_common_value(self, y):
        """
        Finding the most common value in y
        Args:
            y (pd.Series): a vector of discrete ground-truth labels
        """
        return y.value_counts().idxmax() 

    def find_best_attribute(self, X, y, attributes):
        """
        Finding the best attribute, i.e. the one with the highest gain.
        
        Args:
            X (pd.DataFrame): a matrix with discrete value where
                each row is a sample and the columns correspond
                to the features.
            y (pd.Series): a vector of discrete ground-truth labels
            attributes (np.array): list of the remaining column-names in X
        """
        tot_entropy = entropy(y.value_counts())
        gains = {}
        for attribute in attributes:
            gains[attribute] = self.gain(X, y, attribute, tot_entropy)
        return max(gains, key=lambda k: gains[k])

    def gain(self, X, y, attribute, tot_entropy):
        """
        Calculating the gain.
        """
        values = X[attribute].unique()
        return tot_entropy - np.sum([(len(y[X[attribute] == value])/len(y))*entropy(y[X[attribute] == value].value_counts()) for value in values])


    def predict(self, X, tree=None):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (pd.DataFrame): an mxn discrete matrix where
                each row is a sample and the columns correspond
                to the features.
            
        Returns:
            A length m vector with predictions
        """
        if tree == None:
            tree = self.tree
        return np.array([self.predict_sample(row.to_dict(), tree) for _, row in X.iterrows()])

    def predict_sample(self, sample, tree):
        """
        Predicting a sample based on the tree generated during the fitting.

        If there exist a direct path, one can whoose two different method of obtaining a label. Eiter use the find_most_similar- or the move_back_in_tree-function.
        """
        rules = []
        for conditions, label in tree:
            conditions_satisfied = all(sample[attr] == val for attr, val in conditions)
            if conditions_satisfied == True:
                return label # If a perfect fit was found
            rules.append([[sample[attr] == val for attr, val in conditions], label, conditions])
        
        
        if self.predict_mtd == "most_similar_branch":
            return self.find_most_similar(rules) # If no perfect fit was found
        elif self.predict_mtd == "move_back_in_tree":
            return self.move_back_in_tree(sample, tree)
        else:
            return random.choice(self.booleans)
    
    def move_back_in_tree(self, sample, tree):
        """
        Move to an earlier node in the tree, and return the label that is the most common among the branches stemming from this node. Performs recursion if there are no such branches by moving furter up.
        """
        for element in reversed(self.tree_hierachy):
            if element in sample:
                del sample[element]
                break
        potential_labels = []
        sample_len = len(sample)
        for conditions, label in tree:
            temp_conditions_satisfied = 0
            for attr, val in conditions:
                if attr in sample and sample[attr] == val:
                    temp_conditions_satisfied += 1
            if temp_conditions_satisfied == sample_len:
                potential_labels.append(label)

        if not potential_labels:
            return self.move_back_in_tree(sample, tree)
        return pd.Series(potential_labels).value_counts().idxmax()

    def find_most_similar(self, rules):
        # If no "direct" path was found, find the other rule that if the most "similar" to the sample we want to predict.
        """
        Find the boolean label corresponding to the path in the tree that has the most in common with the sample.
        For each path in the tree, check how many rules (from the sample) that the path satisfies. Return the boolean label from the path that satisfies the most rules.

        POTENTIAL IMPROVEMENT: Take the "importance" of each of the rules into account, i.e. the gain of the attribute that corresponds to the spesific rule.
        """
        max_true_count = 0
        best_index = -1
        for i, rule in enumerate(rules):
            lst = rule[0]
            true_count = lst.count(True)
            if true_count > max_true_count:
                max_true_count = true_count
                best_index = i
        return rules[best_index][1]

    def get_rules(self):
        """
        Returns the decision tree as a list of rules
        
        Each rule is given as an implication "x => y" where
        the antecedent is given by a conjuction of attribute
        values and the consequent is the predicted label
        
            attr1=val1 ^ attr2=val2 ^ ... => label
        
        Example output:
        >>> model.get_rules()
        [
            ([('Outlook', 'Overcast')], 'Yes'),
            ([('Outlook', 'Rain'), ('Wind', 'Strong')], 'No'),
            ...
        ]
        """
        return self.tree
    
    ############### TWO TREE-PRUNING AGLORITHMS ###############
    # These are not getting used because it was not neccessary for the dataset I was working with, but can be of importance for other instances.

    def prune_tree_rep(self, X_val, y_val):
        best_accuracy = accuracy(y_val, self.predict(X_val, self.tree))
        # pruned_tree = self.tree.copy()
        original_tree_len = len(self.tree)
        for i, (conditions, _) in enumerate(self.tree):
            if conditions:
                pruned_subtree = self.tree[:i] + self.tree[i+1:]
                accuracy_without_subtree = accuracy(y_val, self.predict(X_val, pruned_subtree))

                if accuracy_without_subtree > best_accuracy:
                    best_accuracy = accuracy_without_subtree
                    self.tree = pruned_subtree
        print(f"Performed tree pruning: {len(self.tree) - original_tree_len} rule(s) removed.")

    def prune_tree_ccp(self, X_val, y_val, alpha, print_end=False):
        original_tree_len = len(self.tree)
        node_costs = self.calculate_node_costs(self.tree, X_val, y_val, alpha)
        pruned_tree = self.tree.copy()

        for i, (conditions, _) in enumerate(self.tree):
            if conditions:
                if node_costs[i] > 0:  # Prune if the node cost is positive
                    pruned_subtree = self.tree[:i] + self.tree[i+1:]
                    accuracy_without_subtree = accuracy(y_val, self.predict(X_val, pruned_subtree))

                    if accuracy_without_subtree >= accuracy(y_val, self.predict(X_val, pruned_tree)):
                        pruned_tree = pruned_subtree
        if print_end == True:
            print(f"Performed CCP pruning: {original_tree_len - len(pruned_tree)} rule(s) removed.")
        self.tree = pruned_tree

    def calculate_node_costs(self, tree, X_val, y_val, alpha):
        node_costs = []
        for conditions, _ in tree:
            if conditions:
                pruned_subtree = [node for node in tree if node != (conditions, _)]
                accuracy_without_node = accuracy(y_val, self.predict(X_val, pruned_subtree))
                misclass_count = len(y_val) - int(len(y_val) * accuracy_without_node)
                node_cost = misclass_count + alpha * len(pruned_subtree)
                node_costs.append(node_cost)
            else:
                node_costs.append(0)
        return node_costs
    
    def tune_alpha(self, X_val, y_val):
        best_accuracy = 0
        best_alpha = None

        for alpha in self.alphas:
            # Create a copy of the tree to prevent modifying the original tree
            tree_copy = self.tree.copy()

            # Prune the tree using the current alpha value
            self.prune_tree_ccp(X_val, y_val, alpha)

            # Calculate accuracy on the validation set
            accuracy_val = accuracy(y_val, self.predict(X_val, self.tree))

            # Update the best accuracy and alpha if needed
            if accuracy_val > best_accuracy:
                best_accuracy = accuracy_val
                best_alpha = alpha

            # Reset the tree to the original state
            self.tree = tree_copy

        print(f"Best alpha: {best_alpha}, Best accuracy: {best_accuracy}")
        return best_alpha

# --- Some utility functions 
    
def accuracy(y_true, y_pred):
    """
    Computes discrete classification accuracy
    
    Args:
        y_true (array<m>): a length m vector of ground truth labels
        y_pred (array<m>): a length m vector of predicted labels
        
    Returns:
        The average number of correct predictions
    """
    assert y_true.shape == y_pred.shape
    return (y_true == y_pred).mean()


def entropy(counts):
    """
    Computes the entropy of a partitioning
    
    Args:
        counts (array<k>): a lenth k int array >= 0. For instance,
            an array [3, 4, 1] implies that you have a total of 8
            datapoints where 3 are in the first group, 4 in the second,
            and 1 one in the last. This will result in entropy > 0.
            In contrast, a perfect partitioning like [8, 0, 0] will
            result in a (minimal) entropy of 0.0
            
    Returns:
        A positive float scalar corresponding to the (log2) entropy
        of the partitioning.
    
    """
    assert (counts >= 0).all()
    probs = counts / counts.sum()
    probs = probs[probs > 0]  # Avoid log(0)
    return - np.sum(probs * np.log2(probs))