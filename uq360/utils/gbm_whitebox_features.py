
import sys
import numpy as np


class GBM_WhiteboxFeatureExtractor():
    def __init__(self, gbm):
        self.gbm = gbm

        # The whitebox feature structure gets passed in and we add to it
        self.whitebox_features = {}

    def add_whitebox_feature(self, feature_name, value):
        self.whitebox_features[feature_name] = value

    def compute_gbm_internal_whitebox_features(self, x_test, y_test, x_prod):
        self.compute_decision_distance(x_test, x_prod)
        self.compute_leaf_frequencies(x_test, y_test, x_prod)
        return self.whitebox_features

    # fyi, a few lines are commented out as they maybe required for debugging
    def compute_leaf_frequencies(self, x_test, y_test, x_prod):
        # Track how the right vs wrong nodes from test go through the tree
        # Compute distributions and accuracies for each node

        preds = self.gbm.predict(x_test)
        pred_correct = preds == y_test

        x_test_correct = x_test[pred_correct]
        x_test_incorrect = x_test[~pred_correct]

        if len(x_test_correct) == 0:
            raise Exception("Need to handle this rare corner case")
        
        # Get the leaf nodes that were hit in each tree for these data point
        # NOTE:  I shouldn't need to call apply on correct, incorrect, then x_test
        #        I should be able to call it just on x_test then tease it apart
        nodes_correct = self.gbm.apply(x_test_correct).astype(int)
        if len(x_test_incorrect) > 0:
            nodes_incorrect = self.gbm.apply(x_test_incorrect).astype(int)
            have_incorrect_predictions = True
        else:
            nodes_incorrect = None
            have_incorrect_predictions = False
            
        test_leaves = self.gbm.apply(x_test).astype(int)
        prod_leaves = self.gbm.apply(x_prod).astype(int)

        # Find the number of nodes by looking through the output of both right and wrong
        max1 = np.amax(nodes_correct)
        if have_incorrect_predictions:
            max2 = np.amax(nodes_incorrect)
        else:
            max2 = 0
        max3 = np.amax(test_leaves)
        max4 = np.amax(prod_leaves)
        max_node_id = max(max1, max2, max3, max4)+1

        node_counts_correct = self.count_nodes(nodes_correct, max_node_id)

        if have_incorrect_predictions:
            node_counts_incorrect = self.count_nodes(nodes_incorrect, max_node_id)
            node_counts_total = node_counts_correct + node_counts_incorrect
        else:
            node_counts_total = node_counts_correct
            
        node_accuracy = node_counts_correct / node_counts_total
        node_accuracy = np.nan_to_num(node_accuracy)
        
        # print ('Correct', node_counts_correct)
        # print ('Incorrect', node_counts_incorrect)
        # print ('Total', node_counts_total)
        
        # print ('Accuracy', node_accuracy)

        # Now call them again with both test and prod, passing in the node accuracies
        # So we can see how they vary

        np.set_printoptions(suppress=True)
        node_counts_test, accuracy_test = self.count_nodes(test_leaves, max_node_id, node_accuracy)
        # print('Test')
        # print(node_counts_test)
        # print('Test node accuracy', accuracy_test)

        
        node_counts_prod, accuracy_prod = self.count_nodes(prod_leaves, max_node_id, node_accuracy)
        # print('Prod')
        # print(node_counts_prod)
        # print('Prod node accuracy', accuracy_prod)
        
        
        # print('Delta (abs)')
        deltas = abs(node_counts_prod - node_counts_test)
        # print('max', round(deltas.max(),5))
        # print('sum', round(deltas.sum(),5))
        # print('std', round(deltas.std(),5))

        self.add_whitebox_feature('gbm_node_freq_delta_abs_max', deltas.max())
        self.add_whitebox_feature('gbm_node_freq_delta_abs_sum', deltas.sum())
        self.add_whitebox_feature('gbm_node_freq_delta_abs_std', deltas.std())
        
        delta_accuracy = accuracy_prod - accuracy_test
        # print("Delta accuracy", round(delta_accuracy,4))

        self.add_whitebox_feature('gbm_node_accuracy_delta', delta_accuracy)
        self.add_whitebox_feature('gbm_node_accuracy_delta_abs', abs(delta_accuracy))

    def count_nodes(self, data, max_node_id, accuracy_map=None):
        if accuracy_map is None:
            compute_accuracy = False
        else:
            compute_accuracy = True
            accuracy = 0

        shape = data.shape
        num_trees = shape[1]*shape[2]
        num_nodes = max_node_id * num_trees
        num_data_points = len(data)
        # print('Max node id', max_node_id)
        # print('Num trees', num_trees)
        # print('Num nodes', num_nodes)
        # print('Num data points', num_data_points)
        # print ('max_node_id', max_node_id)
        
        node_counts = np.zeros((num_trees, max_node_id))
        for datum in data:
            trees = datum.flatten()
            tree_num = -1
            for node in trees:
                tree_num += 1
                node_counts[tree_num, node] += 1
                if compute_accuracy:
                    accuracy += accuracy_map[tree_num, node]

        node_averages = node_counts / num_data_points / num_trees

        if compute_accuracy:
            avg_accuracy = accuracy / num_data_points / num_trees
            return node_averages, avg_accuracy
        return node_averages

    def compute_decision_distance(self, x_test, x_prod):
        test = self.compute_depth_sums(x_test)
        prod = self.compute_depth_sums(x_prod)
        # print ("Test avgs", test)
        # print ("Prod avgs", prod)
        deltas = prod - test
        # print ("Deltas", deltas)

        i = 0
        for delta in deltas:
            self.add_whitebox_feature('gbm_delta_depth_' + str(i), delta)
            self.add_whitebox_feature('gbm_delta_depth_abs_' + str(i), abs(delta))
            i+=1
        
        sum_delta = prod.sum() - test.sum()
        # print ("Sum delta", sum_delta)
        self.add_whitebox_feature('gbm_delta_sum', sum_delta)
        self.add_whitebox_feature('gbm_delta_sum_abs', abs(sum_delta))
        
    def compute_depth_sums(self, data):
        max_depth = self.gbm.max_depth+1
        # print ("Max depth", max_depth)
        depth_sum_sums = np.zeros(max_depth)
        num_nodes=0
        num_trees=0    
        for estimators in self.gbm.estimators_:
            for tree in estimators:
                num_trees+= 1
                n_nodes = tree.tree_.node_count
                num_nodes += n_nodes
                depth_sums = self.compute_tree_depth_sums(tree, data, max_depth)
                depth_sum_sums += depth_sums
        # print ("Num trees", num_trees)
        # print ("Num nodes", num_nodes)
        # print ("Num data points", len(data))

        # divide by num data points (total, across all trees) so it doesn't matter if test and prod have different number of data points
        depth_sum_avg = depth_sum_sums / (len(data) * num_trees)
        return depth_sum_avg 

    def compute_tree_depth_sums(self, estimator, data, max_depth):
        leave_id = estimator.apply(data)
        n_nodes = estimator.tree_.node_count
        children_left = estimator.tree_.children_left
        children_right = estimator.tree_.children_right
        feature = estimator.tree_.feature
        threshold = estimator.tree_.threshold
        
        node_indicator = estimator.decision_path(data)
        
        depth_sums = np.zeros(max_depth)
        for sample_id in range(len(data)):
            node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                            node_indicator.indptr[sample_id + 1]]
            depth = -1
            for node_id in node_index:
                # I'm not entirely sure whether this iteration is in order, and this depth is
                # actually the tree depth... but we'll pretend that it is
                depth += 1
                if leave_id[sample_id] == node_id:
                    continue

                feature_val = data[sample_id, feature[node_id]]
                threshold_val = threshold[node_id]
                delta = abs(feature_val - threshold_val)
                depth_sums[depth] += delta
        return depth_sums
