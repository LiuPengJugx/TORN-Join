from re import X
from scipy import rand
from partition_tree import PartitionTree
from partition_node import PartitionNode
import time
from rtree import index
import numpy as np
import copy
import random
from line_profiler import LineProfiler
class PartitionAlgorithm:
    '''
    The partition algorithms, inlcuding NORA, QdTree and kd-tree.
    '''
    def __init__(self, data_threshold = 10000):
        self.partition_tree = None
        self.data_threshold = data_threshold
    
    # = = = = = public functions (API) = = = = =
    
    def InitializeWithPAW(self, queries, num_dims, boundary, dataset, data_threshold, max_active_ratio = 3, strategy = 1,
                          using_beam_search = False, candidate_size = 1, candidate_depth = 1, beam_search_mode = 0):
        self.partition_tree = PartitionTree(num_dims, boundary)
        self.partition_tree.table_dataset_size=dataset.shape[0]
        self.partition_tree.name='PAW'

        self.partition_tree.pt_root.node_size = len(dataset)
        self.partition_tree.pt_root.dataset = dataset
        self.partition_tree.pt_root.queryset = queries # assume all queries overlap with the boundary
        self.partition_tree.pt_root.generate_query_MBRs()
        
        start_time = time.time()
        if using_beam_search:
            self.__PAW_Beam_Search(data_threshold, queries, max_active_ratio, strategy, candidate_size, candidate_depth,
                                   None, beam_search_mode)
        else:
            self.__PAW(data_threshold, max_active_ratio, strategy)
        end_time = time.time()        
        print("Build Time (s):", end_time-start_time)
        self.partition_tree.build_time =end_time-start_time

    def InitializeWithPAW2(self, queries, num_dims, boundary, dataset, data_threshold, max_active_ratio=3,
                          using_beam_search=False, candidate_size=1, candidate_depth=1, beam_search_mode=0,strategy=0):
        self.partition_tree = PartitionTree(num_dims, boundary)
        self.partition_tree.table_dataset_size = dataset.shape[0]
        self.partition_tree.name='PAW2'
        self.partition_tree.pt_root.node_size = len(dataset)
        self.partition_tree.pt_root.tree_boundary = boundary
        self.partition_tree.pt_root.dataset = dataset
        self.partition_tree.pt_root.queryset = queries  # assume all queries overlap with the boundary
        self.partition_tree.pt_root.generate_query_MBRs()

        start_time = time.time()
        if using_beam_search:
            # after applying beam search, try general group split
            # skip = self.partition_tree.pt_root.if_general_group_split_3(data_threshold)
            # self.partition_tree.apply_split(self.partition_tree.pt_root.nid, None, None, 3)
            # print(f"apply group split: total {len(self.partition_tree.pt_root.children_ids)} nodes.")
            self.__PAW_Beam_Search2(data_threshold, queries, max_active_ratio, strategy, candidate_size, candidate_depth,
                                   None, beam_search_mode)
        else:
            self.__PAW(data_threshold, max_active_ratio,strategy=1)
        end_time = time.time()
        print("Build Time (s):", end_time - start_time)
        self.partition_tree.build_time = end_time - start_time



    def InitializeWithJNORA(self,queries, num_dims, boundary, dataset, data_threshold, join_attr, using_1_by_1 = False, using_kd = False,using_am=False,
                             candidate_size = 2,candidate_depth = 2,depth_limit =None,join_depth=3):
        self.partition_tree = PartitionTree(num_dims, boundary)
        self.partition_tree.table_dataset_size = dataset.shape[0]
        self.partition_tree.join_attr=join_attr
        self.partition_tree.join_depth=join_depth
        self.partition_tree.name = 'JNORA'
        self.partition_tree.pt_root.node_size = len(dataset)
        self.partition_tree.pt_root.dataset = dataset
        self.partition_tree.pt_root.queryset = queries  # assume all queries overlap with the boundary
        self.partition_tree.pt_root.generate_query_MBRs()
        start_time = time.time()
        # get partitions for top join attribute
        time0=time.time()
        if using_am:
            self.__AMT_JOIN(data_threshold,join_attr,join_depth)
        else:
            # self.__JNORA_Beam_Search_Plus(data_threshold,queries,join_attr,candidate_size,candidate_depth,depth_limit,join_depth=join_depth)
            self.__JNORA_Plus(data_threshold,join_attr,join_depth)
            # self.__AMT_JOIN(data_threshold,join_attr,join_depth)
        print("time for cut top join: ",time.time()-time0)
        # query_cross_cost=sum([len(self.partition_tree.query_single(query)) - 1 for query in queries])
        # for rest leafs, get partitions for all attribute
        self.__JNORA_Beam_Search(data_threshold,candidate_size,candidate_depth)
        if using_kd:
            for leaf in self.partition_tree.get_leaves():
                if leaf.is_irregular_shape or leaf.is_irregular_shape_parent:
                    continue
                self.__KDT(0, data_threshold, leaf)
        end_time = time.time()

        print("Build Time (s):", end_time - start_time)
        # print("Query cross count:",query_cross_cost)
        # self.query_count=query_cross_cost
        self.partition_tree.build_time = end_time - start_time

    def InitializeWithADP(self, queries, num_dims, boundary, dataset, data_threshold,join_attr,join_depth=3):
        '''
        # should I also store the candidate cut positions in Partition Node ?
        The dimension of queries should match the dimension of boundary and dataset!
        '''
        self.partition_tree = PartitionTree(num_dims, boundary)
        self.partition_tree.name='AdaptDB'
        self.partition_tree.join_attr = join_attr
        self.partition_tree.join_depth = join_depth
        self.partition_tree.pt_root.node_size = len(dataset)
        self.partition_tree.pt_root.dataset = dataset
        self.partition_tree.pt_root.queryset = queries # assume all queries overlap with the boundary
        start_time = time.time()
        self.__AMT_JOIN(data_threshold, join_attr, join_depth)
        print(f"{join_depth} ______________________________________")
        self.__JQDT(data_threshold)
        end_time = time.time()
        print("Build Time (s):", end_time-start_time)
        self.partition_tree.build_time = end_time - start_time

    def InitializeWithAMT(self, num_dims, boundary, dataset, data_threshold):
        self.partition_tree = PartitionTree(num_dims, boundary)
        self.partition_tree.pt_root.node_size = len(dataset)
        self.partition_tree.pt_root.dataset = dataset
        self.partition_tree.pt_root.depth = 0
        p=0.001
        x_prev=data_threshold
        x=data_threshold/num_dims
        while abs(x-x_prev)>p:
            x_prev=x
            x=((num_dims-1)*x+data_threshold/pow(x,num_dims-1))/num_dims
        allocation_per_attribute=x
        self.partition_tree.allocations=[allocation_per_attribute for _ in range(num_dims)]

        start_time = time.time()
        self.__AMT(num_dims,data_threshold)
        end_time = time.time()
        print("Build Time (s):", end_time-start_time)

    def InitializeWithNORA(self, queries, num_dims, boundary, dataset, data_threshold, using_1_by_1 = False, using_kd = False, 
                           depth_limit = None, return_query_cost = False, using_beam_search = False,using_beam_search_plus = False, candidate_size = 2,
                           candidate_depth = 2):
        '''
        using_1_BY_1: using some optimizations including new bounding split and bounding split in internal node
        using_kd: split leaf node by kd if still greater than b
        '''
        self.partition_tree = PartitionTree(num_dims, boundary)
        self.partition_tree.table_dataset_size = dataset.shape[0]
        self.partition_tree.name='NORA'
        self.partition_tree.pt_root.node_size = len(dataset)
        self.partition_tree.pt_root.dataset = dataset
        self.partition_tree.pt_root.queryset = queries # assume all queries overlap with the boundary
        start_time = time.time()
        if using_1_by_1:
            self.partition_tree.pt_root.generate_query_MBRs()
            if using_beam_search:
                # profile = LineProfiler(self.__NORA_Beam_Search)
                # profile.runcall(self.__NORA_Beam_Search,data_threshold, candidate_size, candidate_depth)
                # profile.print_stats()
                self.__NORA_Beam_Search(data_threshold, candidate_size, candidate_depth)
            elif using_beam_search_plus:
                self.__NORA_Beam_Search_Plus(data_threshold,queries,candidate_size, candidate_depth)
            else:
                self.__NORA_1_BY_1(data_threshold, depth_limit)
        else:
            self.__NORA(data_threshold, depth_limit)
        if using_kd:
            for leaf in self.partition_tree.get_leaves():
                if leaf.is_irregular_shape or leaf.is_irregular_shape_parent:
                    continue
                self.__KDT(0, data_threshold, leaf)
        end_time = time.time()
        
        if return_query_cost:
            return self.partition_tree.evaluate_query_cost(queries, False)*len(queries)
        
        print("Build Time (s):", end_time-start_time)
        self.partition_tree.build_time = end_time - start_time
    
    def InitializeWithQDT(self, queries, num_dims, boundary, dataset, data_threshold):
        '''
        # should I also store the candidate cut positions in Partition Node ?
        The dimension of queries should match the dimension of boundary and dataset!
        '''
        self.partition_tree = PartitionTree(num_dims, boundary)
        self.partition_tree.name='QDT'
        self.partition_tree.pt_root.node_size = len(dataset)
        self.partition_tree.pt_root.dataset = dataset
        self.partition_tree.pt_root.queryset = queries # assume all queries overlap with the boundary
        self.partition_tree.pt_root.generate_query_MBRs()
        start_time = time.time()
        self.__QDT(data_threshold)
        end_time = time.time()
        print("Build Time (s):", end_time-start_time)
        self.partition_tree.build_time = end_time - start_time


    def InitializeWithKDT(self, num_dims, boundary, dataset, data_threshold):
        '''
        num_dims denotes the (first) number of dimension to split, usually it should correspond with the boundary
        rewrite the KDT using PartitionTree data structure
        call the recursive __KDT methods
        '''
        self.partition_tree = PartitionTree(num_dims, boundary)
        self.partition_tree.name = 'KDT'
        self.partition_tree.pt_root.node_size = len(dataset)
        self.partition_tree.pt_root.dataset = dataset
        # start from the first dimension
        start_time = time.time()
        self.__KDT(0, data_threshold, self.partition_tree.pt_root)
        end_time = time.time()
        print("Build Time (s):", end_time-start_time)
        self.partition_tree.build_time = end_time - start_time
    
    def ContinuePartitionWithKDT(self, existing_partition_tree, data_threshold):
        '''
        pass in a PartitionTree instance
        then keep partition its leaf nodes with KDT, if available
        '''
        self.partition_tree = existing_partition_tree
        leaves = existing_partition_tree.get_leaves()
        for leaf in leaves:
            self.__KDT(0, data_threshold, leaf)
    
    def CreateRtreeFilter(self, data_threshold, capacity_ratio = 0.5):
        '''
        create Rtree MBRs for leaf nodes as a filter layer for skew dataset
        '''
        for leaf in self.partition_tree.get_leaves():
            if leaf.is_irregular_shape:
                continue
            else:
                MBRs = self.__CreateRtreeMBRs(leaf.dataset, data_threshold, capacity_ratio)   
                leaf.rtree_filters = MBRs



    def RedundantPartitions(self, redundant_space, queries, dataset, data_threshold, weight = None):
        '''
        create redundant partitions to maximize the cost deduction, the extra space is limited by the redundant space
        this is a typical dynamic programming problem
        '''
        old_costs = self.partition_tree.get_queryset_cost(queries)
        spaces = self.__real_result_size(dataset, queries)
        spaces = [max(s, data_threshold) for s in spaces]
        gains = [old_costs[i]-spaces[i] for i in range(len(queries))]
        
        #print("old cost:",old_costs)
        #print("spaces:", spaces)
        #print("gains:", gains)
        
        if weight is not None: # the expected query amount
            gains = [gains[i]*weight[i] for i in range(len(queries))]
        
        max_total_gain, materialized_queries = self.__RPDP(0, gains, spaces, redundant_space, {})
        
        query_size = len(queries) if weight is None else sum(weight)
        old_query_cost = sum(old_costs)
        old_average_query_cost = old_query_cost / query_size
        new_query_cost = old_query_cost - max_total_gain
        new_average_query_cost = new_query_cost / query_size
        
        #print("max total gain:", max_total_gain)
        #print("old_query_cost:", old_query_cost, "new_query_cost:", new_query_cost)
        #print("old_average_query_cost:", old_average_query_cost, "new_average_query_cost:", new_average_query_cost)
        
        return max_total_gain, materialized_queries
    
    # = = = = = internal functions = = = = =
    
    
    def __CreateRtreeMBRs(self, dataset, data_threshold, capacity_ratio = 0.5):
    
        def DatasetGenerator(dataset):
            for i in range(len(dataset)):
                yield(i, tuple(dataset[i].tolist()+dataset[i].tolist()), dataset[i])
            return

        p = index.Property()
        p.dimension = dataset.shape[1]
        p.leaf_capacity = int(capacity_ratio * data_threshold) # cannot be less than 100, indicate the maximum capacity
        p.fill_factor = 0.9
        p.overwrite = True
        p.interleaved = False

        rtree_idx = index.Index(DatasetGenerator(dataset), properties = p)
    #     rtree_idx = index.Index(properties = p) # Rtree index for queries
    #     for i in range(len(dataset)):
    #         rtree_idx.insert(i, tuple(dataset[i].tolist()+dataset[i].tolist())) # Check whether this operation is correct !!!

        leaves = rtree_idx.leaves()
        #print(leaves)

        MBRs = [] # check whether the False interleaved property will make the mbr not interleaved? -> Result: Yes
        for leaf in leaves:
            MBRs.append(leaf[2]) # [0]: id?; [1]: [records...]; [2]: boundary

        return MBRs
    
    
    def __RPDP(self, i, gains, spaces, total_space, i_space_dict):
        '''
        i: the current query id to be considered
        total_space: the remaining redundant space
        '''
        key = (i, total_space)
        if key in i_space_dict:
            return i_space_dict[key]
        
        if i >= len(gains): # end
            return (0, [])
        
        gain, Q = None, None
        if total_space > spaces[i]:
            # create RP for this query
            (gain1, Q1) = self.__RPDP(i+1, gains, spaces, total_space-spaces[i], i_space_dict)
            (gain1, Q1) = (gains[i] + gain1, [i] + Q1)
            # do not create RP for this query
            (gain2, Q2) = self.__RPDP(i+1, gains, spaces, total_space, i_space_dict) 
            (gain, Q) = (gain1, Q1) if gain1 >= gain2 else (gain2, Q2)
        else:
            # do not create RP for this query
            (gain, Q) = self.__RPDP(i+1, gains, spaces, total_space, i_space_dict)
        
        i_space_dict[key] = (gain, Q)
        return (gain, Q)
        
    
    def __real_result_size(self, dataset, queries):
        num_dims = dataset.shape[1]
        results = []
        for query in queries:
            constraints = []
            for d in range(num_dims):
                constraint_L = dataset[:,d] >= query[d]
                constraint_U = dataset[:,d] <= query[num_dims + d]
                constraints.append(constraint_L)
                constraints.append(constraint_U)
            constraint = np.all(constraints, axis=0)
            result_size = np.count_nonzero(constraint)
            results.append(result_size)
        return results
    
    def __max_bound(self, num_dims, queryset):
        '''
        bound the queries by their maximum bounding rectangle
        '''
        max_bound_L = np.amin(np.array(queryset)[:,0:num_dims],axis=0).tolist()
        max_bound_U = np.amax(np.array(queryset)[:,num_dims:],axis=0).tolist()
        max_bound = max_bound_L + max_bound_U # concat
        return max_bound
    

    def __PAW_Beam_Search(self, data_threshold, queries, max_active_ratio = 3, strategy = 1, candidate_size = 1, 
                          candidate_depth = 1, depth_limit = None, beam_search_mode = 0):
        
        '''
        strategy: related to group split
        beam_search_mode: 0 = subsequent calls also use beam search; 1 = subsequent calls only use the best
        '''
        
        if depth_limit is not None and depth_limit < candidate_depth:
            print("-> Enter beam search for leaf", self.partition_tree.pt_root.nid, "depth_limit:", depth_limit,"/", candidate_depth)
        
        def len_DQ(someset):
            if someset is not None:
                return len(someset)
            return 0
        
        CanSplit = True
        while CanSplit:
            CanSplit = False           
            
            # DO NOT CONSIDER Partitionable!
            leaves = self.partition_tree.get_leaves()
            #print("# number of leaf nodes:",len(leaves))
            
            whole_partition_cost = 0
            
            for leaf in leaves:
                
                if leaf.node_size < 2 * data_threshold or leaf.queryset is None or leaf.is_irregular_shape or leaf.no_valid_partition:
                    whole_partition_cost += len_DQ(leaf.queryset) * leaf.node_size
                    print("X IGNORE leaf", leaf.nid, "leaf node dataset", len_DQ(leaf.dataset), "queryset size:",len_DQ(leaf.queryset))
                    print("[ignore leaf] current whole partition cost:", whole_partition_cost, " + step:", len_DQ(leaf.queryset) * leaf.node_size)
                    continue
                print("O CONSIDER leaf node id:",leaf.nid, "leaf node dataset", len_DQ(leaf.dataset), "queryset size:",len_DQ(leaf.queryset))
                    
                split_candidates = []
                skip = 0
                
                # always extend the candidate cut with medians
                candidate_cuts = leaf.get_candidate_cuts(True)

                # try general group split
                if leaf.node_size <= max_active_ratio * data_threshold:
                    if strategy == 0:
                        skip = leaf.if_general_group_split(data_threshold)
                    else:
                        skip=leaf.if_general_group_split_2(data_threshold)
                    split_candidates.append((skip, None, None, 3))

                for split_dim, split_value in candidate_cuts:

                    # first try normal split
                    valid, skip, left_size, right_size = leaf.if_split(split_dim, split_value, data_threshold)
                    if valid:
                        split_candidates.append((skip, split_dim, split_value, 0))
                        
                split_candidates.sort(key=lambda item: item[0], reverse=True) # from the most skip gain to least
                
                best_split = None
                min_cost = float('inf')
                
                if depth_limit is None:
                    depth_limit = candidate_depth
                if depth_limit == 0:
                    print(f"candidate cuts:{len(candidate_cuts)} split_candidates:{len(split_candidates)}")
                    # return the cost of max skip split directly       
                    leaf_cost = len(leaf.queryset) * leaf.node_size - split_candidates[0][0] # original cost - skip = after split cost
                    whole_partition_cost += leaf_cost
                    print("[depth limit 0] current whole partition cost:", whole_partition_cost, "+ step:",leaf_cost,"skip:",split_candidates[0][0])
                    if candidate_depth != 0:
                        continue
                    else:
                        best_split = split_candidates[0]
                        min_cost = leaf_cost
                    #return self.partition_tree.evaluate_query_cost(training_set, False)
                if candidate_depth != 0:
                    print("START beam search for node", leaf.nid)
                    if beam_search_mode == 0 or depth_limit == candidate_depth:
                        # num_beams = min(candidate_size, len(split_candidates))
                        for i in range(min(candidate_size, len(split_candidates))):

                            # I need to copy myself for the split
                            temp_node = copy.deepcopy(leaf)
                            temp_queries = copy.deepcopy(leaf.queryset)

                            # apply this split on temp node
                            temp_tree = PartitionTree(temp_node.num_dims, temp_node.boundary)
                            temp_tree.pt_root = temp_node
                            temp_tree.nid_node_dict = {temp_node.nid:  temp_node}
                            temp_tree.node_count = temp_node.nid + 1
                            temp_tree.apply_split(temp_node.nid, split_candidates[i][1], split_candidates[i][2], split_candidates[i][3])

                            temp_algo = PartitionAlgorithm()
                            temp_algo.partition_tree = temp_tree

                            child_cost = temp_algo._PartitionAlgorithm__PAW_Beam_Search(data_threshold, temp_queries, max_active_ratio, 
                                                                                        strategy, candidate_size, candidate_depth, 
                                                                                        depth_limit-1, beam_search_mode)
                            print("<- beam search result cost for leaf", leaf.nid, "cut",i, "type:",split_candidates[i][3],"cost:",child_cost)
                            if child_cost < min_cost:
                                best_split = split_candidates[i]
                                min_cost = child_cost
                                
                    elif beam_search_mode == 1:
                    # num_beams = 1
                        temp_node = copy.deepcopy(leaf)
                        temp_queries = copy.deepcopy(leaf.queryset)

                        # apply this split on temp node
                        temp_tree = PartitionTree(temp_node.num_dims, temp_node.boundary)
                        temp_tree.pt_root = temp_node
                        temp_tree.nid_node_dict = {temp_node.nid:  temp_node}
                        temp_tree.node_count = temp_node.nid + 1
                        temp_tree.apply_split(temp_node.nid, split_candidates[0][1], split_candidates[0][2], split_candidates[0][3])

                        temp_algo = PartitionAlgorithm()
                        temp_algo.partition_tree = temp_tree

                        child_cost = temp_algo._PartitionAlgorithm__PAW_Beam_Search(data_threshold, temp_queries, max_active_ratio, 
                                                                                    strategy, candidate_size, candidate_depth, 
                                                                                    depth_limit-1, beam_search_mode)
                        min_cost = child_cost

                    if depth_limit != candidate_depth:
                        whole_partition_cost += min_cost
                        print("[after beam search] current whole partition cost:", whole_partition_cost, "+step:",min_cost)
                        continue
                        #return min_cost
                
                # if this is the current level (i.e., after the beam search returned), apply the search result
                print("= = = AFTER beam search for node", leaf.nid)
                if min_cost < len(leaf.queryset) * leaf.node_size and best_split is not None and best_split[0] > 0:
                    # if the cost become smaller, apply the cut
                    self.partition_tree.apply_split(leaf.nid, best_split[1], best_split[2], best_split[3])
                    print("= = = SPLIT on node id:", leaf.nid, "split type:", best_split[3])
                    CanSplit = True
                else:
                    print("leaf node id:",leaf.nid,"does not have valid split")
                    leaf.no_valid_partition = True
            
            if depth_limit != candidate_depth and candidate_depth != 0:
                return whole_partition_cost
            
        # if it already met the bottom of the tree, (even it hasn't met the depth limit)
        return int(self.partition_tree.evaluate_query_cost(queries) * len(queries))

    def __PAW_Beam_Search2(self, data_threshold, queries, max_active_ratio=3, strategy=1, candidate_size=1,
                          candidate_depth=1, depth_limit=None, beam_search_mode=0):
        '''
            simplify beam search
        '''
        CanSplit = True
        depth=candidate_depth
        while CanSplit:
            CanSplit = False
            # only mode=1 or 3, we stop the cut for leafs every when depth=0
            if beam_search_mode == 1 and depth<1:
                break
            # DO NOT CONSIDER Partitionable!
            leaves = self.partition_tree.get_leaves()
            # print("# number of leaf nodes:",len(leaves))

            for leaf in leaves:

                if leaf.node_size < 2 * data_threshold or leaf.queryset is None or leaf.is_irregular_shape or leaf.no_valid_partition:
                    continue

                split_candidates = []
                skip = 0

                # always extend the candidate cut with medians
                candidate_cuts = leaf.get_candidate_cuts(True)

                # try general group split
                if leaf.node_size <= max_active_ratio * data_threshold:
                    if strategy == 0:
                        skip = leaf.if_general_group_split_2(data_threshold)
                    else:
                        skip = leaf.if_general_group_split_3(data_threshold)
                    split_candidates.append((skip, None, None, 3))

                for split_dim, split_value in candidate_cuts:

                    # first try normal split
                    valid, skip, left_size, right_size = leaf.if_split(split_dim, split_value, data_threshold)
                    if valid:
                        split_candidates.append((skip, split_dim, split_value, 0))

                split_candidates.sort(key=lambda item: item[0], reverse=True)  # from the most skip gain to least

                best_split = None
                min_cost = float('inf')
                # when mode=3, we begin the max cut for leafs; when mode=1 and depth<1, we begin the max cut for leafs
                if (beam_search_mode==0 and depth<1) or beam_search_mode==3:
                    best_split=split_candidates[0]
                    min_cost=-1
                else:
                    num_beams = min(candidate_size, len(split_candidates))
                    for i in range(num_beams):

                        # I need to copy myself for the split
                        temp_node = copy.deepcopy(leaf)
                        temp_queries = copy.deepcopy(leaf.queryset)

                        # apply this split on temp node
                        temp_tree = PartitionTree(temp_node.num_dims, temp_node.boundary)
                        temp_tree.pt_root = temp_node
                        temp_tree.nid_node_dict = {temp_node.nid: temp_node}
                        temp_tree.node_count = temp_node.nid + 1
                        temp_tree.apply_split(temp_node.nid, split_candidates[i][1], split_candidates[i][2],
                                              split_candidates[i][3])
                        temp_algo = PartitionAlgorithm()
                        temp_algo.partition_tree = temp_tree
                        # if beam_search_mode == 0:
                        #     child_cost = temp_algo._PartitionAlgorithm__PAW_Beam_Search2(data_threshold, temp_queries,
                        #                                                                  max_active_ratio,
                        #                                                                  strategy, candidate_size,
                        #                                                                  candidate_depth-1,
                        #                                                                  None,
                        #                                                                  beam_search_mode=1)
                        #     print("<- beam search result cost for leaf", leaf.nid, "cut", i,
                        #                                     "type:",
                        #                                     split_candidates[i][3], "cost:", child_cost)
                        # else:
                        child_cost = temp_algo._PartitionAlgorithm__PAW_Beam_Search2(data_threshold, temp_queries,
                                                                                     max_active_ratio,
                                                                                     strategy, candidate_size,
                                                                                     candidate_depth,
                                                                                     None,
                                                                                     beam_search_mode=3)
                        if child_cost < min_cost:
                            best_split = split_candidates[i]
                            min_cost = child_cost

                # if this is the current level (i.e., after the beam search returned), apply the search result
                if beam_search_mode==0: print("= = = AFTER beam search for node", leaf.nid)
                if min_cost < len(leaf.queryset) * leaf.node_size and best_split is not None and best_split[0] > 0:
                    # if the cost become smaller, apply the cut
                    self.partition_tree.apply_split(leaf.nid, best_split[1], best_split[2], best_split[3])
                    if beam_search_mode==0: print("= = = SPLIT on node id:", leaf.nid, "split type:", best_split[3])
                    CanSplit = True
                else:
                    if beam_search_mode==0: print("leaf node id:", leaf.nid, "does not have valid split")
                    leaf.no_valid_partition = True
            depth-=1
        # if it already met the bottom of the tree, (even it hasn't met the depth limit)
        return int(self.partition_tree.evaluate_query_cost(queries) * len(queries))


    def __PAW(self, data_threshold, max_active_ratio = 3, strategy = 1):
        '''
        using Ken's Algorithm 1: iteratively bound 2 MBRs if they are less than b
        max_active_ratio: when the partition size is <= ratio * b: consider group split
        strategy: 0 = using merge without considering overlap; 1 = using extend, aborted when overlapped
        ''' 
        CanSplit = True
        while CanSplit:
            CanSplit = False           
            
            # DO NOT CONSIDER Partitionable!
            leaves = self.partition_tree.get_leaves()
            #print("# number of leaf nodes:",len(leaves))
            for leaf in leaves:
                
                #print("current leaf node id:",leaf.nid, "leaf node dataset size:", leaf.node_size)
                if leaf.node_size < 2 * data_threshold or leaf.queryset is None or leaf.is_irregular_shape or leaf.no_valid_partition:
                    continue
                    
                # get best candidate cut position
                skip, max_skip, max_skip_split_dim, max_skip_split_value, max_skip_split_type = 0, 0, 0, 0, 0
                
                # always extend the candidate cut with medians
                candidate_cuts = leaf.get_candidate_cuts(True)
                
                # try general group split
                if leaf.node_size <= max_active_ratio * data_threshold:
                    if strategy == 0:
                        skip = leaf.if_general_group_split(data_threshold)
                    elif strategy == 1:
                        skip = leaf.if_general_group_split_2(data_threshold)
                        
                    #print("PAW: general group split is tried")
                    if skip > max_skip:
                        max_skip, max_skip_split_dim, max_skip_split_value, max_skip_split_type = skip, None, None, 3
                
                for split_dim, split_value in candidate_cuts:
                    # first try normal split
                    valid, skip, left_size, right_size = leaf.if_split(split_dim, split_value, data_threshold)
                    if valid and skip > max_skip:
                        max_skip, max_skip_split_dim, max_skip_split_value, max_skip_split_type = skip, split_dim, split_value, 0

                if max_skip > 0:
                    # if the cost become smaller, apply the cut
                    child_node1, child_node2 = self.partition_tree.apply_split(leaf.nid, max_skip_split_dim, max_skip_split_value, max_skip_split_type)
                    if max_skip_split_type==3:
                        print("Group Split is used!!!")
                    #print(" Split on node id:", leaf.nid, "split type:", max_skip_split_type, "split gain:", max_skip, "split value:", max_skip_split_value, "split dim:",max_skip_split_dim)
                    #print("after split, child node 1 queryset size:",len(child_node1.queryset), "child node 2 queryset size:", len(child_node2.queryset))
                    CanSplit = True
                else:
                    #print("leaf node id:",leaf.nid,"does not have valid split")
                    leaf.no_valid_partition = True

    # def __PAW2(self, data_threshold, max_active_ratio=3):
    #     '''
    #     using Ken's Algorithm 1: iteratively bound 2 MBRs if they are less than b
    #     max_active_ratio: when the partition size is <= ratio * b: consider group split
    #     strategy: 0 = using merge without considering overlap; 1 = using extend, aborted when overlapped
    #     '''
    #     CanSplit = True
    #     while CanSplit:
    #         CanSplit = False
    #
    #         # DO NOT CONSIDER Partitionable!
    #         leaves = self.partition_tree.get_leaves()
    #         # print("# number of leaf nodes:",len(leaves))
    #         for leaf in leaves:
    #             # print("current leaf node id:",leaf.nid, "leaf node dataset size:", leaf.node_size)
    #             if leaf.node_size < 2 * data_threshold or leaf.queryset is None or leaf.is_irregular_shape or leaf.no_valid_partition:
    #                 continue
    #
    #             # get best candidate cut position
    #             skip, max_skip, max_skip_split_dim, max_skip_split_value, max_skip_split_type = 0, 0, 0, 0, 0
    #
    #             # always extend the candidate cut with medians
    #             candidate_cuts = leaf.get_candidate_cuts(True)
    #
    #             # try general group split
    #             if leaf.node_size <= max_active_ratio * data_threshold:
    #                 skip = leaf.if_general_group_split_2(data_threshold)
    #
    #                 # print("PAW: general group split is tried")
    #                 if skip > max_skip:
    #                     max_skip, max_skip_split_dim, max_skip_split_value, max_skip_split_type = skip, None, None, 3
    #
    #             for split_dim, split_value in candidate_cuts:
    #                 # first try normal split
    #                 valid, skip, left_size, right_size = leaf.if_split(split_dim, split_value, data_threshold)
    #                 if valid and skip > max_skip:
    #                     max_skip, max_skip_split_dim, max_skip_split_value, max_skip_split_type = skip, split_dim, split_value, 0
    #
    #             if max_skip > 0:
    #                 # if the cost become smaller, apply the cut
    #                 child_node1, child_node2 = self.partition_tree.apply_split(leaf.nid, max_skip_split_dim,
    #                                                                            max_skip_split_value,
    #                                                                            max_skip_split_type)
    #                 if max_skip_split_type == 3:
    #                     print("Group Split is used!!!")
    #                 # print(" Split on node id:", leaf.nid, "split type:", max_skip_split_type, "split gain:", max_skip, "split value:", max_skip_split_value, "split dim:",max_skip_split_dim)
    #                 # print("after split, child node 1 queryset size:",len(child_node1.queryset), "child node 2 queryset size:", len(child_node2.queryset))
    #                 CanSplit = True
    #             else:
    #                 # print("leaf node id:",leaf.nid,"does not have valid split")
    #                 leaf.no_valid_partition = True

    def __NORA_Beam_Search_Plus(self, data_threshold, queries, candidate_size=1,candidate_depth=1, depth_limit=None, beam_search_mode=0):
        if depth_limit is not None and depth_limit < candidate_depth:
            print("-> Enter beam search for leaf", self.partition_tree.pt_root.nid, "depth_limit:", depth_limit, "/",
                  candidate_depth)

        def len_DQ(someset):
            if someset is not None:
                return len(someset)
            return 0

        CanSplit = True
        while CanSplit:
            CanSplit = False

            # DO NOT CONSIDER Partitionable!
            leaves = self.partition_tree.get_leaves(use_partitionable = True)
            # print("# number of leaf nodes:",len(leaves))

            whole_partition_cost = 0

            for leaf in leaves:

                if leaf.node_size < 2 * data_threshold or leaf.queryset is None:
                    whole_partition_cost += len_DQ(leaf.queryset) * leaf.node_size
                    print("X IGNORE leaf", leaf.nid, "leaf node dataset", len_DQ(leaf.dataset), "queryset size:",
                          len_DQ(leaf.queryset))
                    print("[ignore leaf] current whole partition cost:", whole_partition_cost, " + step:",
                          len_DQ(leaf.queryset) * leaf.node_size)
                    continue
                print("O CONSIDER leaf node id:", leaf.nid, "leaf node dataset", len_DQ(leaf.dataset), "queryset size:",
                      len_DQ(leaf.queryset))

                if len(leaf.queryset) == 1:
                    # try bounding split here
                    valid, skip, bound = leaf.if_bounding_split(data_threshold, approximate=False)

                    if valid:
                        # apply this split
                        # print(" = = = Apply Old Bounding Split = = = ")
                        self.partition_tree.apply_split(leaf.nid, None, None, 1, bound)
                        # print("!!!Split From Internal Bounding Split!!!")
                        continue
                        # design an variant of bounding split, which will handle node size problem # done
                        # if valid, mark the children as  don't consider anymore" -> partitionable = False

                if leaf.if_new_bounding_split(data_threshold):
                    # print(" = = = Apply New Bounding Split = = = ")
                    # create new bounding split sub-partitions
                    self.partition_tree.apply_split(leaf.nid, None, None, 3)
                    continue

                skip, max_skip, max_skip_split_dim, max_skip_split_value, max_skip_split_type = 0, -1, 0, 0, 0
                # always extend the candidate cut with medians
                candidate_cuts = leaf.get_candidate_cuts(True)
                split_candidates = []

                for split_dim, split_value in candidate_cuts:

                    # first try normal split
                    valid, skip, left_size, right_size = leaf.if_split(split_dim, split_value, data_threshold)
                    if valid:
                        num_query_crossed = leaf.num_query_crossed(split_dim, split_value)
                        split_candidates.append((skip, split_dim, split_value, 0, valid, num_query_crossed))

                    # Should we remove the leaf node case here in 1 BY 1?
                    # the following cases here are applied only for leaf nodes
                    # if it's available for bounding split, try it
                    if leaf.node_size < 3 * data_threshold:
                        # try bounding split
                        valid, skip, _ = leaf.if_bounding_split(data_threshold, approximate=False)
                        if valid and skip > max_skip:
                            split_candidates.append((skip, split_dim, split_value, 1, valid, 0))
                            # max_skip, max_skip_split_dim, max_skip_split_value, max_skip_split_type = skip, split_dim, split_value, 1

                    # if it's availble for dual-bounding split, try it
                    elif leaf.node_size < 4 * data_threshold and left_size < 2 * data_threshold and right_size < 2 * data_threshold:
                        # try dual-bounding split
                        valid, skip = leaf.if_dual_bounding_split(split_dim, split_value, data_threshold,
                                                                  approximate=False)
                        if valid and skip > max_skip:
                            split_candidates.append((skip, split_dim, split_value, 1, valid, 2))
                            # max_skip, max_skip_split_dim, max_skip_split_value, max_skip_split_type = skip, split_dim, split_value, 2

                if len(split_candidates) == 0:
                    continue

                split_candidates.sort(key=lambda item: item[0], reverse=True)  # from the most skip gain to least
                best_split = None
                min_cost = float('inf')

                if depth_limit is None:
                    depth_limit = candidate_depth
                if depth_limit == 0:
                    print(f"candidate cuts:{len(candidate_cuts)} split_candidates:{len(split_candidates)}")
                    # return the cost of max skip split directly
                    leaf_cost = len(leaf.queryset) * leaf.node_size - split_candidates[0][
                        0]  # original cost - skip = after split cost
                    whole_partition_cost += leaf_cost
                    print("[depth limit 0] current whole partition cost:", whole_partition_cost, "+ step:", leaf_cost,
                          "skip:", split_candidates[0][0])
                    if candidate_depth != 0:
                        continue
                    else:
                        best_split = split_candidates[0]
                        min_cost = leaf_cost
                    # return self.partition_tree.evaluate_query_cost(training_set, False)
                if candidate_depth != 0:
                    print("START beam search for node", leaf.nid)
                    if beam_search_mode == 0 or depth_limit == candidate_depth:
                        # num_beams = min(candidate_size, len(split_candidates))
                        for i in range(min(candidate_size, len(split_candidates))):

                            # I need to copy myself for the split
                            temp_node = copy.deepcopy(leaf)
                            temp_queries = copy.deepcopy(leaf.queryset)

                            # apply this split on temp node
                            temp_tree = PartitionTree(temp_node.num_dims, temp_node.boundary)
                            temp_tree.pt_root = temp_node
                            temp_tree.nid_node_dict = {temp_node.nid: temp_node}
                            temp_tree.node_count = temp_node.nid + 1
                            temp_tree.apply_split(temp_node.nid, split_candidates[i][1], split_candidates[i][2],
                                                  split_candidates[i][3])

                            temp_algo = PartitionAlgorithm()
                            temp_algo.partition_tree = temp_tree

                            child_cost = temp_algo._PartitionAlgorithm__NORA_Beam_Search_Plus(data_threshold, temp_queries,
                                                                                         candidate_size,
                                                                                        candidate_depth,
                                                                                        depth_limit - 1,
                                                                                        beam_search_mode)
                            print("<- beam search result cost for leaf", leaf.nid, "cut", i, "type:",
                                  split_candidates[i][3], "cost:", child_cost)
                            if child_cost < min_cost:
                                best_split = split_candidates[i]
                                min_cost = child_cost

                    elif beam_search_mode == 1:
                        # num_beams = 1
                        temp_node = copy.deepcopy(leaf)
                        temp_queries = copy.deepcopy(leaf.queryset)

                        # apply this split on temp node
                        temp_tree = PartitionTree(temp_node.num_dims, temp_node.boundary)
                        temp_tree.pt_root = temp_node
                        temp_tree.nid_node_dict = {temp_node.nid: temp_node}
                        temp_tree.node_count = temp_node.nid + 1
                        temp_tree.apply_split(temp_node.nid, split_candidates[0][1], split_candidates[0][2],
                                              split_candidates[0][3])

                        temp_algo = PartitionAlgorithm()
                        temp_algo.partition_tree = temp_tree

                        child_cost = temp_algo._PartitionAlgorithm__NORA_Beam_Search_Plus(data_threshold, temp_queries,
                                                                                     candidate_size,
                                                                                    candidate_depth,
                                                                                    depth_limit - 1,
                                                                                    beam_search_mode)
                        min_cost = child_cost

                    if depth_limit != candidate_depth:
                        whole_partition_cost += min_cost
                        print("[after beam search] current whole partition cost:", whole_partition_cost, "+step:",
                              min_cost)
                        continue
                        # return min_cost

                # if this is the current level (i.e., after the beam search returned), apply the search result
                print("= = = AFTER beam search for node", leaf.nid)
                if min_cost < len(leaf.queryset) * leaf.node_size and best_split is not None and best_split[0] > 0:
                    # if the cost become smaller, apply the cut
                    self.partition_tree.apply_split(leaf.nid, best_split[1], best_split[2], best_split[3])
                    print("= = = SPLIT on node id:", leaf.nid, "split type:", best_split[3])
                    CanSplit = True
                else:
                    print("leaf node id:", leaf.nid, "does not have valid split")
                    leaf.no_valid_partition = True

            if depth_limit != candidate_depth and candidate_depth != 0:
                return whole_partition_cost

        # if it already met the bottom of the tree, (even it hasn't met the depth limit)
        return int(self.partition_tree.evaluate_query_cost(queries) * len(queries))

    def __NORA_Beam_Search(self, data_threshold, candidate_size = 1, candidate_depth = 2):
        '''
        using beam search to improve the search space
        candidate_size: how many candidate splits we maintain in a layer
        candidate_depth: how many layers we keep during the search
        '''
        '''
        the NORA algorithm that optimized for 1 by 1 mapping scenario
        '''
        
        CanSplit = True
        while CanSplit:
            CanSplit = False
            # for leaf in self.partition_tree.get_leaves():
            leaves = self.partition_tree.get_leaves(use_partitionable = True) # there could be large irregular shape partitions
            #print("# number of leaf nodes:",len(leaves))
            for leaf in leaves:
                # print("current leaf node id:",leaf.nid, "leaf node dataset size:",len(leaf.dataset))
                if leaf.node_size < 2 * data_threshold or leaf.queryset is None:
                    continue
                if len(leaf.queryset) == 1:
                    # try bounding split here
                    valid, skip, bound = leaf.if_bounding_split(data_threshold, approximate = False)
                    if valid:
                        # apply this split
                        #print(" = = = Apply Old Bounding Split = = = ")
                        self.partition_tree.apply_split(leaf.nid, None, None, 1, bound)
                        #print("!!!Split From Internal Bounding Split!!!")
                        continue
                        # design an variant of bounding split, which will handle node size problem # done
                        # if valid, mark the children as  don't consider anymore" -> partitionable = False
                if leaf.node_size <= 3 * data_threshold:
                    if leaf.if_new_bounding_split(data_threshold):
                        #print(" = = = Apply New Bounding Split = = = ")
                        # create new bounding split sub-partitions
                        self.partition_tree.apply_split(leaf.nid, None, None, 3)
                        continue

                # get best candidate cut position
                skip, max_skip, max_skip_split_dim, max_skip_split_value, max_skip_split_type = 0, -1, 0, 0, 0
                # extend the candidate cut with medians when it reach the bottom
                #candidate_cuts = leaf.get_candidate_cuts(True) if leaf.node_size < 4 * data_threshold else leaf.get_candidate_cuts()     
                candidate_cuts = leaf.get_candidate_cuts(True) # try extends it always
                
                split_candidates = []
                for split_dim, split_value in candidate_cuts:
                    # first try normal split
                    valid, skip, left_size, right_size = leaf.if_split(split_dim, split_value, data_threshold)
                    #if valid and skip > max_skip:
                    #    max_skip, max_skip_split_dim, max_skip_split_value, max_skip_split_type = skip, split_dim, split_value, 0
                    if valid:
                        num_query_crossed = leaf.num_query_crossed(split_dim, split_value)
                        split_candidates.append((skip, split_dim, split_value, 0, valid, num_query_crossed))
                    
                    # Should we remove the leaf node case here in 1 BY 1?
                    # the following cases here are applied only for leaf nodes
                    # if it's available for bounding split, try it
                    if leaf.node_size < 3 * data_threshold:
                        # try bounding split
                        valid, skip,_ = leaf.if_bounding_split(data_threshold, approximate = False)
                        if valid and skip > max_skip:
                            split_candidates.append((skip, split_dim, split_value, 1, valid, 0))
                            #max_skip, max_skip_split_dim, max_skip_split_value, max_skip_split_type = skip, split_dim, split_value, 1

                    # if it's availble for dual-bounding split, try it
                    elif leaf.node_size < 4 * data_threshold and left_size < 2 * data_threshold and right_size < 2 * data_threshold:
                        # try dual-bounding split
                        valid, skip = leaf.if_dual_bounding_split(split_dim, split_value, data_threshold, approximate = False)
                        if valid and skip > max_skip:
                            split_candidates.append((skip, split_dim, split_value, 1, valid, 2))
                            #max_skip, max_skip_split_dim, max_skip_split_value, max_skip_split_type = skip, split_dim, split_value, 2
                
                if len(split_candidates) == 0:
                    continue
                split_candidates.sort(key=lambda item: item[0], reverse=True) # from the most skip gain to least
                found_available_split = False
                split_context = None
                split_context = split_candidates[0]
                
                # Beam search on the first few candidates
                min_cost = leaf.node_size * len(leaf.queryset)
                for i in range(min(candidate_size, len(split_candidates))):
                    # explored_cost = pass
                    # first create a partition tree using the current partition
                    # then split it with depth constraint!
                    # third, analyze the cost, maybe I just used the minimum total (avg) cost for comparison
                    
                    fake_split = split_candidates[i]
                    left_child, right_child = self.partition_tree.apply_split(leaf.nid, fake_split[1], fake_split[2], fake_split[3],
                                                                             pretend = True) # do not actually apply this split
                    temp_left_partition = PartitionAlgorithm()
                    temp_right_partition = PartitionAlgorithm()
                    if left_child.queryset:
                        # WAIT! , it should be the left cost + the right cost instead of directly using the current leaf!
                        left_cost = temp_left_partition.InitializeWithNORA(left_child.queryset, leaf.num_dims, left_child.boundary,
                                                                           left_child.dataset, data_threshold, using_1_by_1 = False,
                                                                           using_kd = False, depth_limit = candidate_depth,
                                                                           return_query_cost = True)
                    else: left_cost=0
                    if right_child.queryset:
                        right_cost = temp_right_partition.InitializeWithNORA(right_child.queryset, leaf.num_dims, right_child.boundary,
                                                                         right_child.dataset, data_threshold, using_1_by_1 = False, 
                                                                         using_kd = False, depth_limit = candidate_depth, 
                                                                         return_query_cost = True)
                    else: right_cost=0
                    explored_cost = left_cost + right_cost
                    if explored_cost < min_cost:
                        min_cost = explored_cost
                        split_context = split_candidates[i]          
                
                # apply split context
                if split_context[0] > 0:
                    self.partition_tree.apply_split(leaf.nid, split_context[1], split_context[2], split_context[3])
                    #print(" Split on node id:", leaf.nid)
                    #print("split_context:",split_context)
                    CanSplit = True

    def __NORA_1_BY_1(self, data_threshold, depth_limit = None):
        '''
        the NORA algorithm that optimized for 1 by 1 mapping scenario
        '''
        CanSplit = True
        while CanSplit:
            CanSplit = False           
            
            # for leaf in self.partition_tree.get_leaves():
            leaves = self.partition_tree.get_leaves(use_partitionable = True) # there could be large irregular shape partitions
            #print("# number of leaf nodes:",len(leaves))
            for leaf in leaves:
                
                # print("current leaf node id:",leaf.nid, "leaf node dataset size:",len(leaf.dataset))
                if leaf.node_size < 2 * data_threshold or leaf.queryset is None or (depth_limit is not None and leaf.depth >= depth_limit):
                    continue
                
                if len(leaf.queryset) == 1:
                    # try bounding split here
                    valid, skip, bound = leaf.if_bounding_split(data_threshold, approximate = False)
                    
                    if valid:
                        # apply this split
                        #print(" = = = Apply Old Bounding Split = = = ")
                        self.partition_tree.apply_split(leaf.nid, None, None, 1, bound)
                        #print("!!!Split From Internal Bounding Split!!!")
                        continue    
                        # design an variant of bounding split, which will handle node size problem # done
                        # if valid, mark the children as  don't consider anymore" -> partitionable = False
                   
                if leaf.if_new_bounding_split(data_threshold):
                    #print(" = = = Apply New Bounding Split for node:", leaf.nid)
                    # create new bounding split sub-partitions
                    self.partition_tree.apply_split(leaf.nid, None, None, 3)
                    continue
                
                # get best candidate cut position
                skip, max_skip, max_skip_split_dim, max_skip_split_value, max_skip_split_type = 0, -1, 0, 0, 0
                # extend the candidate cut with medians when it reach the bottom
                #candidate_cuts = leaf.get_candidate_cuts(True) if leaf.node_size < 4 * data_threshold else leaf.get_candidate_cuts()     
                candidate_cuts = leaf.get_candidate_cuts(True) # try extends it always
                
                split_candidates = []
                for split_dim, split_value in candidate_cuts:

                    # first try normal split
                    valid, skip, left_size, right_size = leaf.if_split(split_dim, split_value, data_threshold)
                    #if valid and skip > max_skip:
                    #    max_skip, max_skip_split_dim, max_skip_split_value, max_skip_split_type = skip, split_dim, split_value, 0
                    if valid:
                        num_query_crossed = leaf.num_query_crossed(split_dim, split_value)
                        split_candidates.append((skip, split_dim, split_value, 0, valid, num_query_crossed))
                    
                    # Should we remove the leaf node case here in 1 BY 1?
                    # the following cases here are applied only for leaf nodes
                    # if it's available for bounding split, try it
                    if leaf.node_size < 3 * data_threshold:
                        # try bounding split
                        valid, skip,_ = leaf.if_bounding_split(data_threshold, approximate = False)
                        if valid and skip > max_skip:
                            split_candidates.append((skip, split_dim, split_value, 1, valid, 0))
                            #max_skip, max_skip_split_dim, max_skip_split_value, max_skip_split_type = skip, split_dim, split_value, 1

                    # if it's availble for dual-bounding split, try it
                    elif leaf.node_size < 4 * data_threshold and left_size < 2 * data_threshold and right_size < 2 * data_threshold:
                        # try dual-bounding split              
                        valid, skip = leaf.if_dual_bounding_split(split_dim, split_value, data_threshold, approximate = False)
                        if valid and skip > max_skip:
                            split_candidates.append((skip, split_dim, split_value, 1, valid, 2))
                            #max_skip, max_skip_split_dim, max_skip_split_value, max_skip_split_type = skip, split_dim, split_value, 2
                
                if len(split_candidates) == 0:
                    continue
                
                split_candidates.sort(key=lambda item: item[0], reverse=True) # from the most skip gain to least
                found_available_split = False
                split_context = None
                
                # only apply QAVP on internal nodes
#                 if leaf.node_size > 4 * data_threshold:
#                     for split in split_candidates:
#                         if split[-1] == 0: # num_query_cross = 0 and valid = True
#                             # apply this split
#                             found_available_split = True
#                             split_context = split

#                     if not found_available_split:
#                         # idea 1: compare the cost of apply bounding and split the rest by general NORA

#                         # idea 2: use the least crossed 
#                         #split_candidates.sort(key=lambda item: (item[-1], -item[0]))
#                         #split_context = split_candidates[0] # use the first one for split

#                         # idea 3: just use the maximum skip one, currently I think this one could be easier
#                         split_context = split_candidates[0]        
#                 else:
#                     split_context = split_candidates[0]

                split_context = split_candidates[0]
    
                # apply split context
                if split_context[0] > 0:
                    self.partition_tree.apply_split(leaf.nid, split_context[1], split_context[2], split_context[3])
                    #print(" Split on node id:", leaf.nid)
                    #print("split_context:",split_context)
                    CanSplit = True
                
#                 if max_skip > 0:
#                     # if the cost become smaller, apply the cut
#                     child_node1, child_node2 = self.partition_tree.apply_split(leaf.nid, max_skip_split_dim, max_skip_split_value, max_skip_split_type)
#                     print(" Split on node id:", leaf.nid)
#                     CanSplit = True
    
    def __NORA(self, data_threshold, depth_limit = None):
        '''
        the general NORA algorithm, which utilize bounding split, daul-bounding split and extend candidate cuts with medians
        '''
        CanSplit = True
        while CanSplit:
            CanSplit = False           
            
            # for leaf in self.partition_tree.get_leaves():
            leaves = self.partition_tree.get_leaves()
            #print("# number of leaf nodes:",len(leaves))
            for leaf in leaves:
                
                # print("current leaf node id:",leaf.nid, "leaf node dataset size:",len(leaf.dataset))
                if leaf.node_size < 2 * data_threshold or leaf.queryset is None or (depth_limit is not None and leaf.depth >= depth_limit):
                    continue
                    
                # get best candidate cut position
                skip, max_skip, max_skip_split_dim, max_skip_split_value, max_skip_split_type = 0, -1, 0, 0, 0
                # extend the candidate cut with medians when it reach the bottom
                candidate_cuts = leaf.get_candidate_cuts(True) if leaf.node_size < 4 * data_threshold else leaf.get_candidate_cuts()     
                             
                for split_dim, split_value in candidate_cuts:

                    # first try normal split
                    valid, skip, left_size, right_size = leaf.if_split(split_dim, split_value, data_threshold)
                    if valid and skip > max_skip:
                        max_skip, max_skip_split_dim, max_skip_split_value, max_skip_split_type = skip, split_dim, split_value, 0

                    # if it's available for bounding split, try it
                    if leaf.node_size < 3 * data_threshold:
                        # try bounding split
                        valid, skip,_ = leaf.if_bounding_split(data_threshold, approximate = False)
                        if valid and skip > max_skip:
                            max_skip, max_skip_split_dim, max_skip_split_value, max_skip_split_type = skip, split_dim, split_value, 1

                    # if it's availble for dual-bounding split, try it
                    elif leaf.node_size < 4 * data_threshold and left_size < 2 * data_threshold and right_size < 2 * data_threshold:
                        # try dual-bounding split              
                        valid, skip = leaf.if_dual_bounding_split(split_dim, split_value, data_threshold, approximate = False)
                        if valid and skip > max_skip:
                            max_skip, max_skip_split_dim, max_skip_split_value, max_skip_split_type = skip, split_dim, split_value, 2

                if max_skip > 0:
                    # if the cost become smaller, apply the cut
                    child_node1, child_node2 = self.partition_tree.apply_split(leaf.nid, max_skip_split_dim, max_skip_split_value, max_skip_split_type)
                    #print(" Split on node id:", leaf.nid)
                    CanSplit = True
        # return int(self.partition_tree.evaluate_query_cost(queries) * len(queries))
    def __JNORA_Plus(self,data_threshold,join_attr,join_depth):
        CanSplit = True
        cur_depth = 0
        while CanSplit:
            CanSplit = False
            cur_depth += 1
            if cur_depth > join_depth:
                break
            leaves = self.partition_tree.get_leaves()
            # print("# number of leaf nodes:",len(leaves))
            for leaf in leaves:
                # get best candidate cut position
                skip, max_skip, max_skip_split_dim, max_skip_split_value, max_skip_split_type = 0, -1, 0, 0, 0
                # extend the candidate cut with medians when it reach the bottom
                candidate_cuts = leaf.get_candidate_cuts_for_join(join_attr)
                for split_dim, split_value in candidate_cuts:

                    # first try normal split
                    valid, skip, left_size, right_size = leaf.if_split(split_dim, split_value, data_threshold)
                    if valid and skip > max_skip:
                        max_skip, max_skip_split_dim, max_skip_split_value, max_skip_split_type = skip, split_dim, split_value, 0

                    # if it's available for bounding split, try it
                    if leaf.node_size < 3 * data_threshold:
                        # try bounding split
                        valid, skip, _ = leaf.if_bounding_split(data_threshold, approximate=False)
                        if valid and skip > max_skip:
                            max_skip, max_skip_split_dim, max_skip_split_value, max_skip_split_type = skip, split_dim, split_value, 1

                    # if it's availble for dual-bounding split, try it
                    elif leaf.node_size < 4 * data_threshold and left_size < 2 * data_threshold and right_size < 2 * data_threshold:
                        # try dual-bounding split
                        valid, skip = leaf.if_dual_bounding_split(split_dim, split_value, data_threshold,
                                                                  approximate=False)
                        if valid and skip > max_skip:
                            max_skip, max_skip_split_dim, max_skip_split_value, max_skip_split_type = skip, split_dim, split_value, 2

                if max_skip > 0:
                    # if the cost become smaller, apply the cut
                    self.partition_tree.apply_split(leaf.nid, max_skip_split_dim,max_skip_split_value,max_skip_split_type)
                    # print(" Split on node id:", leaf.nid)
                    CanSplit = True
    def __JNORA_Beam_Search_Plus(self, data_threshold, queries, join_attr,candidate_size=1,candidate_depth=1, depth_limit=None, beam_search_mode=0,join_depth=3):
        if depth_limit is not None and depth_limit < candidate_depth:
            print("-> Enter beam search for leaf", self.partition_tree.pt_root.nid, "depth_limit:", depth_limit, "/",candidate_depth)

        def len_DQ(someset):
            if someset is not None:
                return len(someset)
            return 0

        CanSplit = True
        cur_depth=0
        while CanSplit:
            CanSplit = False
            cur_depth+=1
            if cur_depth>join_depth:
                break
            # DO NOT CONSIDER Partitionable!
            leaves = self.partition_tree.get_leaves()
            # leaves = self.partition_tree.get_leaves(use_partitionable = True)
            # print("# number of leaf nodes:",len(leaves))

            whole_partition_cost = 0

            for leaf in leaves:
                # if leaf.node_size < 2 * data_threshold or leaf.queryset is None:
                #     # TODO define cost: temporary cross-par query number
                #     # whole_partition_cost += sum([1 for query in leaf.queryset if len(self.partition_tree.query_single(query))>=2])
                #     # redefine cost: skip num
                #     whole_partition_cost+=len_DQ(leaf.queryset)*leaf.node_size
                #     print("X IGNORE leaf", leaf.nid, "leaf node dataset", len_DQ(leaf.dataset), "queryset size:",len_DQ(leaf.queryset))
                #     print("[ignore leaf] current whole partition cost:", whole_partition_cost, " + step:",0)
                #     continue
                print("O CONSIDER leaf node id:", leaf.nid, "leaf node dataset", len_DQ(leaf.dataset), "queryset size:",
                      len_DQ(leaf.queryset))

                # always extend the candidate cut with medians
                candidate_cuts = leaf.get_candidate_cuts_for_join(join_attr)
                split_candidates = []

                for split_dim, split_value in candidate_cuts:

                    # TODO keep the skip metric
                    valid, skip, left_size, right_size = leaf.if_split(split_dim, split_value, data_threshold)
                    # cross_query_num=leaf.num_query_crossed(split_dim, split_value)
                    if valid:
                        split_candidates.append((skip, split_dim, split_value, 0, valid))

                if len(split_candidates) == 0:
                    continue

                # 考虑cross_query_num,带来的效益真的大吗
                # split_candidates.sort(key=lambda item: (item[0],-item[5]), reverse=True)
                split_candidates.sort(key=lambda item: item[0], reverse=True)
                best_split = None
                min_cost = float('inf')

                if depth_limit is None:
                    depth_limit = candidate_depth
                if depth_limit == 0:
                    print(f"candidate cuts:{len(candidate_cuts)} split_candidates:{len(split_candidates)}")
                    # TODO: compute leaf cost: the number of query which cross mult-nodes in leaf
                    # leaf_cost = split_candidates[0][5]+sum([1 for query in leaf.queryset if len(self.partition_tree.query_single(query))>=2])
                    # recompute leaf cost
                    leaf_cost = len(leaf.queryset) * leaf.node_size - split_candidates[0][0]
                    whole_partition_cost += leaf_cost
                    print("[depth limit 0] current whole partition cost:", whole_partition_cost, "+ step:", leaf_cost)
                    if candidate_depth != 0:
                        continue
                    else:
                        best_split = split_candidates[0]
                        min_cost = leaf_cost
                    # return self.partition_tree.evaluate_query_cost(training_set, False)
                if candidate_depth != 0:
                    print("START beam search for node", leaf.nid)
                    if beam_search_mode == 0 or depth_limit == candidate_depth:
                        # num_beams = min(candidate_size, len(split_candidates))
                        for i in range(min(candidate_size, len(split_candidates))):
                            # I need to copy myself for the split
                            temp_node = copy.deepcopy(leaf)
                            temp_queries = copy.deepcopy(leaf.queryset)

                            # apply this split on temp node
                            temp_tree = PartitionTree(temp_node.num_dims, temp_node.boundary)
                            temp_tree.pt_root = temp_node
                            temp_tree.nid_node_dict = {temp_node.nid: temp_node}
                            temp_tree.node_count = temp_node.nid + 1
                            temp_tree.apply_split(temp_node.nid, split_candidates[i][1], split_candidates[i][2],
                                                  split_candidates[i][3])

                            temp_algo = PartitionAlgorithm()
                            temp_algo.partition_tree = temp_tree

                            child_cost = temp_algo._PartitionAlgorithm__JNORA_Beam_Search_Plus(data_threshold, temp_queries,join_attr,
                                                                                         candidate_size,
                                                                                        candidate_depth,
                                                                                        depth_limit - 1,
                                                                                        beam_search_mode,join_depth-1)
                            print("<- beam search result cost for leaf", leaf.nid, "cut", i, "type:",
                                  split_candidates[i][3], "cost:", child_cost)
                            if child_cost < min_cost:
                                best_split = split_candidates[i]
                                min_cost = child_cost

                    elif beam_search_mode == 1:
                        # num_beams = 1
                        temp_node = copy.deepcopy(leaf)
                        temp_queries = copy.deepcopy(leaf.queryset)

                        # apply this split on temp node
                        temp_tree = PartitionTree(temp_node.num_dims, temp_node.boundary)
                        temp_tree.pt_root = temp_node
                        temp_tree.nid_node_dict = {temp_node.nid: temp_node}
                        temp_tree.node_count = temp_node.nid + 1
                        temp_tree.apply_split(temp_node.nid, split_candidates[0][1], split_candidates[0][2],
                                              split_candidates[0][3])

                        temp_algo = PartitionAlgorithm()
                        temp_algo.partition_tree = temp_tree

                        child_cost = temp_algo._PartitionAlgorithm__JNORA_Beam_Search_Plus(data_threshold, temp_queries,join_attr,
                                                                                     candidate_size,
                                                                                    candidate_depth,
                                                                                    depth_limit - 1,
                                                                                    beam_search_mode,join_depth-1)
                        min_cost = child_cost

                    if depth_limit != candidate_depth:
                        whole_partition_cost += min_cost
                        print("[after beam search] current whole partition cost:", whole_partition_cost, "+step:",
                              min_cost)
                        continue
                        # return min_cost

                # if this is the current level (i.e., after the beam search returned), apply the search result
                print("= = = AFTER beam search for node", leaf.nid)
                if  min_cost < len(leaf.queryset) * leaf.node_size and best_split is not None and best_split[0] > 0:
                    # if the cost become smaller, apply the cut
                    self.partition_tree.apply_split(leaf.nid, best_split[1], best_split[2], best_split[3])
                    print("= = = SPLIT on node id:", leaf.nid, "split type:", best_split[3])
                    CanSplit = True
                else:
                    print("leaf node id:", leaf.nid, "does not have valid split")
                    leaf.no_valid_partition = True

            if depth_limit != candidate_depth and candidate_depth != 0:
                return whole_partition_cost

        # if it already met the bottom of the tree, (even it hasn't met the depth limit)
        # return sum([len(self.partition_tree.query_single(query))-1 for query in queries])

        return int(self.partition_tree.evaluate_query_cost(queries) * len(queries))

    def __JNORA_Beam_Search(self,data_threshold, candidate_size=1, candidate_depth=2):
        '''
        using beam search to improve the search space
        candidate_size: how many candidate splits we maintain in a layer
        candidate_depth: how many layers we keep during the search
        '''
        '''
        the NORA algorithm that optimized for 1 by 1 mapping scenario
        '''

        CanSplit = True
        while CanSplit:
            CanSplit = False

            # for leaf in self.partition_tree.get_leaves():
            leaves = self.partition_tree.get_leaves(use_partitionable=True)  # there could be large irregular shape partitions
            # print("# number of leaf nodes:",len(leaves))
            for leaf in leaves:

                # print("current leaf node id:",leaf.nid, "leaf node dataset size:",len(leaf.dataset))
                if leaf.node_size < 2 * data_threshold or leaf.queryset is None:
                    continue

                if len(leaf.queryset) == 1:
                    # try bounding split here
                    valid, skip, bound = leaf.if_bounding_split(data_threshold, approximate=False)

                    if valid:
                        # apply this split
                        # print(" = = = Apply Old Bounding Split = = = ")
                        self.partition_tree.apply_split(leaf.nid, None, None, 1, bound)
                        # print("!!!Split From Internal Bounding Split!!!")
                        continue
                        # design an variant of bounding split, which will handle node size problem # done
                        # if valid, mark the children as  don't consider anymore" -> partitionable = False
                if leaf.node_size<=3*data_threshold:
                    if leaf.if_new_bounding_split(data_threshold):
                        # print(" = = = Apply New Bounding Split = = = ")
                        # create new bounding split sub-partitions
                        self.partition_tree.apply_split(leaf.nid, None, None, 3)
                        continue

                # get best candidate cut position
                skip, max_skip, max_skip_split_dim, max_skip_split_value, max_skip_split_type = 0, -1, 0, 0, 0
                # extend the candidate cut with medians when it reach the bottom
                # candidate_cuts = leaf.get_candidate_cuts(True) if leaf.node_size < 4 * data_threshold else leaf.get_candidate_cuts()
                candidate_cuts = leaf.get_candidate_cuts(extended=True, begin_pos=2)  # try extends it always

                split_candidates = []
                for split_dim, split_value in candidate_cuts:

                    # first try normal split
                    valid, skip, left_size, right_size = leaf.if_split(split_dim, split_value, data_threshold)
                    # if valid and skip > max_skip:
                    #    max_skip, max_skip_split_dim, max_skip_split_value, max_skip_split_type = skip, split_dim, split_value, 0
                    if valid:
                        num_query_crossed = leaf.num_query_crossed(split_dim, split_value)
                        split_candidates.append((skip, split_dim, split_value, 0, valid,num_query_crossed))

                    # Should we remove the leaf node case here in 1 BY 1?
                    # the following cases here are applied only for leaf nodes
                    # if it's available for bounding split, try it
                    if leaf.node_size < 3 * data_threshold:
                        # try bounding split
                        valid, skip, _ = leaf.if_bounding_split(data_threshold, approximate=False)
                        if valid and skip > max_skip:
                            split_candidates.append((skip, split_dim, split_value, 1, valid, 0))
                            # max_skip, max_skip_split_dim, max_skip_split_value, max_skip_split_type = skip, split_dim, split_value, 1

                    # if it's availble for dual-bounding split, try it
                    elif leaf.node_size < 4 * data_threshold and left_size < 2 * data_threshold and right_size < 2 * data_threshold:
                        # try dual-bounding split
                        valid, skip = leaf.if_dual_bounding_split(split_dim, split_value, data_threshold,
                                                                  approximate=False)
                        if valid and skip > max_skip:
                            split_candidates.append((skip, split_dim, split_value, 1, valid, 2))
                            # max_skip, max_skip_split_dim, max_skip_split_value, max_skip_split_type = skip, split_dim, split_value, 2

                if len(split_candidates) == 0:
                    continue

                split_candidates.sort(key=lambda item: item[0], reverse=True)  # from the most skip gain to least
                found_available_split = False
                split_context = None
                split_context = split_candidates[0]

                # Beam search on the first few candidates
                min_cost = leaf.node_size * len(leaf.queryset)
                for i in range(min(candidate_size, len(split_candidates))):
                    # explored_cost = pass
                    # first create a partition tree using the current partition
                    # then split it with depth constraint!
                    # third, analyze the cost, maybe I just used the minimum total (avg) cost for comparison

                    fake_split = split_candidates[i]
                    left_child, right_child = self.partition_tree.apply_split(leaf.nid, fake_split[1], fake_split[2],
                                                                              fake_split[3],
                                                                              pretend=True)  # do not actually apply this split

                    temp_left_partition = PartitionAlgorithm()
                    temp_right_partition = PartitionAlgorithm()
                    if left_child.queryset:
                        # WAIT! , it should be the left cost + the right cost instead of directly using the current leaf!
                        left_cost = temp_left_partition.InitializeWithNORA(left_child.queryset, leaf.num_dims,
                                                                           left_child.boundary,
                                                                           left_child.dataset, data_threshold,
                                                                           using_1_by_1=False,
                                                                           using_kd=False, depth_limit=candidate_depth,
                                                                           return_query_cost=True)
                    else:
                        left_cost = 0
                    if right_child.queryset:
                        right_cost = temp_right_partition.InitializeWithNORA(right_child.queryset, leaf.num_dims,
                                                                             right_child.boundary,
                                                                             right_child.dataset, data_threshold,
                                                                             using_1_by_1=False,
                                                                             using_kd=False,
                                                                             depth_limit=candidate_depth,
                                                                             return_query_cost=True)
                    else:
                        right_cost = 0
                    explored_cost = left_cost + right_cost
                    if explored_cost < min_cost:
                        min_cost = explored_cost
                        split_context = split_candidates[i]

                        # apply split context
                if split_context[0] > 0:
                    self.partition_tree.apply_split(leaf.nid, split_context[1], split_context[2], split_context[3])
                    # print(" Split on node id:", leaf.nid)
                    # print("split_context:",split_context)
                    CanSplit = True


    def __QDT(self, data_threshold):
        '''
        the QdTree partition algorithm
        '''
        CanSplit = True
        print_s=True
        while CanSplit:
            CanSplit = False           
            
            # for leaf in self.partition_tree.get_leaves():
            leaves = self.partition_tree.get_leaves()
            #print("# number of leaf nodes:",len(leaves))
            for leaf in leaves:
                     
                # print("current leaf node id:",leaf.nid, "leaf node dataset size:",len(leaf.dataset))
                if leaf.node_size < 2 * data_threshold:
                    continue
                
                candidate_cuts = leaf.get_candidate_cuts()

                # get best candidate cut position
                skip, max_skip, max_skip_split_dim, max_skip_split_value = 0, -1, 0, 0
                for split_dim, split_value in candidate_cuts:

                    valid,skip,_,_ = leaf.if_split(split_dim, split_value, data_threshold)
                    if valid and skip > max_skip:
                        max_skip = skip
                        max_skip_split_dim = split_dim
                        max_skip_split_value = split_value

                if max_skip > 0:
                    # if the cost become smaller, apply the cut
                    if print_s:
                        print("QDTREE CUT!")
                        print_s=False
                    child_node1, child_node2 = self.partition_tree.apply_split(leaf.nid, max_skip_split_dim, max_skip_split_value)
                    # print(" Split on node id:", leaf.nid)
                    CanSplit = True

    def __JQDT(self, data_threshold):
        '''
        the QdTree partition algorithm
        '''
        CanSplit = True
        print_s = True
        while CanSplit:
            CanSplit = False

            # for leaf in self.partition_tree.get_leaves():
            leaves = self.partition_tree.get_leaves()
            # print("# number of leaf nodes:",len(leaves))
            for leaf in leaves:

                # print("current leaf node id:",leaf.nid, "leaf node dataset size:",len(leaf.dataset))
                if leaf.node_size < 2 * data_threshold:
                    continue

                candidate_cuts = leaf.get_candidate_cuts(extended=False, begin_pos=2)

                # get best candidate cut position
                skip, max_skip, max_skip_split_dim, max_skip_split_value = 0, -1, 0, 0
                for split_dim, split_value in candidate_cuts:

                    valid, skip, _, _ = leaf.if_split(split_dim, split_value, data_threshold)
                    if valid and skip > max_skip:
                        max_skip = skip
                        max_skip_split_dim = split_dim
                        max_skip_split_value = split_value

                if max_skip > 0:
                    # if the cost become smaller, apply the cut
                    if print_s:
                        print("QDTREE CUT!")
                        print_s = False
                    child_node1, child_node2 = self.partition_tree.apply_split(leaf.nid, max_skip_split_dim,
                                                                               max_skip_split_value)
                    # print(" Split on node id:", leaf.nid)
                    CanSplit = True

    # Ameoba
    def __AMT(self,num_dims,data_threshold):
        canSplit=True
        depth=-1
        while canSplit:
            canSplit=False
            leaves=self.partition_tree.get_leaves()
            depth+=1
            for leaf in leaves:
                if leaf.node_size < 2 * data_threshold or leaf.depth<depth:
                    continue
                # 选择剩余可分配数据最大的属性作为下一个切割点
                temp_allocations=self.partition_tree.allocations.copy()
                vaild_dim=[i for i in range(num_dims)]
                for _ in range(num_dims):
                    test_dim=vaild_dim[temp_allocations.index(max(temp_allocations))]
                    median = np.median(leaf.dataset[:,test_dim])
                    sub_dataset1_size = np.count_nonzero(leaf.dataset[:,test_dim] < median)
                    sub_dataset2_size = len(leaf.dataset) - sub_dataset1_size
                    if sub_dataset1_size < data_threshold or sub_dataset2_size < data_threshold:
                        idx=vaild_dim.index(test_dim)
                        del temp_allocations[idx]
                        del vaild_dim[idx]
                        continue
                    else:
                        child_node1, child_node2 = self.partition_tree.apply_split(leaf.nid, test_dim, median)
                        child_node1.depth=depth+1
                        child_node2.depth=depth+1
                        self.partition_tree.allocations[test_dim] -= 2.0 / pow(2,depth)
                        canSplit=True
                        break

    def __AMT_JOIN(self,data_threshold,join_attr,join_depth):
        canSplit = True
        cur_depth = -1
        while canSplit:
            canSplit = False
            leaves = self.partition_tree.get_leaves()
            cur_depth += 1
            if cur_depth>=join_depth: break
            for leaf in leaves:
                if leaf.node_size < 2 * data_threshold or leaf.depth < cur_depth:
                    continue
                # 选择剩余可分配数据最大的属性作为下一个切割点
                temp_allocations = self.partition_tree.allocations.copy()
                split_dim=join_attr
                split_value = np.median(leaf.dataset[:, split_dim])
                valid, skip, _, _ = leaf.if_split(split_dim, split_value, data_threshold)
                if valid:
                    child_node1, child_node2 = self.partition_tree.apply_split(leaf.nid, split_dim, split_value)
                    child_node1.depth = leaf.depth + 1
                    child_node2.depth = leaf.depth + 1
                    # self.partition_tree.allocations[split_dim] -= 2.0 / pow(2, cur_depth)
                    canSplit = True

    def __KDT(self, current_dim, data_threshold, current_node):
        '''
        Store the dataset in PartitionNode: we can keep it, but only as a tempoary attribute
        '''
        # cannot be further split
        if current_node.node_size < 2 * data_threshold:
            return   
        
        # split the node into equal halves by its current split dimension
        median = np.median(current_node.dataset[:,current_dim])
        
        sub_dataset1_size = np.count_nonzero(current_node.dataset[:,current_dim] < median)
        sub_dataset2_size = len(current_node.dataset) - sub_dataset1_size
        
        if sub_dataset1_size < data_threshold or sub_dataset2_size < data_threshold:
            pass
        else:
            child_node1, child_node2 = self.partition_tree.apply_split(current_node.nid, current_dim, median)
            
            # update next split dimension
            current_dim += 1
            if current_dim >= current_node.num_dims:
                current_dim %= current_node.num_dims
    
            # recursive call on sub nodes
            self.__KDT(current_dim, data_threshold, child_node1)
            self.__KDT(current_dim, data_threshold, child_node2)