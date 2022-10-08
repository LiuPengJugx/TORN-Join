import time
from sympy import im
import pickle
from partition_node import PartitionNode
import numpy as np
from numpy import genfromtxt
import copy
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
class PartitionTree:
        '''
        The data structure that represent the partition layout, which also maintain the parent, children relation info
        Designed to provide efficient online query and serialized ability
        
        The node data structure could be checked from the PartitionNode class
        
        '''   
        def __init__(self, num_dims = 0, boundary = []):
            
            # the node id of root should be 0, its pid should be -1
            # note this initialization does not need dataset and does not set node size!

            self.pt_root = PartitionNode(num_dims, boundary, nid = 0, pid = -1, is_irregular_shape_parent = False, 
                                         is_irregular_shape = False, num_children = 0, children_ids = [], is_leaf = True, node_size = 0)
            self.nid_node_dict = {0: self.pt_root} # node id to node dictionary
            self.node_count = 1 # the root node
            self.allocations=list()
            self.table_dataset_size=None # record the table data size
            self.redundant_data_ratio=None # record the ratio of redundant data
        
        # = = = = = public functions (API) = = = = =
        
        def save_tree(self, path):
            node_list = self.__generate_node_list(self.pt_root) # do we really need this step?
            # serialized_node_list = self.__serialize(node_list)
            serialized_node_list = self.__serialize_by_pickle(node_list)
            #print(serialized_node_list)
            # np.savetxt(path, serialized_node_list, delimiter=',')
            with open(path, "wb") as f:
                # perform different operations for join tree and normal tree
                if hasattr(self, 'join_attr'):
                    pickle.dump((self.join_attr,self.join_depth,serialized_node_list), f, True)
                else:
                    pickle.dump(serialized_node_list, f, True)
            return serialized_node_list
            
        def load_tree(self, path):
            # serialized_node_list = genfromtxt(path, delimiter=',')
            with open(path, "rb") as f:
                if hasattr(self, 'join_attr'):
                    res = pickle.load(f)
                    self.join_attr=res[0]
                    self.join_depth=res[1]
                    serialized_node_list=res[2]
                else:
                    serialized_node_list=pickle.load(f)
            self.__build_tree_from_serialized_node_list(serialized_node_list)
        def query_single_join(self, query):
            partition_ids = self.__find_overlapped_partition_consider_depth(self.pt_root, query, 1)
            return partition_ids

        def query_single(self, query, using_rtree_filter = False, print_info = False, redundant_partitions = None):
            '''
            query is in plain form, i.e., [l1,l2,...,ln, u1,u2,...,un]
            return the overlapped leaf partitions ids!
            redundant_partition: [(boundary, size)...]
            '''
            # used only when redundant_partition is given
            def check_inside(query, partition_boundary):
                num_dims = len(query)//2
                for i in range(num_dims):
                    if query[i] >= partition_boundary[i] and query[num_dims + i] <= partition_boundary[num_dims + i]:
                        pass
                    else:
                        return False
                return True
            
            if redundant_partitions is None:

                partition_ids = self.__find_overlapped_partition(self.pt_root, query, using_rtree_filter, print_info)
                return partition_ids
            else:
                # first check if a query is inside any redundant partition, find the smallest one
                costs = []
                for rp in redundant_partitions:
                    if check_inside(query, rp[0]):
                        costs.append(rp[1])
                if len(costs) == 0:
                    # if not, use regular approach
                    partition_ids = self.__find_overlapped_partition(self.pt_root, query, using_rtree_filter, print_info)
                    return partition_ids
                else:
                    # return the smallest rp size
                    return [-min(costs)] # the minus sign used to differentiate from partition ides
        
        def query_batch(self, queries):
            '''
            to be implemented
            '''
            pass
        
        def get_queryset_cost(self, queries):
            '''
            return the cost array directly
            '''
            costs = []
            for query in queries:
                overlapped_leaf_ids = self.query_single(query)
                cost = 0
                for nid in overlapped_leaf_ids:
                    cost += self.nid_node_dict[nid].node_size
                costs.append(cost)
            return costs

        def set_redundant_partition(self,queries,data_threshold):
            overlap_ids_for_query=list()
            for qid,query in enumerate(queries):
                overlapped_leaf_ids = self.query_single(query)
                used_node_ratio_dict={}
                for nid in overlapped_leaf_ids:
                    used_size=self.nid_node_dict[nid].query_result_size(query, approximate=False)
                    node_size=self.nid_node_dict[nid].node_size
                    used_node_ratio_dict[nid]=(used_size,used_size/node_size)
                overlap_ids_for_query.append(used_node_ratio_dict)
            gain_queries={}
            for qid,query_dict in enumerate(overlap_ids_for_query):
                node_ids=query_dict.keys()
                redundant_nids=[nid for nid in node_ids if query_dict[nid][1]<=0.35]
                redundant_sizes = [self.nid_node_dict[idx].node_size for idx in redundant_nids]
                gain=0
                if len(redundant_nids)>=1:
                    flag=False
                    # only part nodes whose ratio <0.5, and follow steps are used to compute gain
                    if len(node_ids)>len(redundant_nids):
                        flag=True
                        gain=sum(redundant_sizes)
                        linked_nids = list(set(node_ids) - set(redundant_nids))
                        # print("case 1 !!!!!!!")
                    #all nodes
                    elif len(redundant_nids)>=2:
                        flag=True
                        min_redun_nid=redundant_nids[redundant_sizes.index(min(redundant_sizes))]
                        linked_nids=[min_redun_nid]
                        redundant_nids.remove(min_redun_nid)
                        gain=sum(redundant_sizes)-min(redundant_sizes)
                        # print("case 2 !!!!!!!")
                    #   distribute linked nids for redundant nids
                    if flag:
                        group_rels = {}
                        cnt = 0
                        for redundant_nid in redundant_nids:
                            linked_nid = linked_nids[cnt]
                            if linked_nid not in group_rels.keys():
                                group_rels[linked_nid] = []
                            group_rels[linked_nid].append(redundant_nid)
                            cnt += 1
                            if cnt>=len(linked_nids): cnt=0
                        gain_queries[qid]=({'gain':gain,'group':group_rels})
            # print(gain_queries)
            # record storage space for redundant partitions
            redundant_data_size=0
            for qid in gain_queries.keys():
                gain_query=gain_queries[qid]
                if gain_query['gain']>0:
                    # print("Query #",qid," is set redundant!",self.query_single(queries[qid]),queries[qid])
                    for linked_nid in gain_query['group'].keys():
                        l_node=self.nid_node_dict[linked_nid]
                        # print(f"link:{linked_nid}, redundant:{gain_query['group'][linked_nid]}")
                        for redundant_nid in gain_query['group'][linked_nid]:
                            r_node=self.nid_node_dict[redundant_nid]
                            if linked_nid not in r_node.linked_ids:
                                r_node.linked_ids.append(linked_nid)
                            actual_boundary=r_node.max_bound_for_query(queries[qid])
                            l_node.redundant_boundaries.append(actual_boundary)
                            actual_dataset=r_node.get_query_result(actual_boundary)
                            l_node.redundant_datasets.append(actual_dataset)
                            redundant_data_size+=actual_dataset.shape[0]
                            l_node.node_size+=overlap_ids_for_query[qid][redundant_nid][0]
            self.redundant_data_ratio=redundant_data_size/self.table_dataset_size

        def evaluate_query_cost(self, queries, print_result = False, using_rtree_filter = False, redundant_partitions = None):
            # if len(queries)==0: return 0
            '''
            get the logical IOs of the queris
            return the average query cost
            '''
            total_cost = 0
            case = 0
            total_overlap_ids = {}
            case_cost = {}
            reduced_cost = 0
            for count,query in enumerate(queries):
                cost = 0
                overlapped_leaf_ids = self.query_single(query)
                total_overlap_ids[case] = overlapped_leaf_ids
                actual_data_size=[]
                for nid in overlapped_leaf_ids:
                    if nid >= 0:
                        cur_node=self.nid_node_dict[nid]
                        cost += cur_node.node_size
                        actual_data_size.append(cur_node.node_size)
                        # consider redundant partitions
                        # is_exist=False
                        # if cur_node.linked_ids:
                        #     query_boundary=cur_node.max_bound_for_query(query)
                        #     for link_id in cur_node.linked_ids:
                        #         link_node=self.nid_node_dict[link_id]
                        #         bound_id=link_node.if_boundary_redundant(query_boundary)
                        #         print(bound_id)
                        #         if bound_id>=0:
                        #             new_size=len(link_node.redundant_datasets[bound_id])
                        #             # cost+=new_size
                        #             reduced_cost+=cur_node.node_size-new_size
                        #             is_exist=True
                        #             break
                        # if not is_exist:
                        #     cost += cur_node.node_size
                    else:
                        cost += (-nid) # redundant partition cost
                # print(f"query #{count}: {actual_data_size}")
                # print(f"query #{count}: cost:{sum(actual_data_size)} num:{len(actual_data_size)}")
                total_cost += cost
                case_cost[case] = cost
                case += 1
            
            if print_result:
                print("Total logical IOs:", total_cost)
                print("Average logical IOs:", total_cost // len(queries))
                # for case, ids in total_overlap_ids.items():
                    # print("query",case, ids, "cost:", case_cost[case])
            
            return total_cost // len(queries)
        
        def get_pid_for_data_point(self, point):
            '''
            get the corresponding leaf partition nid for a data point
            point: [dim1_value, dim2_value...], contains the same dimenions as the partition tree
            '''
            return self.__find_resided_partition(self.pt_root, point)
        
        def add_node(self, parent_id, child_node):
            child_node.nid = self.node_count
            self.node_count += 1
            
            child_node.pid = parent_id
            self.nid_node_dict[child_node.nid] = child_node
            
            child_node.depth = self.nid_node_dict[parent_id].depth + 1
            
            self.nid_node_dict[parent_id].children_ids.append(child_node.nid)
            self.nid_node_dict[parent_id].num_children += 1
            self.nid_node_dict[parent_id].is_leaf = False
        
        
        def apply_split(self, parent_nid, split_dim, split_value, split_type = 0, extended_bound = None, approximate = False,
                        pretend = False):
            '''
            split_type = 0: split a node into 2 sub-nodes by a given dimension and value, distribute dataset
            split_type = 1: split a node by bounding split (will create an irregular shape partition)
            split_type = 2: split a node by daul-bounding split (will create an irregular shape partition)
            split_type = 3: split a node by var-bounding split (multi MBRs), distribute dataset
            extended_bound is only used in split type 1
            approximate: used for measure query result size
            pretend: if pretend is True, return the split result, but do not apply this split
            '''
            parent_node = self.nid_node_dict[parent_nid]
            if pretend:
                parent_node = copy.deepcopy(self.nid_node_dict[parent_nid])
            
            child_node1, child_node2 = None, None
            
            if split_type == 0:
                
                #print("[Apply Split] Before split node", parent_nid, "node queryset:", parent_node.queryset, "MBRs:", parent_node.query_MBRs)
            
                # create sub nodes
                child_node1 = copy.deepcopy(parent_node)
                child_node1.boundary[split_dim + child_node1.num_dims] = split_value
                child_node1.children_ids = []

                child_node2 = copy.deepcopy(parent_node)
                child_node2.boundary[split_dim] = split_value
                child_node2.children_ids = []
                
                if parent_node.query_MBRs is not None:
                    MBRs1, MBRs2 = parent_node.split_query_MBRs(split_dim, split_value)
                    child_node1.query_MBRs = MBRs1
                    child_node2.query_MBRs = MBRs2
                    
                # if parent_node.dataset != None: # The truth value of an array with more than one element is ambiguous.
                # https://stackoverflow.com/questions/36783921/valueerror-when-checking-if-variable-is-none-or-numpy-array
                if parent_node.dataset is not None:
                    child_node1.dataset = parent_node.dataset[parent_node.dataset[:,split_dim] < split_value]
                    child_node1.node_size = len(child_node1.dataset)
                    child_node2.dataset = parent_node.dataset[parent_node.dataset[:,split_dim] >= split_value]
                    child_node2.node_size = len(child_node2.dataset)

                if parent_node.queryset is not None:
                    left_part, right_part, mid_part = parent_node.split_queryset(split_dim, split_value)
                    child_node1.queryset = left_part + mid_part
                    child_node2.queryset = right_part + mid_part
                    
                #print("[Apply Split] After split node", parent_nid, "left child queryset:", child_node1.queryset, "MBRs:", child_node1.query_MBRs)
                #print("[Apply Split] After split node", parent_nid, "right child queryset:", child_node2.queryset, "MBRs:", child_node2.query_MBRs)

                # update current node
                if not pretend:
                    self.add_node(parent_nid, child_node1)
                    self.add_node(parent_nid, child_node2)
                    self.nid_node_dict[parent_nid].split_type = "candidate cut"
            
            elif split_type == 1: # must reach leaf node, hence no need to maintain dataset and queryset any more
                
                child_node1 = copy.deepcopy(parent_node) # the bounding partition
                child_node2 = copy.deepcopy(parent_node) # the remaining partition, i.e., irregular shape
                
                child_node1.is_leaf = True
                child_node2.is_leaf = True
                
                child_node1.children_ids = []
                child_node2.children_ids = []
                
                max_bound = None
                if extended_bound is not None:
                    max_bound = extended_bound
                else:
                    max_bound = parent_node._PartitionNode__max_bound(parent_node.queryset)
                child_node1.boundary = max_bound
                child_node2.is_irregular_shape = True
                
                bound_size = parent_node.query_result_size(max_bound, approximate = False)
                remaining_size = parent_node.node_size - bound_size           
                child_node1.node_size = bound_size
                child_node2.node_size = remaining_size
                
                child_node1.partitionable = False
                child_node2.partitionable = False
                
                if not pretend:
                    self.add_node(parent_nid, child_node1)
                    self.add_node(parent_nid, child_node2)
                    self.nid_node_dict[parent_nid].is_irregular_shape_parent = True
                    self.nid_node_dict[parent_nid].split_type = "sole-bounding split"
            
            elif split_type == 2: # must reach leaf node, hence no need to maintain dataset and queryset any more
                
                child_node1 = copy.deepcopy(parent_node) # the bounding partition 1
                child_node2 = copy.deepcopy(parent_node) # the bounding partition 2
                child_node3 = copy.deepcopy(parent_node) # the remaining partition, i.e., irregular shape
                
                child_node1.is_leaf = True
                child_node2.is_leaf = True
                child_node3.is_leaf = True
                
                child_node1.children_ids = []
                child_node2.children_ids = []
                child_node3.children_ids = []
                
                left_part, right_part, mid_part = parent_node.split_queryset(split_dim, split_value)
                max_bound_1 = parent_node._PartitionNode__max_bound(left_part)
                max_bound_2 = parent_node._PartitionNode__max_bound(right_part)
                
                child_node1.boundary = max_bound_1
                child_node2.boundary = max_bound_2
                child_node3.is_irregular_shape = True          
                
                # Should we only consider the case when left and right cannot be further split? i.e., [b,2b)
                # this check logic is given in the PartitionAlgorithm, not here, as the split action should be general
                naive_left_size = np.count_nonzero(parent_node.dataset[:,split_dim] < split_value)
                naive_right_size = parent_node.node_size - naive_left_size

                # get (irregular-shape) sub-partition size
                bound_size_1 = parent_node.query_result_size(max_bound_1, approximate)
                if bound_size_1 is None: # there is no query within the left 
                    bound_size_1 = naive_left_size # use the whole left part as its size
               
                bound_size_2 = parent_node.query_result_size(max_bound_2, approximate)
                if bound_size_2 is None: # there is no query within the right
                    bound_size_2 = naive_right_size # use the whole right part as its size
               
                remaining_size = parent_node.node_size - bound_size_1 - bound_size_2
                
                child_node1.node_size = bound_size_1
                child_node2.node_size = bound_size_2
                child_node3.node_size = remaining_size
                
                child_node1.partitionable = False
                child_node2.partitionable = False
                child_node3.partitionable = False
                
                if not pretend:
                    self.add_node(parent_nid, child_node1)
                    self.add_node(parent_nid, child_node2)
                    self.add_node(parent_nid, child_node3)
                    self.nid_node_dict[parent_nid].is_irregular_shape_parent = True
                    self.nid_node_dict[parent_nid].split_type = "dual-bounding split"
            
            elif split_type == 3: # new bounding split, create a collection of MBR partitions
                
                remaining_size = parent_node.node_size
                for MBR in parent_node.query_MBRs:
                    child_node = copy.deepcopy(parent_node)
                    child_node.is_leaf = True
                    child_node.children_ids = []
                    child_node.boundary = MBR.boundary
                    child_node.node_size = MBR.bound_size
                    child_node.partitionable = False
                    remaining_size -= child_node.node_size
                    child_node.dataset = self.__extract_sub_dataset(parent_node.dataset, child_node.boundary)
                    child_node.queryset = MBR.queries # no other queries could overlap this MBR, or it's invalid
                    child_node.query_MBRs = [MBR]
                        
                    if not pretend:
                        self.add_node(parent_nid, child_node)
                
                # the last irregular shape partition, we do not need to consider its dataset
                # when the new query arrive , it's hard to preserve the irregular shape partition
                # solve method : node==> (regular partition)+(irregular partition)
                child_node = copy.deepcopy(parent_node)
                child_node.is_leaf = True
                child_node.children_ids = []
                child_node.is_irregular_shape = True
                child_node.node_size = remaining_size
                child_node.partitionable = False
                child_node.dataset = None
                child_node.queryset = None
                child_node.query_MBRs = None
                
                if not pretend:
                    self.add_node(parent_nid, child_node)
                    self.nid_node_dict[parent_nid].is_irregular_shape_parent = True
                    self.nid_node_dict[parent_nid].split_type = "var-bounding split"
            
            else:
                print("Invalid Split Type!")
            
            if not pretend:
                del self.nid_node_dict[parent_nid].dataset
                del self.nid_node_dict[parent_nid].queryset
                #del self.nid_node_dict[parent_nid].query_MBRs
                #self.nid_node_dict[parent_nid] = parent_node
                
            return child_node1, child_node2
        
        def get_leaves(self, use_partitionable = False):
            nodes = []
            if use_partitionable:
                for nid, node in self.nid_node_dict.items():
                    if node.is_leaf and node.partitionable:
                        nodes.append(node)
            else:
                for nid, node in self.nid_node_dict.items():
                    if node.is_leaf:
                        nodes.append(node)
            return nodes
        
        def visualize(self, dims = [0, 1], queries = [], path = None, focus_region = None, add_text = True, use_sci = False):
            '''
            visualize the partition tree's leaf nodes
            focus_region: in the shape of boundary
            '''
            if len(dims) == 2:
                self.__visualize_2d(dims, queries, path, focus_region, add_text, use_sci)
            else:
                self.__visualize_3d(dims[0:3], queries, path, focus_region)
            
        
        # = = = = = internal functions = = = = =
        
        def __extract_sub_dataset(self, dataset, query):
            constraints = []
            num_dims = self.pt_root.num_dims
            for d in range(num_dims):
                constraint_L = dataset[:,d] >= query[d]
                constraint_U = dataset[:,d] <= query[num_dims + d]
                constraints.append(constraint_L)
                constraints.append(constraint_U)
            constraint = np.all(constraints, axis=0)
            sub_dataset = dataset[constraint]
            return sub_dataset
        
        def __generate_node_list(self, node):
            '''
            recursively add childrens into the list
            '''
            node_list = [node]
            for nid in node.children_ids:
                node_list += self.__generate_node_list(self.nid_node_dict[nid])
            return node_list

        def __serialize_by_pickle(self,node_list):
            serialized_node_list = []
            for node in node_list:
                # follow the same order of attributes in partition class
                attributes = [node.num_dims,node.boundary,node.nid,node.pid]
                attributes.append(1 if node.is_irregular_shape_parent else 0)
                attributes.append(1 if node.is_irregular_shape else 0)
                attributes.append(node.num_children) # number of children
                #attributes += node.children_ids
                attributes.append(1 if node.is_leaf else 0)
                attributes.append(node.node_size)
                attributes.append(node.depth)
                attributes.append(node.redundant_boundaries)
                attributes.append(node.linked_ids)
                serialized_node_list.append(attributes)
            return serialized_node_list

        def __serialize(self, node_list):
            '''
            convert object to attributes to save
            '''
            serialized_node_list = []
            for node in node_list:
                # follow the same order of attributes in partition class
                attributes = [node.num_dims]
                #attributes += node.boundary
                if isinstance(node.boundary, list):
                    attributes += node.boundary
                else:
                    attributes += node.boundary.tolist()
                attributes.append(node.nid) # node id = its ow id
                attributes.append(node.pid) # parent id
                attributes.append(1 if node.is_irregular_shape_parent else 0)
                attributes.append(1 if node.is_irregular_shape else 0)
                attributes.append(node.num_children) # number of children
                #attributes += node.children_ids
                attributes.append(1 if node.is_leaf else 0)
                attributes.append(node.node_size)
                attributes.append(node.depth)
                attributes.append(str(node.redundant_boundaries))
                attributes.append(str(node.linked_ids))
                serialized_node_list.append(attributes)
            return serialized_node_list
        
        def __build_tree_from_serialized_node_list(self, serialized_node_list):
            
            self.pt_root = None
            self.nid_node_dict.clear()
            pid_children_ids_dict = {}
            for serialized_node in serialized_node_list:
                num_dims = serialized_node[0]
                boundary = serialized_node[1]
                nid = serialized_node[2]  # node id
                pid = serialized_node[3]  # parent id
                is_irregular_shape_parent = False if serialized_node[4] == 0 else True
                is_irregular_shape = False if serialized_node[5] == 0 else True
                num_children = serialized_node[6]
                is_leaf = False if serialized_node[7] == 0 else True
                node_size = serialized_node[8]
                node = PartitionNode(num_dims, boundary, nid, pid, is_irregular_shape_parent,
                                     is_irregular_shape, num_children, [], is_leaf,
                                     node_size)  # let the children_ids empty
                node.depth=serialized_node[9]
                node.redundant_boundaries=serialized_node[10]
                node.linked_ids=serialized_node[11]
                self.nid_node_dict[nid] = node  # update dict

                if node.pid in pid_children_ids_dict:
                    pid_children_ids_dict[node.pid].append(node.nid)
                else:
                    pid_children_ids_dict[node.pid] = [node.nid]
#             for serialized_node in serialized_node_list:
#                 num_dims = int(serialized_node[0])
#                 boundary = serialized_node[1: 1+2*num_dims]
#                 nid = int(serialized_node[1+2*num_dims]) # node id
#                 pid = int(serialized_node[2+2*num_dims]) # parent id
#                 is_irregular_shape_parent = False if serialized_node[3+2*num_dims] == 0 else True
#                 is_irregular_shape = False if serialized_node[4+2*num_dims] == 0 else True
#                 num_children = int(serialized_node[5+2*num_dims])
# #                 children_ids = []
# #                 if num_children != 0:
# #                     children_ids = serialized_node[1+5+2*num_dims: 1+num_children+1+5+2*num_dims] # +1 for the end exclusive
# #                 is_leaf = False if serialized_node[1+num_children+5+2*num_dims] == 0 else True
# #                 node_size = serialized_node[2+num_children+5+2*num_dims] # don't use -1 in case of match error
#                 is_leaf = False if serialized_node[6+2*num_dims] == 0 else True
#                 node_size = int(serialized_node[7+2*num_dims])
#                 depth=int(serialized_node[8+2*num_dims])
#                 redundant_boundaries=int(serialized_node[9+2*num_dims])
#                 linked_ids=int(serialized_node[10+2*num_dims])
#                 node = PartitionNode(num_dims, boundary, nid, pid, is_irregular_shape_parent,
#                                      is_irregular_shape, num_children, [], is_leaf, node_size) # let the children_ids empty
#                 self.nid_node_dict[nid] = node # update dict
#
#                 if node.pid in pid_children_ids_dict:
#                     pid_children_ids_dict[node.pid].append(node.nid)
#                 else:
#                     pid_children_ids_dict[node.pid] = [node.nid]

            # make sure the irregular shape partition is placed at the end of the child list
            for pid, children_ids in pid_children_ids_dict.items():
                if pid == -1:
                    continue
                if self.nid_node_dict[pid].is_irregular_shape_parent and not self.nid_node_dict[children_ids[-1]].is_irregular_shape:
                    # search for the irregular shape partition
                    new_children_ids = []
                    irregular_shape_id = None
                    for nid in children_ids:
                        if self.nid_node_dict[nid].is_irregular_shape:
                            irregular_shape_id = nid
                        else:
                            new_children_ids.append(nid)
                    new_children_ids.append(irregular_shape_id)
                    self.nid_node_dict[pid].children_ids = new_children_ids
                else:
                    self.nid_node_dict[pid].children_ids = children_ids
            
            self.pt_root = self.nid_node_dict[0]
        
        def __bound_query_by_boundary(self, query, boundary):
            '''
            bound the query by a node's boundary
            '''
            bounded_query = query.copy()
            num_dims = self.pt_root.num_dims
            for dim in range(num_dims):
                bounded_query[dim] = max(query[dim], boundary[dim])
                bounded_query[num_dims+dim] = min(query[num_dims+dim], boundary[num_dims+dim])
            return bounded_query
        
        def __find_resided_partition(self, node, point):
            '''
            for data point only
            '''
            #print("enter function!")
            if node.is_leaf:
                #print("within leaf",node.nid)
                if node.is_contain(point):
                    if node.linked_ids:
                        write_ids=[node.nid]
                        for link_id in node.linked_ids:
                            link_node = self.nid_node_dict[link_id]
                            if link_node.is_redundant_contain(point):
                                write_ids.append(link_id)
                        return write_ids
                    return [node.nid]
            
            for nid in node.children_ids:
                if self.nid_node_dict[nid].is_contain(point):
                    #print("within child", nid, "of parent",node.nid)
                    return self.__find_resided_partition(self.nid_node_dict[nid], point)
            
            #print("no children of node",node.nid,"contains point")
            return [-1]

        def __find_overlapped_partition_consider_depth(self, node, query,depth):
            if node.is_leaf or depth==self.join_depth:
                return [node.nid] if node.is_overlap(query) > 0 else []
            else:
                node_id_list = []
                for nid in node.children_ids:
                    node_id_list += self.__find_overlapped_partition_consider_depth(self.nid_node_dict[nid], query,depth+1)
                return node_id_list

        def __find_overlapped_partition(self, node, query,using_rtree_filter = False, print_info = False):
            
            if print_info:
                print("Enter node", node.nid)
                
            if node.is_leaf:
                if print_info:
                    print("node", node.nid, "is leaf")
                
                if using_rtree_filter and node.rtree_filters is not None:
                    for mbr in node.rtree_filters:
                        if node._PartitionNode__is_overlap(mbr, query) > 0:
                            return [node.nid]
                    return []
                else:
                    if print_info and node.is_overlap(query) > 0:
                        print("node", node.nid, "is added as result")
                    if node.is_overlap(query) > 0:
                        if  node.linked_ids:
                            query_boundary=node.max_bound_for_query(query)
                            for link_id in node.linked_ids:
                                link_node=self.nid_node_dict[link_id]
                                bound_id=link_node.if_boundary_redundant(query_boundary)
                                if bound_id>=0:
                                    return [link_id]
                        return [node.nid]
                    else:
                        return []
                    # return [node.nid] if node.is_overlap(query) > 0 else []
            node_id_list = []
            if node.is_overlap(query) <= 0:
                if print_info:
                    print("node", node.nid, "is not overlap with the query")
                pass
            elif node.is_irregular_shape_parent: # special process for irregular shape partitions!
                if print_info:
                    print("node", node.nid, "is_irregular_shape_parent")
                    
                # bound the query with parent partition's boundary, that's for the inside case determination
                bounded_query = self.__bound_query_by_boundary(query, node.boundary)
                
                overlap_irregular_shape_node_flag = True
                for nid in node.children_ids[0: -1]: # except the last one, should be the irregular shape partition
                    overlap_case = self.nid_node_dict[nid].is_overlap(bounded_query)
                    if overlap_case == 2: # inside
                        node_id_list += self.__find_overlapped_partition(self.nid_node_dict[nid], query, using_rtree_filter, print_info)
                        overlap_irregular_shape_node_flag = False
                        if print_info:
                            print("query within children node", nid, "irregular shape neighbors")
                        break
                    if overlap_case == 1: # overlap
                        node_id_list += self.__find_overlapped_partition(self.nid_node_dict[nid], query, using_rtree_filter, print_info)
                        overlap_irregular_shape_node_flag = True
                        if print_info:
                            print("query overlap children node", nid, "irregular shape neighbors")
                if overlap_irregular_shape_node_flag:
                    if print_info:
                        print("query overlap irregular shape child node", node.children_ids[-1])
                    node_id_list.append(node.children_ids[-1])
            else:
                if print_info:
                    print("searching childrens for node", node.nid)
                for nid in node.children_ids:
                    node_id_list += self.__find_overlapped_partition(self.nid_node_dict[nid], query, using_rtree_filter, print_info)
            return list(set(node_id_list))
        
        def __visualize_2d(self, dims, queries = [], path = None, focus_region = None, add_text = True, use_sci = False):
            '''
            focus_region: in the shape of boundary
            '''
            fig, ax = plt.subplots(1)
            num_dims = self.pt_root.num_dims
            
            plt.xlim(self.pt_root.boundary[dims[0]], self.pt_root.boundary[dims[0]+num_dims])
            plt.ylim(self.pt_root.boundary[dims[1]], self.pt_root.boundary[dims[1]+num_dims])


            case = 0
            for query in queries:
                lower1 = query[dims[0]]
                lower2 = query[dims[1]]
                upper1 = query[dims[0] + num_dims]
                upper2 = query[dims[1] + num_dims]

                rect = Rectangle((lower1, lower2), upper1 - lower1, upper2 - lower2, fill=False, edgecolor='r',
                                 linewidth=1)
                if add_text:
                    ax.text(upper1, upper2, case, color='b', fontsize=7)
                case += 1
                ax.add_patch(rect)
            
            leaves = self.get_leaves()
            for leaf in leaves: 
                lower1 = leaf.boundary[dims[0]]
                lower2 = leaf.boundary[dims[1]]             
                upper1 = leaf.boundary[dims[0]+num_dims]
                upper2 = leaf.boundary[dims[1]+num_dims]

                rect = Rectangle((lower1,lower2),upper1-lower1,upper2-lower2,fill=False,edgecolor='black',linewidth=1)
                if add_text:
                    ax.text(lower1, lower2, leaf.nid, fontsize=7)
                ax.add_patch(rect)
            # if self.pt_root.query_MBRs:
            #     for MBR in self.pt_root.query_MBRs:
            #         lower1 = MBR.boundary[dims[0]]
            #         lower2 = MBR.boundary[dims[1]]
            #         upper1 = MBR.boundary[dims[0]+num_dims]
            #         upper2 = MBR.boundary[dims[1]+num_dims]
            #
            #         rect = Rectangle((lower1,lower2),upper1-lower1,upper2-lower2,fill=False,edgecolor='y',linewidth=1)
            #         ax.add_patch(rect)


            ax.set_xlabel('dimension 1', fontsize=15)
            ax.set_ylabel('dimension 2', fontsize=15)
            if use_sci:
                plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0))
            #plt.xticks(np.arange(0, 400001, 100000), fontsize=10)
            #plt.yticks(np.arange(0, 20001, 5000), fontsize=10)

            plt.tight_layout() # preventing clipping the labels when save to pdf
            if focus_region is not None:
                
                # reform focus region into interleaf format
                formated_focus_region = []
                for i in range(2):
                    formated_focus_region.append(focus_region[i])
                    formated_focus_region.append(focus_region[2+i])
                
                plt.axis(formated_focus_region)
            path=f'/home/liupengju/pycharmProjects/NORA_JOIN_SIMULATION/images/{self.name}.png'
            if path is not None:
                fig.savefig(path)
            plt.show()
        
        # %matplotlib notebook
        def __visualize_3d(self, dims, queries = [], path = None, focus_region = None):
            fig = plt.figure()
            ax = Axes3D(fig)
            
            num_dims = self.pt_root.num_dims
            plt.xlim(self.pt_root.boundary[dims[0]], self.pt_root.boundary[dims[0]+num_dims])
            plt.ylim(self.pt_root.boundary[dims[1]], self.pt_root.boundary[dims[1]+num_dims])
            ax.set_zlim(self.pt_root.boundary[dims[2]], self.pt_root.boundary[dims[2]+num_dims])
            
            leaves = self.get_leaves()
            for leaf in leaves:
                
                L1 = leaf.boundary[dims[0]]
                L2 = leaf.boundary[dims[1]]
                L3 = leaf.boundary[dims[2]]      
                U1 = leaf.boundary[dims[0]+num_dims]
                U2 = leaf.boundary[dims[1]+num_dims]
                U3 = leaf.boundary[dims[2]+num_dims]
                
                # the 12 lines to form a rectangle
                x = [L1, U1]
                y = [L2, L2]
                z = [L3, L3]
                ax.plot3D(x,y,z,color="g")
                y = [U2, U2]
                ax.plot3D(x,y,z,color="g")
                z = [U3, U3]
                ax.plot3D(x,y,z,color="g")
                y = [L2, L2]
                ax.plot3D(x,y,z,color="g")

                x = [L1, L1]
                y = [L2, U2]
                z = [L3, L3]
                ax.plot3D(x,y,z,color="g")
                x = [U1, U1]
                ax.plot3D(x,y,z,color="g")
                z = [U3, U3]
                ax.plot3D(x,y,z,color="g")
                x = [L1, L1]
                ax.plot3D(x,y,z,color="g")

                x = [L1, L1]
                y = [L2, L2]
                z = [L3, U3]
                ax.plot3D(x,y,z,color="g")
                x = [U1, U1]
                ax.plot3D(x,y,z,color="g")
                y = [U2, U2]
                ax.plot3D(x,y,z,color="g")
                x = [L1, L1]
                ax.plot3D(x,y,z,color="g")
            
            for query in queries:

                L1 = query[dims[0]]
                L2 = query[dims[1]]
                L3 = query[dims[2]]
                U1 = query[dims[0]+num_dims]
                U2 = query[dims[1]+num_dims]
                U3 = query[dims[2]+num_dims]

                # the 12 lines to form a rectangle
                x = [L1, U1]
                y = [L2, L2]
                z = [L3, L3]
                ax.plot3D(x,y,z,color="r")
                y = [U2, U2]
                ax.plot3D(x,y,z,color="r")
                z = [U3, U3]
                ax.plot3D(x,y,z,color="r")
                y = [L2, L2]
                ax.plot3D(x,y,z,color="r")

                x = [L1, L1]
                y = [L2, U2]
                z = [L3, L3]
                ax.plot3D(x,y,z,color="r")
                x = [U1, U1]
                ax.plot3D(x,y,z,color="r")
                z = [U3, U3]
                ax.plot3D(x,y,z,color="r")
                x = [L1, L1]
                ax.plot3D(x,y,z,color="r")

                x = [L1, L1]
                y = [L2, L2]
                z = [L3, U3]
                ax.plot3D(x,y,z,color="r")
                x = [U1, U1]
                ax.plot3D(x,y,z,color="r")
                y = [U2, U2]
                ax.plot3D(x,y,z,color="r")
                x = [L1, L1]
                ax.plot3D(x,y,z,color="r")
            # path=f'/home/liupengju/pycharmProjects/NORA_JOIN_SIMULATION/images/3d/{time.time()}.png'
            if path is not None:
                fig.savefig(path)

            plt.show()

