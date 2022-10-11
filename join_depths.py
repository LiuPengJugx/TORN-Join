from join_until import JOIN_UNTIL
import ray
from partition_algorithm import PartitionAlgorithm
from partition_tree import PartitionTree
import numpy as np
import pickle
import random
"""
A tool class for determining the depths of every tree for multi-tables Joins and it provides some methods:
e.g., 
a) compute best depth for tree B when the depth of tree A is determined.
b) compute best depths for tree A and B when existing the constrains that DEPTH(A)=DEPTH(B) 
c) compute hyper join cost when given the depths of tree A and B
"""
class MultiTableJoin:
    def __init__(self,table_num,helper,dataset,boundary,block_size,join_attr,used_dims):
        # self.q_amount_tables = (np.array([300, 200, 150, 100, 75, 50])*1.4).astype('int')
        self.q_amount_tables = (np.array([300, 150, 120, 80, 145, 50])*1.4).astype('int')
        self.q_join_amount_tables = [70, 50, 35, 30, 20, 10]
        self.setting_dict={'max_range':[random.sample([0.1,0.15,0.2,0.3],1)[0] for _ in range(table_num)],
                    'q_type':[random.sample([0,1,3],1)[0] for _ in range(table_num)]}
        self.setting_dict={'max_range':[0.15]*table_num,'q_type':[3]*table_num}
        print(self.setting_dict)
        # self.q_join_amount_tables = (self.q_amount_tables*0.3).astype('int')
        self.dataset=dataset
        self.boundary=boundary
        self.block_size=block_size
        self.store_query_cost_dict={}
        for x in range(table_num-1): self.store_query_cost_dict[x]={x+1:{}}
        self.join_attr = join_attr
        self.used_dims=used_dims
        self.group_type=-1
        self.pa_dict = {}
        self.join_base_depth = 2
        self.max_depth=8
        self.scale_factor=helper.scale_factor
        self.scale_w = 1 / 10  # hyper cost only access the join attribute data.
        # dim_prob_filter_join = [1 if i == self.join_attr else 1 for i in range(len(used_dims))]
        dim_prob_filter_join=[1,1,1,1]
        self.training_set_tables = {}
        self.join_temp_set_tables = {}
        self.join_queries_tables = {}
        self.model_name='adaptdb'
        # step1: generate normal queries for all tables
        for tid, join_q_amount in enumerate(self.q_amount_tables[:table_num]):
            helper.maximum_range_percent=self.setting_dict['max_range'][tid]
            training_set, _ = helper.generate_queryset_and_save(join_q_amount, dim_prob=dim_prob_filter_join,queryset_type=self.setting_dict['q_type'][tid])
            # join_set,_=helper.generate_queryset_and_save(int(self.q_amount_tables[tid]*0.8),queryset_type=self.setting_dict['q_type'][tid])
            join_set=[]
            self.training_set_tables[tid] = training_set
            self.join_temp_set_tables[tid] = join_set
        # step2: generate tree dict for different depths in every table
        for i in range(table_num):
            self.pa_dict[i]=[]
        # step3: generate join queries among two tables
        for link in range(table_num - 1):
            for target in range(link + 1, table_num):
                ju = JOIN_UNTIL(self.training_set_tables[link], self.training_set_tables[target], self.join_attr, len(used_dims))
                random_link_training_set=random.sample(self.training_set_tables[link],int(self.q_amount_tables[link]/2)) 
                link_training_set_for_join=random_link_training_set+self.join_temp_set_tables[link]  
                # link_training_set_for_join=self.join_temp_set_tables[link]                        
                random_target_training_set=random.sample(self.training_set_tables[target],int(self.q_amount_tables[target]/2))                                                            
                target_training_set_for_join=random_target_training_set+self.join_temp_set_tables[target]                                            
                # target_training_set_for_join=self.join_temp_set_tables[target]  
                link_join_queries, target_join_queries = ju.generate_join_queries(link_training_set_for_join,
                                                                                  target_training_set_for_join,int(self.q_join_amount_tables[link]*0.5/2))
                # link_join_queries, target_join_queries = ju.generate_join_queries(random_link_training_set,
                                                                                #   random_target_training_set,self.q_join_amount_tables[link])
                if link not in self.join_queries_tables.keys():
                    self.join_queries_tables[link] = {}
                self.join_queries_tables[link][target] = [link_join_queries, target_join_queries]
                break


    @ray.remote(num_returns=3)
    def compute_hyper_by_depth(self, link,target,ju, a_training_set, b_training_set, join_attr, join_depth, a_join_queries,
                               b_join_queries):
        pa_A_list=self.pa_dict[link]
        pa_B_list=self.pa_dict[target]
        if len(pa_A_list)==0:
        # if join_depth-self.join_base_depth>=len(pa_A_list)-1:
            pa_A = PartitionAlgorithm()
            if self.model_name=='nora':
                pa_A.InitializeWithJNORA(a_training_set, len(self.boundary) // 2, self.boundary, self.dataset,
                                         data_threshold=self.block_size,
                                         join_attr=join_attr, using_kd=True, using_am=False, candidate_size=2,
                                         candidate_depth=1,
                                         join_depth=join_depth)
                pa_A.partition_tree.set_redundant_partition(queries=a_training_set, data_threshold=self.block_size)
                pa_A.partition_tree.name = "JNora_"+str(link)
            elif self.model_name=='adaptdb':
                pa_A.InitializeWithADP(a_training_set, len(self.boundary) // 2, self.boundary, self.dataset,
                                     data_threshold=self.block_size,
                                     join_attr=join_attr, join_depth=join_depth)
                pa_A.partition_tree.name = "Adaptdb_" + str(link)
        else:
            print(f"table {link} is exist, save time!")
            pa_A=pa_A_list[join_depth-self.join_base_depth]

        if len(pa_B_list) == 0:
        # if join_depth - self.join_base_depth >= len(pa_A_list) - 1:
            pa_B = PartitionAlgorithm()
            if self.model_name == 'nora':
                pa_B.InitializeWithJNORA(b_training_set, len(self.boundary) // 2, self.boundary, self.dataset,
                                         data_threshold=self.block_size,
                                         join_attr=join_attr, using_kd=True, using_am=False, candidate_size=2,
                                         candidate_depth=1,
                                         join_depth=join_depth)
                pa_B.partition_tree.set_redundant_partition(queries=b_training_set, data_threshold=self.block_size)
                pa_B.partition_tree.name = "JNora_"+str(target)
            elif self.model_name=='adaptdb':
                pa_B.InitializeWithADP(b_training_set, len(self.boundary) // 2, self.boundary, self.dataset,
                                     data_threshold=self.block_size,
                                     join_attr=join_attr, join_depth=join_depth)
                pa_B.partition_tree.name = "Adaptdb_" + str(target)
        else:
            print(f"table {target} is exist, save time!")
            pa_B = pa_B_list[join_depth - self.join_base_depth]
        # compute blocks for multi-table queries
        ju.set_partitioner(pa_A, pa_B)
        pa_cost = pa_A.partition_tree.evaluate_query_cost(a_training_set, True) * len(a_training_set)
        pb_cost = pa_B.partition_tree.evaluate_query_cost(b_training_set, True) * len(b_training_set)
        total_cost_detail=[]
        total_cost_detail.append(pa_cost)
        total_cost_detail.append(pb_cost)
        # min_hyper_blocks_size, min_table = ju.compute_join_blocks_for_main_table(a_join_queries, b_join_queries)
        min_hyper_blocks_size,_,_=ju.compute_total_shuffle_hyper_cost(a_join_queries,b_join_queries,group_type=self.group_type)
        total_cost_detail.append(min_hyper_blocks_size*self.scale_w)
        return pa_A,pa_B,sum(total_cost_detail)

    def save_queryset(self,suffix=''):
        base_join_path = '/home/liupengju/pycharmProjects/NORA_JOIN_SIMULATION/NORA_experiments/queryset/multi_join/'+suffix
        for tid in self.training_set_tables.keys():
            np.savetxt(f"{base_join_path}{tid}_prob2_{self.scale_factor}_train.csv", self.training_set_tables[tid], delimiter=',')
        for link in self.join_queries_tables.keys():
            for target in self.join_queries_tables[link].keys():
                with open(f"{base_join_path}join_{link}_{target}_a_prob2_{self.scale_factor}_train", 'wb') as f:
                    pickle.dump(self.join_queries_tables[link][target][0], f, True)
                with open(f"{base_join_path}join_{link}_{target}_b_prob2_{self.scale_factor}_train", 'wb') as f:
                    pickle.dump(self.join_queries_tables[link][target][1], f, True)

    def save_tree_model(self,best_depth_tables,suffix=''):
        partition_base_path = '/home/liupengju/pycharmProjects/NORA_JOIN_SIMULATION/PartitionLayout/multi_join/'+suffix
        for tid in best_depth_tables.keys():
            pa=self.pa_dict[tid][best_depth_tables[tid]-self.join_base_depth]
            pa.partition_tree.save_tree(f'{partition_base_path}prob2_jnora_{tid}_scale{self.scale_factor}')
    def save_tree_model_for_adaptdb(self,suffix=''):
        depth=3
        partition_base_path = '/home/liupengju/pycharmProjects/NORA_JOIN_SIMULATION/PartitionLayout/multi_join/'+suffix
        for tid in self.training_set_tables.keys():
            pa = PartitionAlgorithm()
            pa.InitializeWithADP(self.training_set_tables[tid], len(self.boundary) // 2, self.boundary, self.dataset, data_threshold=self.block_size,
                                   join_attr=self.join_attr, join_depth=depth)
            pa.partition_tree.name = "Adaptdb_"+str(tid)
            pa.partition_tree.save_tree(f'{partition_base_path}prob2_adaptdb_{tid}_scale{self.scale_factor}')

    def print_query_cost_for_adaptdb(self):
        candidate_depths=[3,4,5]
        min_query_cost=float('inf')
        min_adaptdb_res=[]
        min_depth=0
        for depth in candidate_depths:
            # depth = 3
            pa_list=[]
            adaptdb_res=[0]
            for tid in self.training_set_tables.keys():
                pa = PartitionAlgorithm()
                pa.InitializeWithADP(self.training_set_tables[tid], len(self.boundary) // 2, self.boundary, self.dataset, data_threshold=self.block_size,
                                       join_attr=self.join_attr, join_depth=depth)
                pa.partition_tree.name = "Adaptdb_"+str(tid)
                pa_list.append(pa)
            for tid in self.training_set_tables.keys():
                link_id=tid
                target_id=tid+1
                if tid<len(self.training_set_tables.keys())-1:
                    print(f"22222  {link_id}:{depth},,{target_id}:{depth}   total_cost:{self.compute_query_cost_between_two_depth(pa_list[link_id], pa_list[target_id], link_id, target_id)}")
                    if tid==0:
                        adaptdb_res.append(adaptdb_res[-1]+self.compute_query_cost_between_two_depth(pa_list[link_id],pa_list[target_id],link_id,target_id))

                    else:
                        adaptdb_res.append(adaptdb_res[-1]+self.compute_query_cost_between_two_depth(pa_list[link_id],pa_list[target_id],link_id,target_id,is_ignore=0))

            if sum(adaptdb_res)<min_query_cost:
                min_query_cost=sum(adaptdb_res)
                min_adaptdb_res=adaptdb_res
                min_depth=depth
        return min_adaptdb_res,min_depth

    @ray.remote(num_returns=1)
    def produce_nora_instance_list(self,tid,depth):
        pa = PartitionAlgorithm()
        pa.InitializeWithJNORA(self.training_set_tables[tid], len(self.boundary) // 2, self.boundary, self.dataset,
                                 data_threshold=self.block_size,
                                 join_attr=self.join_attr, using_kd=True, using_am=False, candidate_size=2,
                                 candidate_depth=1,
                                 join_depth=depth)
        pa.partition_tree.set_redundant_partition(queries=self.training_set_tables[tid], data_threshold=self.block_size)
        pa.partition_tree.name = "JNora_" + str(tid)
        return pa

    def print_query_cost_for_nora(self):
        candidate_depths=[3,4,5]
        # candidate_depths=[4]
        min_query_cost=float('inf')
        min_adaptdb_res=[]
        min_depth=0
        for depth in candidate_depths:
            # depth = 3
            temp_pa_list=[]
            adaptdb_res=[0]
            for tid in self.training_set_tables.keys():
                temp_pa_list.append(self.produce_nora_instance_list.remote(self,tid,depth))

            last_pa_ids=temp_pa_list.copy()
            while len(last_pa_ids):
                done_id, last_pa_ids = ray.wait(last_pa_ids)
            pa_list=[ray.get(item) for item in temp_pa_list]
            for tid in self.training_set_tables.keys():
                link_id=tid
                target_id=tid+1
                if tid<len(self.training_set_tables.keys())-1:
                    print(f"22222  {link_id}:{depth},,{target_id}:{depth}   total_cost:{self.compute_query_cost_between_two_depth(pa_list[link_id], pa_list[target_id], link_id, target_id)}")
                    if tid==0:
                        adaptdb_res.append(adaptdb_res[-1]+self.compute_query_cost_between_two_depth(pa_list[link_id],pa_list[target_id],link_id,target_id))

                    else:
                        adaptdb_res.append(adaptdb_res[-1]+self.compute_query_cost_between_two_depth(pa_list[link_id],pa_list[target_id],link_id,target_id,is_ignore=0))

            if sum(adaptdb_res)<min_query_cost:
                min_query_cost=sum(adaptdb_res)
                min_adaptdb_res=adaptdb_res
                min_depth=depth
        return min_adaptdb_res,min_depth

    def compute_query_cost_for_single_tree(self,pa,tid):
        training_set = self.training_set_tables[tid]
        return pa.partition_tree.evaluate_query_cost(training_set, True) * len(training_set)

    def compute_query_cost_between_two_depth(self,pa_A, pa_B,link,target,is_ignore=-1):
        a_training_set = self.training_set_tables[link]
        b_training_set = self.training_set_tables[target]
        a_join_queries = self.join_queries_tables[link][target][0]
        b_join_queries = self.join_queries_tables[link][target][1]
        ju = JOIN_UNTIL(a_training_set, b_training_set, self.join_attr, len(self.used_dims))
        pa_cost = pa_A.partition_tree.evaluate_query_cost(a_training_set, True) * len(a_training_set)
        pb_cost = pa_B.partition_tree.evaluate_query_cost(b_training_set, True) * len(b_training_set)
        # compute blocks for multi-table queries
        ju.set_partitioner(pa_A, pa_B)
        total_hyper_read_cost, total_shuffle_read_cost, hyper_cost_list = ju.compute_total_shuffle_hyper_cost(
            a_join_queries, b_join_queries, group_type=self.group_type)
        if is_ignore==0: pa_cost=0
        elif is_ignore==1: pb_cost=0
        total_cost_detail=[pa_cost, pb_cost, total_hyper_read_cost * self.scale_w]
        print(f"COST_DETAIL: {total_cost_detail}")
        return sum(total_cost_detail)

    def compute_best_depth_b_given_a(self,link,target,a_best_depth):
        a_training_set=self.training_set_tables[link]
        b_training_set=self.training_set_tables[target]
        a_join_queries=self.join_queries_tables[link][target][0]
        b_join_queries=self.join_queries_tables[link][target][1]
        pa_A_list=self.pa_dict[link]
        pa_B_list=self.pa_dict[target]
        ju = JOIN_UNTIL(a_training_set, b_training_set, self.join_attr, len(self.used_dims))
        best_depth_table = 0
        # try different depth for paA and paB
        b_best_depth = 0
        b_best_hyper_blocks = float('inf')
        temp_cost_res_list=[]
        self.store_query_cost_dict[link][target][a_best_depth]={}
        if best_depth_table == 0:
            pa_A = pa_A_list[a_best_depth - self.join_base_depth]
            pa_cost = pa_A.partition_tree.evaluate_query_cost(a_training_set, True) * len(a_training_set)
            join_depth_candidate = range(self.join_base_depth, self.max_depth)
            for join_depth in join_depth_candidate:
                total_cost_detail = []
                pa_B = pa_B_list[join_depth - self.join_base_depth]
                # compute blocks for single-table queries
                pb_cost = pa_B.partition_tree.evaluate_query_cost(b_training_set, True) * len(b_training_set)
                total_cost_detail.append(pa_cost)
                total_cost_detail.append(pb_cost)
                # compute blocks for multi-table queries
                ju.set_partitioner(pa_A, pa_B)
                total_hyper_read_cost,_,_ = ju.compute_total_shuffle_hyper_cost(a_join_queries, b_join_queries, group_type=self.group_type)
                total_cost_detail.append(total_hyper_read_cost * self.scale_w)
                print(f"COST DETAIL: {total_cost_detail}")
                access_and_hyper_cost = sum(total_cost_detail)
                temp_cost_res_list.append(access_and_hyper_cost)
                self.store_query_cost_dict[link][target][a_best_depth][join_depth] = access_and_hyper_cost
                if access_and_hyper_cost < b_best_hyper_blocks:
                    b_best_hyper_blocks = access_and_hyper_cost
                    b_best_depth = join_depth
                # if len(temp_cost_res_list) >= 4:
                #     if temp_cost_res_list[-1] > temp_cost_res_list[-2] > temp_cost_res_list[-3]:
                #         break
            print(f"a_depth:{a_best_depth},b_depth:{b_best_depth}, total_cost:{b_best_hyper_blocks}")
        return b_best_depth,b_best_hyper_blocks

    def compute_join_depth_between_two_tables(self,link,target):
        a_training_set=self.training_set_tables[link]
        b_training_set=self.training_set_tables[target]
        a_join_queries=self.join_queries_tables[link][target][0]
        b_join_queries=self.join_queries_tables[link][target][1]
        pa_A_list=[]
        pa_B_list=[]
        
        temp_cost_res_list = []
        hyper_blocks_size_list = []
        join_depth_candidate = range(self.join_base_depth, self.max_depth)
        for join_depth in join_depth_candidate:
            # use my join tree base on nora
            new_ju = JOIN_UNTIL(a_training_set, b_training_set, self.join_attr, len(self.used_dims))
            pa_A,pa_B,min_hyper_blocks_size = self.compute_hyper_by_depth.remote(self,link,target,new_ju, a_training_set, b_training_set,
                                                                              self.join_attr, join_depth, a_join_queries,
                                                                              b_join_queries)
            pa_A_list.append(pa_A)
            pa_B_list.append(pa_B)
            hyper_blocks_size_list.append(min_hyper_blocks_size)
        last_pa_ids = hyper_blocks_size_list.copy()
        while len(last_pa_ids):
            done_id, last_pa_ids = ray.wait(last_pa_ids)
        # get data by objectRef
        hyper_blocks_size_list = [ray.get(item) for item in hyper_blocks_size_list]
        pa_A_list=[ray.get(item) for item in pa_A_list]
        pa_B_list=[ray.get(item) for item in pa_B_list]
        self.pa_dict[link]=pa_A_list
        self.pa_dict[target]=pa_B_list
        print(f"select best depth for A: {hyper_blocks_size_list}")
        a_best_hyper_blocks = min(hyper_blocks_size_list)
        link_best_depth = hyper_blocks_size_list.index(a_best_hyper_blocks) + self.join_base_depth
        target_best_depth,best_query_cost=self.compute_best_depth_b_given_a(link,target,link_best_depth)
        # a_best_depths=[a_best_depth]
        # if fixed_link_depth!=a_best_depth and fixed_link_depth>0:
        #     a_best_depths.append(fixed_link_depth)
        return link_best_depth, target_best_depth,best_query_cost