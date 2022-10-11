import os.path
from random import random
import random as rd
import ray
import numpy as np
from data_helper import DatasetAndQuerysetHelper
from partition_algorithm import PartitionAlgorithm
from partition_tree import PartitionTree
import matplotlib.pyplot as plt
from join_until import JOIN_UNTIL
import pickle
"""
Experiment Support Class: provide the main function of testing the performance of different algorithms
"""

# Step 1: Select used attribute dimensions of table data
# used_dims = [1,2,3]  #store_sales with join_attr(index)=0
used_dims = [0,1,2,4]  #lineitem

# Step 2: Specify some parameters for scale factor / sampling rate / scaled block size etc.
scale_factor=1
# this is important! block_size (here) * (1/sampling rate) = 128MB( = 1,000,000 records in Spark cluster)
# # 1.2G == 5998152 == 6000000, block_size (here) * (1/sampling rate) = 128MB( = 600,000 records in Spark cluster)
sampling_rate=1/scale_factor
block_size = 10000 # 1.1M compared to 6M

# Step 3: Define the workload generator, including the path of queries, datasets,
base_path = '/home/liupengju/pycharmProjects/NORA_JOIN_SIMULATION/NORA_experiments'
helper = DatasetAndQuerysetHelper(used_dimensions=used_dims, scale_factor=scale_factor,base_path=base_path)
# helper = DatasetAndQuerysetHelper(used_dimensions=used_dims, scale_factor=scale_factor,base_path=base_path,tab_name='store_sales')
# helper.total_dims=23
# helper.domain_dims=23
# helper.generate_dataset_and_save(base_path+'/dataset/store_sales_50.tbl')
helper.maximum_range_percent = 0.15
dataset, domains = helper.load_dataset(used_dims)
boundary = [interval[0] for interval in domains] + [interval[1] for interval in domains]  # the boundaries of entire table data

# Step 4: user-defined parameters for grouping split / bounding split
max_active_ratio=3  # max_active_ratio=50

pa_algos=[]

# Exp: Use saved data layout to test the performance of singe-table queries
def test_save_compare_without_join():
    global pa_algos
    # dim_prob = [0 if i >1 else 1 for i in range(len(used_dims))]
    training_set = np.genfromtxt(f"/home/liupengju/pycharmProjects/NORA_JOIN_SIMULATION/NORA_experiments/queryset/prob2_{scale_factor}_train.csv", delimiter=',')
    partition_base_path = '/home/liupengju/pycharmProjects/NORA_JOIN_SIMULATION/PartitionLayout/'

    # # = = = = = Test PartitionAlgorithm (QDT) = = = = =  5.882  4.87
    pa2 = PartitionAlgorithm()
    pa2.partition_tree=PartitionTree(len(used_dims))
    pa2.partition_tree.load_tree(partition_base_path + 'prob2_qdtree_scale' + str(scale_factor))

    #
    # # = = = = = Test PartitionAlgorithm (PAW) = = = = = 12.98  8.13
    pa4_2 = PartitionAlgorithm()
    pa4_2.partition_tree = PartitionTree(len(used_dims))
    pa4_2.partition_tree.load_tree(partition_base_path + 'prob2_paw_scale' + str(scale_factor))

    # = = = = = Test PartitionAlgorithm (NORA-Beam+Redundant) = = = = =
    pa7 = PartitionAlgorithm()
    pa7.partition_tree = PartitionTree(len(used_dims))
    pa7.partition_tree.load_tree(partition_base_path + 'prob2_nora_scale' + str(scale_factor))
    pa_algos += [pa2, pa4_2, pa7]
    cost_res = []
    for pa in pa_algos:
        cost_res.append(pa.partition_tree.evaluate_query_cost(training_set, True))
    print(np.round(np.array(cost_res)/dataset.shape[0],4))
    # print_exec_result(training_set)

# Exp: Test the performance of singe-table queries.
# parameter: save, control whether to generate data layout file
def compare_without_join(save=False):
    global pa_algos
    # dim_prob = [0 if i >1 else 1 for i in range(len(used_dims))]
    # training_set, _ = helper.generate_queryset_and_save(200, queryset_type=3)  #lineitem
    training_set, _ = helper.generate_queryset_and_save(400, queryset_type=3,suffix='tpcds/')  # tpcds

    # TPCH Dataset
    # training_set, testing_set = helper.generate_queryset_and_save(100, queryset_type=2)  # uniform
    # training_set, testing_set = helper.generate_queryset_and_save(200, queryset_type=4)  # skew workload

    # OSM Dataset (workload distribution)
    # training_set = helper._DatasetAndQuerysetHelper__generate_random_query(200, [1,1], [[-180,180],[-90,90]], [36,18]) # uniform
    # training_set = helper._DatasetAndQuerysetHelper__generate_distribution_query(200, [1, 1], [[-180, 180], [-90, 90]],[36, 18])  # skew

    # partition_base_path = '/home/liupengju/pycharmProjects/NORA_JOIN_SIMULATION/PartitionLayout/'
    partition_base_path = '/home/liupengju/pycharmProjects/NORA_JOIN_SIMULATION/PartitionLayout/tpcds/'
    # helper.visualize_queryset_and_dataset([0, 1], training_set, testing_set)
    # = = = = = Test PartitionAlgorithm (KDT) = = = = =
    # pa1 = PartitionAlgorithm()
    # pa1.InitializeWithKDT(len(boundary)//2, boundary, dataset, data_threshold = block_size) # 447
    # pa1.partition_tree.visualize(queries = training_set,add_text = False, use_sci = True)

    # # = = = = = Test PartitionAlgorithm (QDT) = = = = =  5.882  4.87
    pa2 = PartitionAlgorithm()
    pa2.InitializeWithQDT(training_set, len(boundary) // 2, boundary, dataset, data_threshold=block_size)
    pa2.partition_tree.visualize(queries=training_set, add_text=False, use_sci=True)
    pa_algos += [pa2]

    #
    # # = = = = = Test PartitionAlgorithm (PAW) = = = = = 12.98  8.13
    pa4_2 = PartitionAlgorithm()
    pa4_2.InitializeWithPAW(training_set, len(boundary) // 2, boundary, dataset, block_size, max_active_ratio=max_active_ratio,strategy=1)
    pa4_2.partition_tree.visualize(queries=training_set, add_text=False, use_sci=True)
    pa_algos += [pa4_2]
    # # = = = = = Test PartitionAlgorithm (PAW-BeamSearch + redundant partition) = = = = =
    # pa4_3 = PartitionAlgorithm()
    # pa4_3.InitializeWithPAW(training_set, len(boundary)//2, boundary, dataset, block_size, using_beam_search=True, candidate_size = 2, candidate_depth = 2, max_active_ratio=max_active_ratio, strategy = 1)
    # pa4_3.partition_tree.set_redundant_partition(queries=training_set,data_threshold=block_size)
    # pa4_3.partition_tree.visualize(queries = training_set,add_text = False, use_sci = True)

    # # = = = = = Test PartitionAlgorithm (PAW-SimplifyBeamSearch+with redundant partitions) = = = = =
    # pa6 = PartitionAlgorithm()
    # pa6.InitializeWithPAW2(training_set, len(boundary) // 2, boundary, dataset, block_size,using_beam_search=True, candidate_size = 4, candidate_depth = 2,max_active_ratio=max_active_ratio,strategy = 0)
    # pa6.partition_tree.set_redundant_partition(queries=training_set,data_threshold=block_size)
    # pa6.partition_tree.visualize(queries=training_set, add_text=False, use_sci=True)

    # # = = = = = Test PartitionAlgorithm (AMT) = = = = =
    # pa5 = PartitionAlgorithm()
    # pa5.InitializeWithAMT(len(boundary)//2, boundary, dataset, data_threshold = block_size)
    # pa5.partition_tree.visualize(queries = training_set,add_text = False, use_sci = True)

    # = = = = = Test PartitionAlgorithm (NORA-Beam+Redundant) = = = = =
    pa7 = PartitionAlgorithm()
    pa7.InitializeWithNORA(training_set, len(boundary)//2, boundary, dataset, data_threshold = block_size, using_beam_search=True, candidate_size = 2, candidate_depth = 2,
                           using_1_by_1 = True, using_kd = True)
    pa7.partition_tree.evaluate_query_cost(training_set, True)
    pa7.partition_tree.set_redundant_partition(queries=training_set,data_threshold=block_size)
    pa7.partition_tree.visualize(queries=training_set, add_text=False, use_sci=True)
    print(f"redundant ratio:{round(pa7.partition_tree.redundant_data_ratio, 4)}, extra size:{pa7.partition_tree.redundant_data_ratio * pa7.partition_tree.table_dataset_size}")
    pa_algos += [pa7]
    if save:
        if not os.path.isfile(partition_base_path + 'prob' + str(2) + '_qdtree_scale' + str(scale_factor)):
            pa2.partition_tree.save_tree(partition_base_path + 'prob' + str(2) + '_qdtree_scale' + str(scale_factor))
            pa4_2.partition_tree.save_tree(partition_base_path + 'prob' + str(2) + '_paw_scale' + str(scale_factor))
            pa7.partition_tree.save_tree(partition_base_path + 'prob' + str(2) + '_nora_scale' + str(scale_factor))

    print_exec_result(training_set)


# Exp: Given two data layouts with specify depths, we test its hyper-join cost
@ray.remote(num_returns=3)
def compute_hyper_by_depth(ju,a_training_set,b_training_set,join_attr,join_depth,a_join_queries,b_join_queries):
    pa_A = PartitionAlgorithm()
    pa_A.InitializeWithJNORA(a_training_set, len(boundary) // 2, boundary, dataset,
                             data_threshold=block_size,
                             join_attr=join_attr, using_kd=True, using_am=False, candidate_size=2,
                             candidate_depth=1,
                             join_depth=join_depth)
    pa_A.partition_tree.set_redundant_partition(queries=a_training_set, data_threshold=block_size)
    pa_A.partition_tree.name = "JNora_A"
    pa_B = PartitionAlgorithm()
    pa_B.InitializeWithJNORA(b_training_set, len(boundary) // 2, boundary, dataset,
                             data_threshold=block_size,
                             join_attr=join_attr, using_kd=True, using_am=False, candidate_size=2,
                             candidate_depth=1,
                             join_depth=join_depth)
    pa_B.partition_tree.set_redundant_partition(queries=b_training_set, data_threshold=block_size)
    pa_B.partition_tree.name = "JNora_B"
    # compute blocks for multi-table queries
    ju.set_partitioner(pa_A, pa_B)
    min_hyper_blocks_size, min_table = ju.compute_join_blocks_for_main_table(a_join_queries, b_join_queries)
    return pa_A,pa_B,min_hyper_blocks_size

# Exp: Given stored two data layouts with specify depths, we test its hyper-join cost
def test_save_compare_join_with_shuffle():
    global pa_algos
    join_attr = 0
    scale_w = 1 / 10  # hyper cost only access the join attribute data.
    base_join_path='/home/liupengju/pycharmProjects/NORA_JOIN_SIMULATION/NORA_experiments/queryset/join/'
    a_training_set = np.genfromtxt(f"{base_join_path}a_prob2_{scale_factor}_train.csv", delimiter=',')
    b_training_set = np.genfromtxt(f"{base_join_path}b_prob2_{scale_factor}_train.csv", delimiter=',')
    ju = JOIN_UNTIL(a_training_set, b_training_set, join_attr, len(used_dims))
    with open(f"{base_join_path}join_a_prob2_{scale_factor}_train",'rb') as f:
        a_join_queries=pickle.load(f)
    with open(f"{base_join_path}join_b_prob2_{scale_factor}_train",'rb') as f:
        b_join_queries=pickle.load(f)
    # a_training_set_for_join=np.genfromtxt(f"{base_join_path}join_a_prob2_{scale_factor}_train.csv", delimiter=',')
    # b_training_set_for_join=np.genfromtxt(f"{base_join_path}join_b_prob2_{scale_factor}_train.csv", delimiter=',')
    # a_join_queries, b_join_queries = ju.generate_join_queries(a_training_set_for_join.tolist(), b_training_set_for_join.tolist())
    partition_base_path = '/home/liupengju/pycharmProjects/NORA_JOIN_SIMULATION/PartitionLayout/join/'
    algo_names=['adaptdb','jnora']
    group_types=[1,3]
    final_query_cost_res = []
    for i in range(2):
        name=algo_names[i]
        pa_A = PartitionAlgorithm()
        pa_A.partition_tree=PartitionTree(len(used_dims))
        pa_A.partition_tree.join_attr=0
        pa_A.partition_tree.load_tree(partition_base_path + 'prob2' + '_'+name+'_A_scale' + str(scale_factor))

        pa_B = PartitionAlgorithm()
        pa_B.partition_tree = PartitionTree(len(used_dims))
        pa_B.partition_tree.join_attr = 0
        pa_B.partition_tree.load_tree(partition_base_path + 'prob2' + '_'+name+'_B_scale' + str(scale_factor))
        pa_cost = pa_A.partition_tree.evaluate_query_cost(a_training_set, True) * len(a_training_set)
        pb_cost = pa_B.partition_tree.evaluate_query_cost(b_training_set, True) * len(b_training_set)
        ju.set_partitioner(pa_A, pa_B)
        total_hyper_read_cost, total_shuffle_read_cost, hyper_cost_list = ju.compute_total_shuffle_hyper_cost(a_join_queries, b_join_queries, group_type=group_types[i])
        final_query_cost_res.append(sum([pa_cost,pb_cost,total_hyper_read_cost*scale_w]))
    print(final_query_cost_res)


# EXp: Generate queries with different query amounts and join ratio, and measure the hyper-join cost and single-table query cost of data layouts optimized by bottom-up algorithm and QDG
def compare_join_depth_with_shuffle_with_ray(save_model=False):
    # vary_query_amount=[(200,150),(300,200),(500,300),(600,350)]
    vary_query_amount=[(300,200)]
    join_original_ratio=0.3
    join_q_ratio=[0,0.05,0.1,0.2,0.3,0.5,1]
    result_dict={}
    for a_b_amount in vary_query_amount:
    # for join_original_ratio in join_q_ratio:
        # a_b_amount=(300,200)

        join_amount=int(a_b_amount[1] / 2 * join_original_ratio)
        join_a_q_amount = a_b_amount[0]-join_amount
        join_b_q_amount = a_b_amount[1]-join_amount

        join_attr = 0
        dim_prob_filter_join = [0 if i == join_attr else 1 for i in range(len(used_dims))]
        a_training_set, _ = helper.generate_queryset_and_save(join_a_q_amount, dim_prob=dim_prob_filter_join,
                                                              queryset_type=3)
        b_training_set, _ = helper.generate_queryset_and_save(join_b_q_amount, dim_prob=dim_prob_filter_join,
                                                              queryset_type=3)
        a_training_set_for_join=rd.sample(a_training_set,int(join_a_q_amount/2))
        b_training_set_for_join=rd.sample(b_training_set,int(join_b_q_amount/2))
        # a_training_set_for_join, _ = helper.generate_queryset_and_save(int(join_a_q_amount/2), queryset_type=3)
        # b_training_set_for_join, _ = helper.generate_queryset_and_save(int(join_b_q_amount/2), queryset_type=3)
        base_join_path='/home/liupengju/pycharmProjects/NORA_JOIN_SIMULATION/NORA_experiments/queryset/join/'
        ju = JOIN_UNTIL(a_training_set, b_training_set, join_attr, len(used_dims))
        a_join_queries, b_join_queries = ju.generate_join_queries(a_training_set_for_join, b_training_set_for_join,join_amount=join_amount)
        if not os.path.isfile(f"{base_join_path}a_prob2_{scale_factor}_train.csv"):
            np.savetxt(f"{base_join_path}a_prob2_{scale_factor}_train.csv",a_training_set,delimiter=',')
            np.savetxt(f"{base_join_path}b_prob2_{scale_factor}_train.csv",b_training_set,delimiter=',')
            # np.savetxt(f"{base_join_path}join_a_prob2_{scale_factor}_train.txt",a_join_queries,delimiter=',')
            # np.savetxt(f"{base_join_path}join_b_prob2_{scale_factor}_train.txt",b_join_queries,delimiter=',')
            print(a_join_queries)
            print(b_join_queries)
            with open(f"{base_join_path}join_a_prob2_{scale_factor}_train",'wb') as f:
                pickle.dump(a_join_queries, f, True)
            with open(f"{base_join_path}join_b_prob2_{scale_factor}_train",'wb') as f:
                pickle.dump(b_join_queries, f, True)

        final_query_cost_res=[]
        partition_base_path = '/home/liupengju/pycharmProjects/NORA_JOIN_SIMULATION/PartitionLayout/join/'
        scale_w=1/10  # hyper cost only access the join attribute data.
        for i in range(2):
            if i==0:
                candidate_depths = [3, 4, 5]
                min_query_cost = float('inf')
                min_adaptdb_res = []
                for join_depth in candidate_depths:
                    # join_depth=3 #decided by experience
                    total_cost_detail = []
                    pa_A = PartitionAlgorithm()
                    pa_A.InitializeWithADP(a_training_set, len(boundary) // 2, boundary, dataset, data_threshold=block_size,
                                             join_attr=join_attr,join_depth=join_depth)
                    pa_A.partition_tree.name = "adaptdb_A"
                    pa_B = PartitionAlgorithm()
                    pa_B.InitializeWithADP(b_training_set, len(boundary) // 2, boundary, dataset, data_threshold=block_size,
                                             join_attr=join_attr,join_depth=join_depth)
                    pa_B.partition_tree.name = "adaptdb_B"
                    pa_cost = pa_A.partition_tree.evaluate_query_cost(a_training_set, True) * len(a_training_set)
                    pb_cost = pa_B.partition_tree.evaluate_query_cost(b_training_set, True) * len(b_training_set)
                    total_cost_detail.append(pa_cost)
                    total_cost_detail.append(pb_cost)
                    # compute blocks for multi-table queries
                    ju.set_partitioner(pa_A, pa_B)
                    total_hyper_read_cost, total_shuffle_read_cost, hyper_cost_list = ju.compute_total_shuffle_hyper_cost(
                        a_join_queries, b_join_queries,group_type=1)
                    total_cost_detail.append(total_hyper_read_cost*scale_w)
                    print(f"COST_DETAIL: {total_cost_detail}")
                    if sum(total_cost_detail) < min_query_cost:
                        min_query_cost = sum(total_cost_detail)
                        min_adaptdb_res = total_cost_detail
                        if save_model:
                            pa_A.partition_tree.save_tree(partition_base_path + 'prob2' + '_adaptdb_A_scale' + str(scale_factor))
                            pa_B.partition_tree.save_tree(partition_base_path + 'prob2' + '_adaptdb_B_scale' + str(scale_factor))
                final_query_cost_res.append(sum(min_adaptdb_res))
            else:
                ray.init(num_cpus=8)
                temp_cost_res_list=[]
                pa_A_list = []
                pa_B_list = []
                hyper_blocks_size_list=[]
                join_depth_candidate = range(2, 8)
                join_base_depth = join_depth_candidate[0]
                for join_depth in join_depth_candidate:
                    # use my join tree base on nora
                    new_ju = JOIN_UNTIL(a_training_set, b_training_set, join_attr, len(used_dims))
                    pa_A, pa_B, min_hyper_blocks_size=compute_hyper_by_depth.remote(new_ju,a_training_set,b_training_set,join_attr,join_depth,a_join_queries,b_join_queries)
                    pa_A_list.append(pa_A)
                    pa_B_list.append(pa_B)
                    hyper_blocks_size_list.append(min_hyper_blocks_size)
                last_pa_ids=hyper_blocks_size_list.copy()
                while len(last_pa_ids):
                    done_id, last_pa_ids = ray.wait(last_pa_ids)
                # get data by objectRef
                hyper_blocks_size_list=[ray.get(item) for item in hyper_blocks_size_list]
                pa_A_list=[ray.get(item) for item in pa_A_list]
                pa_B_list=[ray.get(item) for item in pa_B_list]
                print("final:",hyper_blocks_size_list)
                a_best_hyper_blocks=min(hyper_blocks_size_list)
                a_best_depth=hyper_blocks_size_list.index(a_best_hyper_blocks)+join_base_depth
                best_depth_table = 0
                ray.shutdown()
                # try different depth for paA and paB
                if best_depth_table == 0:
                    temp_cost_res_list.clear()
                    pa_A = pa_A_list[a_best_depth-join_base_depth]
                    pa_cost = pa_A.partition_tree.evaluate_query_cost(a_training_set, True) * len(a_training_set)
                    b_best_depth = 0
                    b_best_hyper_blocks = float('inf')
                    for join_depth in join_depth_candidate:
                        total_cost_detail = []
                        pa_B = pa_B_list[join_depth-join_base_depth]
                        # compute blocks for single-table queries
                        pb_cost = pa_B.partition_tree.evaluate_query_cost(b_training_set, True) * len(b_training_set)
                        total_cost_detail.append(pa_cost)
                        total_cost_detail.append(pb_cost)
                        # compute blocks for multi-table queries
                        ju.set_partitioner(pa_A, pa_B)
                        total_hyper_read_cost, total_shuffle_read_cost, hyper_cost_list = ju.compute_total_shuffle_hyper_cost(a_join_queries, b_join_queries,group_type=3)
                        total_cost_detail.append(total_hyper_read_cost*scale_w)
                        print(f"COST_DETAIL: {total_cost_detail}")
                        temp_cost_res_list.append(sum(total_cost_detail))
                        if sum(total_cost_detail) < b_best_hyper_blocks:
                            b_best_hyper_blocks = sum(total_cost_detail)
                            b_best_depth = join_depth
                        if len(temp_cost_res_list) >= 4:
                            if temp_cost_res_list[-1] > temp_cost_res_list[-2] > temp_cost_res_list[-3]:
                                break
                    print(f"a_depth:{a_best_depth},b_depth:{b_best_depth}")
                    pa_A_best=pa_A_list[a_best_depth-join_base_depth]
                    pa_B_best=pa_B_list[b_best_depth-join_base_depth]
                    if save_model:
                        pa_A_best.partition_tree.save_tree(partition_base_path + 'prob2' + '_jnora_A_scale' + str(scale_factor))
                        pa_B_best.partition_tree.save_tree(partition_base_path + 'prob2' + '_jnora_B_scale' + str(scale_factor))
                print(temp_cost_res_list)
                final_query_cost_res.append(min(temp_cost_res_list))
        print(final_query_cost_res)
        result_dict[f'{str(a_b_amount)} :: {join_a_q_amount+join_b_q_amount+join_amount*2}']=final_query_cost_res
    print(result_dict)
# def compare_join_depth_with_shuffle(save_model=True):
#     join_a_q_amount = 300
#     join_b_q_amount = 200
#     join_attr = 0
#     dim_prob_filter_join = [0 if i == join_attr else 1 for i in range(len(used_dims))]
#     a_training_set, _ = helper.generate_queryset_and_save(join_a_q_amount, dim_prob=dim_prob_filter_join,
#                                                           queryset_type=3)
#     b_training_set, _ = helper.generate_queryset_and_save(join_b_q_amount, dim_prob=dim_prob_filter_join,
#                                                           queryset_type=3)
#     a_training_set_for_join, _ = helper.generate_queryset_and_save(int(join_a_q_amount/2), queryset_type=3)
#     b_training_set_for_join, _ = helper.generate_queryset_and_save(int(join_b_q_amount/2), queryset_type=3)
#     base_join_path='/home/liupengju/pycharmProjects/NORA_JOIN_SIMULATION/NORA_experiments/queryset/join/'
#     ju = JOIN_UNTIL(a_training_set, b_training_set, join_attr, len(used_dims))
#     a_join_queries, b_join_queries = ju.generate_join_queries(a_training_set_for_join, b_training_set_for_join)
#
#     if not os.path.isfile(f"{base_join_path}a_prob2_{scale_factor}_train.csv"):
#         np.savetxt(f"{base_join_path}a_prob2_{scale_factor}_train.csv",a_training_set,delimiter=',')
#         np.savetxt(f"{base_join_path}b_prob2_{scale_factor}_train.csv",b_training_set,delimiter=',')
#         # np.savetxt(f"{base_join_path}join_a_prob2_{scale_factor}_train.txt",a_join_queries,delimiter=',')
#         # np.savetxt(f"{base_join_path}join_b_prob2_{scale_factor}_train.txt",b_join_queries,delimiter=',')
#         print(a_join_queries)
#         print(b_join_queries)
#         with open(f"{base_join_path}join_a_prob2_{scale_factor}_train",'wb') as f:
#             pickle.dump(a_join_queries, f, True)
#         with open(f"{base_join_path}join_b_prob2_{scale_factor}_train",'wb') as f:
#             pickle.dump(b_join_queries, f, True)
#
#     final_query_cost_res=[]
#     partition_base_path = '/home/liupengju/pycharmProjects/NORA_JOIN_SIMULATION/PartitionLayout/join/'
#     scale_w=1/10  # hyper cost only access the join attribute data.
#     for i in range(2):
#         if i==0:
#             join_depth=3 #decided by experience
#             total_cost_detail = []
#             pa_A = PartitionAlgorithm()
#             pa_A.InitializeWithADP(a_training_set, len(boundary) // 2, boundary, dataset, data_threshold=block_size,
#                                      join_attr=join_attr,join_depth=join_depth)
#             pa_A.partition_tree.name = "adaptdb_A"
#             pa_B = PartitionAlgorithm()
#             pa_B.InitializeWithADP(b_training_set, len(boundary) // 2, boundary, dataset, data_threshold=block_size,
#                                      join_attr=join_attr,join_depth=join_depth)
#             pa_B.partition_tree.name = "adaptdb_B"
#             pa_cost = pa_A.partition_tree.evaluate_query_cost(a_training_set, True) * len(a_training_set)
#             pb_cost = pa_B.partition_tree.evaluate_query_cost(b_training_set, True) * len(b_training_set)
#             total_cost_detail.append(pa_cost)
#             total_cost_detail.append(pb_cost)
#             # compute blocks for multi-table queries
#             ju.set_partitioner(pa_A, pa_B)
#             total_hyper_read_cost, total_shuffle_read_cost, hyper_cost_list = ju.compute_total_shuffle_hyper_cost(
#                 a_join_queries, b_join_queries,group_type=1)
#             total_cost_detail.append(total_hyper_read_cost*scale_w)
#             final_query_cost_res.append(sum(total_cost_detail))
#             if save_model:
#                 pa_A.partition_tree.save_tree(partition_base_path + 'prob2' + '_adaptdb_A_scale' + str(scale_factor))
#                 pa_B.partition_tree.save_tree(partition_base_path + 'prob2' + '_adaptdb_B_scale' + str(scale_factor))
#         else:
#             temp_cost_res_list=[]
#             using_am_flag = False
#             a_best_depth = 0
#             best_depth_table = 0
#             a_best_hyper_blocks = float('inf')
#             pa_A_list = []
#             pa_B_list = []
#             join_depth_candidate = range(2, 8)
#             for join_depth in join_depth_candidate:
#                 # use my join tree base on nora
#                 pa_A = PartitionAlgorithm()
#                 pa_A.InitializeWithJNORA(a_training_set, len(boundary) // 2, boundary, dataset,
#                                          data_threshold=block_size,
#                                          join_attr=join_attr, using_kd=True, using_am=using_am_flag, candidate_size=2,
#                                          candidate_depth=1,
#                                          join_depth=join_depth)
#                 pa_A.partition_tree.set_redundant_partition(queries=a_training_set, data_threshold=block_size)
#                 pa_A.partition_tree.name = "JNora_A"
#                 pa_A_list.append(pa_A)
#                 pa_B = PartitionAlgorithm()
#                 pa_B.InitializeWithJNORA(b_training_set, len(boundary) // 2, boundary, dataset,
#                                          data_threshold=block_size,
#                                          join_attr=join_attr, using_kd=True, using_am=using_am_flag, candidate_size=2,
#                                          candidate_depth=1,
#                                          join_depth=join_depth)
#                 pa_B.partition_tree.set_redundant_partition(queries=b_training_set, data_threshold=block_size)
#                 pa_B.partition_tree.name = "JNora_B"
#                 pa_B_list.append(pa_B)
#
#                 # compute blocks for multi-table queries
#                 ju.set_partitioner(pa_A, pa_B)
#                 min_hyper_blocks_size, min_table = ju.compute_join_blocks_for_main_table(a_join_queries, b_join_queries)
#                 if min_hyper_blocks_size < a_best_hyper_blocks:
#                     a_best_hyper_blocks = min_hyper_blocks_size
#                     best_depth_table = min_table
#                     a_best_depth = join_depth
#                 temp_cost_res_list.append(min_hyper_blocks_size)
#                 # check the slope for cost_res curve
#                 if len(temp_cost_res_list) >= 4:
#                     if temp_cost_res_list[-1] > temp_cost_res_list[-2] > temp_cost_res_list[-3]:
#                         break
#             # try different depth for paA and paB
#             if best_depth_table == 0:
#                 temp_cost_res_list.clear()
#                 join_base_depth=join_depth_candidate[0]
#                 pa_A = pa_A_list[a_best_depth-join_base_depth]
#                 pa_cost = pa_A.partition_tree.evaluate_query_cost(a_training_set, True) * len(a_training_set)
#                 b_best_depth = 0
#                 b_best_hyper_blocks = float('inf')
#                 for join_depth in join_depth_candidate:
#                     total_cost_detail = []
#                     if join_depth-join_base_depth > len(pa_B_list) - 1:
#                         pa_B = PartitionAlgorithm()
#                         pa_B.InitializeWithJNORA(b_training_set, len(boundary) // 2, boundary, dataset,
#                                                  data_threshold=block_size,
#                                                  join_attr=join_attr, using_kd=True, using_am=using_am_flag,
#                                                  candidate_size=2, candidate_depth=2,
#                                                  join_depth=join_depth)
#                         pa_B.partition_tree.set_redundant_partition(queries=b_training_set, data_threshold=block_size)
#                         pa_B.partition_tree.name = "JNora_B"
#                         pa_B_list.append(pa_B)
#                     else:
#                         pa_B = pa_B_list[join_depth-join_base_depth]
#                     # compute blocks for single-table queries
#                     pb_cost = pa_B.partition_tree.evaluate_query_cost(b_training_set, True) * len(b_training_set)
#                     total_cost_detail.append(pa_cost)
#                     total_cost_detail.append(pb_cost)
#                     # compute blocks for multi-table queries
#                     ju.set_partitioner(pa_A, pa_B)
#                     total_hyper_read_cost, total_shuffle_read_cost, hyper_cost_list = ju.compute_total_shuffle_hyper_cost(a_join_queries, b_join_queries,group_type=3)
#                     total_cost_detail.append(total_hyper_read_cost*scale_w)
#                     temp_cost_res_list.append(sum(total_cost_detail))
#                     if sum(total_cost_detail) < b_best_hyper_blocks:
#                         b_best_hyper_blocks = sum(total_cost_detail)
#                         b_best_depth = join_depth
#                     if len(temp_cost_res_list) >= 4:
#                         if temp_cost_res_list[-1] > temp_cost_res_list[-2] > temp_cost_res_list[-3]:
#                             break
#                 print(f"a_depth:{a_best_depth},b_depth:{b_best_depth}")
#                 pa_A_best=pa_A_list[a_best_depth-join_base_depth]
#                 pa_B_best=pa_B_list[b_best_depth-join_base_depth]
#                 if save_model:
#                     pa_A_best.partition_tree.save_tree(partition_base_path + 'prob2' + '_jnora_A_scale' + str(scale_factor))
#                     pa_B_best.partition_tree.save_tree(partition_base_path + 'prob2' + '_jnora_B_scale' + str(scale_factor))
#             print(temp_cost_res_list)
#             final_query_cost_res.append(min(temp_cost_res_list))
#     print(final_query_cost_res)

# Exp: we test the performance of join queries over AdaptDB and TORN
def compare_hyper_join_with_multitable(table_num=4,save_model=False,save_queryset=False,suffix=''):
    ray.init(num_cpus=12)
    from join_depths import MultiTableJoin
    join_attr=0
    table_names = ['a', 'b', 'c', 'd', 'e', 'f']
    adaptdb_improved_res,nora_res,adaptdb_improved_res2=[],[],[0]
    # step1: generate normal queries for all tables and generate join queries among two tables
    multi_table_join=MultiTableJoin(table_num,helper,dataset,boundary,block_size,join_attr,used_dims)
    join_base_depth=multi_table_join.join_base_depth
    save_cost_dict_list,adaptdb_fixed_res,nora_fixed_res=[],[],[]
    # -----save for jnora tree-----------
    # step2: sort by their workload weight
    # for model_name in ['adaptdb','nora']:
    for model_name in ['nora']:
        multi_table_join.model_name=model_name
        multi_table_join.group_type=3 if model_name=='nora' else 1
        multi_table_join.pa_dict = {}
        for i in range(table_num): multi_table_join.pa_dict[i] = []
        assign_priority=[i for i in range(table_num)]
        store_query_cost_dict={}
        save_cost_for_tables_dict={}
        best_depth_tables={}
        for i in range(table_num):
            best_depth_tables[i]=0
        idea_depth_dict_two_tabs={}
        for link in assign_priority[:-1]:
            idea_depth_dict_two_tabs[link]={}
            target=assign_priority[link+1]
            # t_pos=assign_priority.index(target)
            link_best_depth,target_best_depth,query_cost=multi_table_join.compute_join_depth_between_two_tables(link,target)
            print(f"11111 {link}:{link_best_depth}, {target}:{target_best_depth}, total_cost:{query_cost}")
            idea_depth_dict_two_tabs[link][target]=[link_best_depth,target_best_depth]
        # exit(0)
        for link in assign_priority[:-1]:
            target=assign_priority[link+1]
            idea_link_depth,idea_target_depth=idea_depth_dict_two_tabs[link][target][0],idea_depth_dict_two_tabs[link][target][1]
            if link==0: 
                best_depth_tables[link],best_depth_tables[target]=idea_link_depth,idea_target_depth
                continue
            if target==assign_priority[-1]:
                if best_depth_tables[link]==idea_link_depth: best_depth_tables[target]=idea_target_depth
                else: best_depth_tables[target],_=multi_table_join.compute_best_depth_b_given_a(link,target,best_depth_tables[link])
                continue
            last_table=assign_priority[assign_priority.index(link)-1]
            next_table=assign_priority[assign_priority.index(target)+1]
            next_idea_target_depth,next_idea_next_depth=idea_depth_dict_two_tabs[target][next_table][0],idea_depth_dict_two_tabs[target][next_table][1]
            
            # link>=1
            # compare the group (>=3 elements)
            print("START COMPARING...")
            
            pa_A_list=multi_table_join.pa_dict[link]
            pa_B_list=multi_table_join.pa_dict[target]
            pa_last_list=multi_table_join.pa_dict[last_table]
            pa_next_list=multi_table_join.pa_dict[next_table]
            redunant_tab_ids=[link,target]
            link_training_set,target_training_set=multi_table_join.training_set_tables[link],multi_table_join.training_set_tables[target]

            if idea_link_depth!=best_depth_tables[link]:
                # plan A: keep the original depth for link
                # solve the conflict for link table
                last_cost_A=multi_table_join.store_query_cost_dict[last_table][link][best_depth_tables[last_table]][best_depth_tables[link]]
                # seek best target depth for stored link depth
                target_depth_A,cur_cost_A=multi_table_join.compute_best_depth_b_given_a(link,target,best_depth_tables[link])
                red_cost1=pa_A_list[best_depth_tables[link]-join_base_depth].partition_tree.evaluate_query_cost(link_training_set, True) * len(link_training_set)
                red_cost2=pa_B_list[target_depth_A-join_base_depth].partition_tree.evaluate_query_cost(target_training_set, True) * len(target_training_set)
                if target_depth_A!=next_idea_target_depth:
                    target_depth_A,next_cost_A=multi_table_join.compute_best_depth_b_given_a(target,next_table,target_depth_A)
                else:
                    next_cost_A=multi_table_join.store_query_cost_dict[target][next_table][next_idea_target_depth][next_idea_next_depth]  
                plan_A_cost=last_cost_A+cur_cost_A+next_cost_A-red_cost1-red_cost2

                # plan B: keep the original depth for target
                last_cost_B=multi_table_join.compute_query_cost_between_two_depth(pa_last_list[best_depth_tables[last_table]-join_base_depth],pa_A_list[idea_link_depth-join_base_depth],last_table,link)
                cur_cost_B=multi_table_join.store_query_cost_dict[link][target][idea_link_depth][idea_target_depth]
                red_cost1=pa_A_list[idea_link_depth-join_base_depth].partition_tree.evaluate_query_cost(link_training_set, True) * len(link_training_set)
                red_cost2=pa_B_list[idea_target_depth-join_base_depth].partition_tree.evaluate_query_cost(target_training_set, True) * len(target_training_set)
                if idea_target_depth!=next_idea_target_depth:
                    target_depth_B,next_cost_B=multi_table_join.compute_best_depth_b_given_a(target,next_table,idea_target_depth)
                    
                else:
                    next_cost_B=multi_table_join.store_query_cost_dict[target][next_table][next_idea_target_depth][next_idea_next_depth]
                
                plan_B_cost=last_cost_B+cur_cost_B+next_cost_B-red_cost1-red_cost2

                print(f"planA:{plan_A_cost},  planB:{plan_B_cost}")
                if plan_A_cost>plan_B_cost:
                    best_depth_tables[link]=idea_link_depth
                    best_depth_tables[target]=idea_target_depth
                else:
                    best_depth_tables[target]=target_depth_A

            else:
                best_depth_tables[target]=idea_target_depth
        # best_depth_tables[target] = target_best_depth
        # if best_depth_tables[link]==0:
        #     best_depth_tables[link]=link_best_depth
        # elif best_depth_tables[link]!=link_best_depth:
        #     # best_depth_tables[target] = fixed_target_depth
        #     # break
        #     print("START COMPARING...")
        #     # there is a conflict about link depth
        #     # plan A: keep the original depth for link
            
        #     plan_a_cost = multi_table_join.store_query_cost_dict[last_table][link][best_depth_tables[last_table]][best_depth_tables[link]] + \
        #                     multi_table_join.compute_query_cost_between_two_depth(pa_A_list[fixed_link_depth - join_base_depth],pa_B_list[fixed_target_depth - join_base_depth], link, target, is_ignore=0)

        #     # plan B: keep the original depth for target
        #     plan_b_cost=multi_table_join.compute_query_cost_between_two_depth(pa_last_list[best_depth_tables[last_table]-join_base_depth],pa_A_list[link_best_depth-join_base_depth],last_table,link,is_ignore=1)+\
        #                 query_cost
        #     print(f"planA:{plan_a_cost},  planB:{plan_b_cost}")
        #     if plan_a_cost>plan_b_cost:
        #         print("update.")
        #         save_cost_for_tables_dict[target]=plan_a_cost-plan_b_cost
        #         best_depth_tables[link] = link_best_depth
        #     else:
        #         best_depth_tables[target] = fixed_target_depth
        # break
        TORN_res=[0]
        for tid in best_depth_tables.keys():
            if tid<len(best_depth_tables.keys())-1:
                link_pa=multi_table_join.pa_dict[tid][best_depth_tables[tid]-join_base_depth]
                link_pb=multi_table_join.pa_dict[tid+1][best_depth_tables[tid+1]-join_base_depth]
                if tid==0:
                    TORN_res.append(TORN_res[-1]+multi_table_join.compute_query_cost_between_two_depth(link_pa,link_pb,tid,tid+1))
                else:
                    TORN_res.append(TORN_res[-1]+multi_table_join.compute_query_cost_between_two_depth(link_pa,link_pb,tid,tid+1,is_ignore=0))
                print(f"{tid+1} tables query cost {TORN_res[-1]}")
        for tid in best_depth_tables.keys():
            print(f"Table: {table_names[tid]}, depth:{best_depth_tables[tid]}")
        adaptdb_fixed_res,min_depth=multi_table_join.print_query_cost_for_adaptdb()
        print(f"fixed depth {min_depth} for adaptdb!!")
        if model_name=='adaptdb':
            adaptdb_improved_res=TORN_res
        elif model_name=='nora':
            nora_res=TORN_res
            nora_fixed_res, min_depth = multi_table_join.print_query_cost_for_nora()
            # print(f"fixed depth {min_depth} for nora!!")
        save_cost_dict_list.append(save_cost_for_tables_dict)

    print(adaptdb_fixed_res)
    # print(adaptdb_improved_res)
    print(nora_fixed_res)
    print(nora_res)
    print(save_cost_dict_list)
    ray.shutdown()
    # [0, 25520961.3, 43236493.400000006, 56774943.400000006, 66285075.60000001]
    # [0, 38024196.5, 64250493.1, 85057804.80000001, 100432841.60000001]
    if save_queryset:
        multi_table_join.save_queryset(suffix)
    if save_model:
        multi_table_join.save_tree_model(best_depth_tables,suffix)
        #----------save for adaptdb tree-------------
        multi_table_join.save_tree_model_for_adaptdb(suffix)



# Exp: we test the performance of join queries over stored AdaptDB layout and TORN layout
def test_save_compare_join_with_multitable(table_num=4):
    join_attr = 0
    scale_w = 1 / 10  # hyper cost only access the join attribute data.
    base_join_path='/home/liupengju/pycharmProjects/NORA_JOIN_SIMULATION/NORA_experiments/queryset/multi_join/'
    base_join_depth=2
    partition_base_path = '/home/liupengju/pycharmProjects/NORA_JOIN_SIMULATION/PartitionLayout/multi_join/'
    algo_names=['adaptdb','jnora']
    group_types=[1,3]
    final_query_cost_res = []
    stored_temp_cost_list=[]
    for i in range(2):
        temp_cost_list=[]
        name = algo_names[i]
        pa_list=[]
        # compute common query cost
        for tid in range(table_num):
            training_set = np.genfromtxt(f"{base_join_path}{tid}_prob2_{scale_factor}_train.csv", delimiter=',')
            pa = PartitionAlgorithm()
            pa.partition_tree = PartitionTree(len(used_dims))
            pa.partition_tree.join_attr = 0
            pa.partition_tree.load_tree(f'{partition_base_path}prob2_{name}_{tid}_scale{scale_factor}')
            pa_list.append(pa)
            pa_cost = pa.partition_tree.evaluate_query_cost(training_set, True) * len(training_set)
            temp_cost_list.append(pa_cost)

        # compute join cost
        for link in range(table_num-1):
            target=link+1
            pa_A=pa_list[link-base_join_depth]
            pa_B=pa_list[target-base_join_depth]
            # pa_A = PartitionAlgorithm()
            # pa_A.partition_tree=PartitionTree(len(used_dims))
            # pa_A.partition_tree.join_attr=0
            # pa_A.partition_tree.load_tree(f'{partition_base_path}prob2_{name}_{link}_scale{scale_factor}')
            #
            # pa_B = PartitionAlgorithm()
            # pa_B.partition_tree = PartitionTree(len(used_dims))
            # pa_B.partition_tree.join_attr = 0
            # pa_B.partition_tree.load_tree(f'{partition_base_path}prob2_{name}_{target}_scale{scale_factor}')

            with open(f"{base_join_path}join_{link}_{target}_a_prob2_{scale_factor}_train", 'rb') as f:
                a_join_queries = pickle.load(f)
            with open(f"{base_join_path}join_{link}_{target}_b_prob2_{scale_factor}_train", 'rb') as f:
                b_join_queries = pickle.load(f)
            ju = JOIN_UNTIL(None, None, join_attr, len(used_dims))
            ju.set_partitioner(pa_A, pa_B)
            total_hyper_read_cost, total_shuffle_read_cost, hyper_cost_list = ju.compute_total_shuffle_hyper_cost(a_join_queries, b_join_queries, group_type=group_types[i])
            temp_cost_list.append(total_hyper_read_cost*scale_w)

        # print(temp_cost_list)
        stored_temp_cost_list.append(temp_cost_list)

        final_query_cost_res.append(sum(temp_cost_list))
    print(stored_temp_cost_list)
    print(final_query_cost_res)

# EXP: our proposed join depth determination algorithm for TORN
def determine_join_depth_with_shuffle_for_NORA():
    join_a_q_amount = 500
    join_b_q_amount = 300
    join_attr = 0
    dim_prob_filter_join = [0 if i==join_attr else 1 for i in range(len(used_dims))]
    a_training_set, _ = helper.generate_queryset_and_save(join_a_q_amount,dim_prob=dim_prob_filter_join, queryset_type=3)
    b_training_set, _ = helper.generate_queryset_and_save(join_b_q_amount,dim_prob=dim_prob_filter_join, queryset_type=3)
    a_training_set_for_join,_=helper.generate_queryset_and_save(join_a_q_amount, queryset_type=3)
    b_training_set_for_join,_=helper.generate_queryset_and_save(join_b_q_amount, queryset_type=3)
    join_depth_candidate = range(0, 8)
    cost_res = []
    ju = JOIN_UNTIL(a_training_set, b_training_set, join_attr, len(used_dims))
    a_join_queries, b_join_queries = ju.generate_join_queries(a_training_set_for_join,b_training_set_for_join)
    for i in range(0,1):
        cost_res.append({'sum1':[],'sum2':[],'detail':[],'hyper_detail':[]})
        # using_am_flag=True if i==0 else False
        using_am_flag=False
        a_best_depth=0
        best_depth_table=0
        a_best_hyper_blocks= float('inf')
        pa_A_list=[]
        pa_B_list=[]
        for join_depth in join_depth_candidate:
            # use my join tree base on nora
            total_cost_detail=[]
            pa_A = PartitionAlgorithm()
            pa_A.InitializeWithJNORA(a_training_set, len(boundary) // 2, boundary, dataset, data_threshold=block_size,
                                     join_attr=join_attr, using_kd=True, using_am=using_am_flag, candidate_size=2, candidate_depth=2,
                                     join_depth=join_depth)
            pa_A.partition_tree.name = "JNora_A"
            pa_A_list.append(pa_A)
            pa_B = PartitionAlgorithm()
            pa_B.InitializeWithJNORA(b_training_set, len(boundary) // 2, boundary, dataset, data_threshold=block_size,
                                     join_attr=join_attr, using_kd=True, using_am=using_am_flag, candidate_size=2, candidate_depth=2,
                                     join_depth=join_depth)
            pa_B.partition_tree.name = "JNora_B"
            pa_B_list.append(pa_B)
            # compute blocks for single-table queries
            pa_cost=pa_A.partition_tree.evaluate_query_cost(a_training_set, True)*join_a_q_amount
            pb_cost=pa_B.partition_tree.evaluate_query_cost(b_training_set, True)*join_b_q_amount
            total_cost_detail.append(pa_cost)
            total_cost_detail.append(pb_cost)
            # compute blocks for multi-table queries
            ju.set_partitioner(pa_A,pa_B)
            min_hyper_blocks_size, min_table = ju.compute_join_blocks_for_main_table(a_join_queries,b_join_queries)
            if min_hyper_blocks_size<a_best_hyper_blocks:
                a_best_hyper_blocks=min_hyper_blocks_size
                best_depth_table=min_table
                a_best_depth=join_depth
            total_hyper_read_cost, total_shuffle_read_cost, hyper_cost_list = ju.compute_total_shuffle_hyper_cost(a_join_queries, b_join_queries,group_type=3)
            total_cost_detail.append(total_hyper_read_cost)
            cost_res[i]['detail'].append(total_cost_detail)
            cost_res[i]['sum1'].append(sum(total_cost_detail))
            cost_res[i]['hyper_detail'].append(hyper_cost_list)
            # check the slope for cost_res curve
            if len(cost_res[i]['sum1'])>=4:
                if cost_res[i]['sum1'][-1] > cost_res[i]['sum1'][-2] > cost_res[i]['sum1'][-3]:
                    break
        # try different depth for paA and paB
        if best_depth_table==0:
            pa_A=pa_A_list[a_best_depth]
            b_best_depth = 0
            b_best_hyper_blocks = float('inf')
            for join_depth in join_depth_candidate:
                total_cost_detail = []
                if join_depth> len(pa_B_list)-1:
                    pa_B = PartitionAlgorithm()
                    pa_B.InitializeWithJNORA(b_training_set, len(boundary) // 2, boundary, dataset,
                                             data_threshold=block_size,
                                             join_attr=join_attr, using_kd=True, using_am=using_am_flag,
                                             candidate_size=2, candidate_depth=2,
                                             join_depth=join_depth)
                    pa_B.partition_tree.name = "JNora_B"
                else:
                    pa_B = pa_B_list[join_depth]
                # compute blocks for single-table queries
                pa_cost = pa_A.partition_tree.evaluate_query_cost(a_training_set, True) * join_a_q_amount
                pb_cost = pa_B.partition_tree.evaluate_query_cost(b_training_set, True) * join_b_q_amount
                total_cost_detail.append(pa_cost)
                total_cost_detail.append(pb_cost)
                # compute blocks for multi-table queries
                ju.set_partitioner(pa_A, pa_B)
                total_hyper_read_cost, total_shuffle_read_cost, hyper_cost_list = ju.compute_total_shuffle_hyper_cost(a_join_queries, b_join_queries,group_type=3)
                total_cost_detail.append(total_hyper_read_cost)
                cost_res[i]['detail'].append(total_cost_detail)
                cost_res[i]['sum2'].append(sum(total_cost_detail))
                cost_res[i]['hyper_detail'].append(hyper_cost_list)
                if sum(total_cost_detail)<b_best_hyper_blocks:
                    b_best_hyper_blocks=sum(total_cost_detail)
                    b_best_depth=join_depth
                if len(cost_res[i]['sum2']) >= 4:
                    if cost_res[i]['sum2'][-1] > cost_res[i]['sum2'][-2] > cost_res[i]['sum2'][-3]:
                        break

            print(f"a_depth:{a_best_depth},b_depth:{b_best_depth}")
            print(cost_res)
            colors = ['r', 'b']
            for i in range(len(cost_res)):
                plt.subplot(2,1,1)
                plt.xlabel("join depth both A,B")
                plt.ylabel("Total accessed blocks")
                y_bottom=0.7*min(cost_res[i]['sum1'])
                plt.plot(range(0,len(cost_res[i]['sum1'])), cost_res[i]['sum1'], 'o-', color=colors[i])
                # plt.ylim(bottom=0)

                plt.subplot(2, 1, 2)
                plt.xlabel(f"join depth for B (A_depth={a_best_depth})")
                plt.ylabel("Total accessed blocks")
                plt.plot(range(0,len(cost_res[i]['sum2'])), cost_res[i]['sum2'], 'o-', color=colors[i])
                plt.plot(b_best_depth, cost_res[i]['sum2'][b_best_depth], '*', color='b')
                # plt.ylim(bottom=y_bottom)
                plt.subplots_adjust(hspace=0.3)
            plt.savefig('/home/liupengju/pycharmProjects/NORA_JOIN_SIMULATION/images/jnora_analysis.png')


def compare_local_join():
    global pa_algos
    join_attr = 0
    dim_prob_filter_join = [0 if i == join_attr else 1 for i in range(len(used_dims))]
    join_query_amount=50
    training_set_for_join,_ = helper.generate_queryset_and_save(join_query_amount, queryset_type=3)
    training_set_no_join, _ = helper.generate_queryset_and_save(query_amount=500, queryset_type=3,dim_prob=dim_prob_filter_join)  # 1-1
    training_set=training_set_no_join+training_set_for_join
    #= = = = = Test PartitionAlgorithm () = = = = =  5.882  4.87
    pa1 = PartitionAlgorithm()
    pa1.InitializeWithQDT(training_set, len(boundary) // 2, boundary, dataset, data_threshold=block_size)
    pa1.partition_tree.name = "Instance-Optimized"
    # # = = = = = Test PartitionAlgorithm (PAW) = = = = = 12.98  8.13
    pa2 = PartitionAlgorithm()
    pa2.InitializeWithPAW(training_set_no_join, len(boundary) // 2, boundary, dataset, block_size, max_active_ratio=max_active_ratio,
                            strategy=1)
    pa2.partition_tree.name = "PAW"
    # # = = = = = Test PartitionAlgorithm (PAW+Beam search) = = = = = 12.98  8.13
    pa2_2 = PartitionAlgorithm()
    pa2_2.InitializeWithPAW(training_set, len(boundary) // 2, boundary, dataset, block_size,using_beam_search=True, candidate_size=2, candidate_depth=2,
                          max_active_ratio=max_active_ratio,
                          strategy=1)
    pa2_2.partition_tree.set_redundant_partition(queries=training_set, data_threshold=block_size)
    pa2_2.partition_tree.name = "PAW+beam-search"
    # = = = = = Test PartitionAlgorithm (NORA-Beam+Redundant) = = = = =
    pa3 = PartitionAlgorithm()
    pa3.InitializeWithNORA(training_set, len(boundary) // 2, boundary, dataset, data_threshold=block_size,
                           using_beam_search=True, candidate_size=2, candidate_depth=2, using_1_by_1=True,
                           using_kd=True)
    pa3.partition_tree.set_redundant_partition(queries=training_set, data_threshold=block_size)
    pa3.partition_tree.name = "NORA"

    pa_algos+=[pa1,pa2,pa2_2,pa3]

# visualize the performance shift of the different depth combinations for join tree
def compare_join_depth_without_shuffle():
    join_depth=5
    # helper.generate_dataset_and_save(base_path+'/dataset/lineitem_1.tbl')
    training_set, _ = helper.generate_queryset_and_save(500, queryset_type=3)  # 1-1
    # helper.visualize_queryset_and_dataset([0, 1], training_set, testing_set)
    # # = = = = = Test PartitionAlgorithm (NORA-Beam | AdaptDB | consider join layer<Median> ) = = = = =
    # pa8 = PartitionAlgorithm()
    # pa8.InitializeWithJNORA(training_set, len(boundary) // 2, boundary, dataset, data_threshold=block_size, join_attr=0,
    #                         using_kd=True, using_am=True, candidate_size=2, candidate_depth=2, join_depth=join_depth)
    # pa8.partition_tree.visualize(queries=training_set, add_text=False, use_sci=True)
    #
    # # = = = = = Test PartitionAlgorithm (NORA-Beam | NORA | consider join layer<Beam+skip> ) = = = = =
    # pa9=PartitionAlgorithm()
    # pa9.InitializeWithJNORA(training_set,len(boundary)//2,boundary,dataset,data_threshold=block_size,join_attr=0,using_kd=True,candidate_size=2,candidate_depth=2,join_depth=join_depth)
    # pa9.partition_tree.visualize(queries=training_set, add_text=False, use_sci=True)
    algos_for_joins=[[],[]]
    join_depth_candidate=range(1,8)
    for join_depth in join_depth_candidate:
        # = = = = = Test PartitionAlgorithm (NORA-AdaptDB) = = = = =
        pa11=PartitionAlgorithm()
        pa11.InitializeWithJNORA(training_set,len(boundary)//2,boundary,dataset,data_threshold=block_size,join_attr=0,using_kd=True,using_am=False,candidate_size=2,candidate_depth=2,join_depth=join_depth)
        pa11.partition_tree.name="JNora1"
        algos_for_joins[0].append(pa11)
        # = = = = = Test PartitionAlgorithm (NORA-based-skip Join) = = = = =
        pa12 = PartitionAlgorithm()
        pa12.InitializeWithJNORA(training_set, len(boundary) // 2, boundary, dataset, data_threshold=block_size,
                                 join_attr=0, using_kd=True, using_am=True, candidate_size=2, candidate_depth=2,
                                 join_depth=join_depth)
        pa12.partition_tree.name = "JNora2"
        pa12.partition_tree.visualize(queries=training_set, add_text=False, use_sci=True)
        algos_for_joins[1].append(pa12)
    query_count_res=[[],[]]
    cost_res = [[],[]]
    time_res=[[],[]]

    # simulate two queries join and compute the cost of shuffle and hyper join
    for aix,algo in enumerate(algos_for_joins):
        for model in algo:
            cost_res[aix].append(model.partition_tree.evaluate_query_cost(training_set, True))
            query_count_res[aix].append(model.query_count)
            time_res[aix].append(model.partition_tree.build_time)
    colors = ['r', 'b']
    for i in range(2):
        plt.subplot(1, 2, 1)
        plt.xlabel("join depth")
        plt.ylabel("Query count cross two blocks")
        plt.plot(join_depth_candidate, query_count_res[i], '*-', color=colors[i])
        plt.subplot(1, 2, 2)
        plt.xlabel("join depth")
        plt.ylabel("Total accessed blocks")
        plt.plot(join_depth_candidate, cost_res[i], 'o-', color=colors[i])
    plt.savefig('/home/liupengju/pycharmProjects/NORA_JOIN_SIMULATION/images/jnora_analysis.png')

# Input: data layout (LB-Cost); workload  Output: scan ratio
def print_exec_result(training_set):
    cost_res=[]
    time_res=[]
    methods=[]
    for pa in pa_algos:
        cost_res.append(pa.partition_tree.evaluate_query_cost(training_set, True))
        time_res.append(pa.partition_tree.build_time)
        methods.append(pa.partition_tree.name)
    result_sizes = helper.real_result_size(dataset, training_set)
    print("Total LB-Cost:", sum(result_sizes))
    cost_res.append(max(sum(result_sizes)/len(result_sizes), block_size))
    print(methods)
    print(np.round(np.array(cost_res)/dataset.shape[0],4))
    print(time_res)

def main():
    # test_save_compare_without_join()
    # compare_without_join(save=True)
    # determine_join_depth_with_shuffle_for_NORA()
    # compare_join_depth_with_shuffle()
    # test_save_compare_join_with_shuffle()
    # compare_join_depth_with_shuffle_with_ray()
    compare_hyper_join_with_multitable(5)
    # compare_hyper_join_with_multitable(5,save_model=False,save_queryset=False,suffix='tpcds/')
    # test_save_compare_join_with_multitable(5)
    # compare_local_join()
    pass
if __name__ == '__main__':
    main()