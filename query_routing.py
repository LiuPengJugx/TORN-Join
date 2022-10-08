import findspark
findspark.init() # this must be executed before the below import
from pyspark.sql import SparkSession
spark = SparkSession \
    .builder \
    .master("local") \
    .appName("Python Spark SQL Execution") \
    .config("spark.executor.memory", "8g") \
    .config("spark.driver.memory","8g") \
    .config("spark.memory.offHeap.enabled",True) \
    .config("spark.memory.offHeap.size","8g") \
    .getOrCreate()
import numpy as np
import time
import rtree
from rtree import index
from partition_tree import PartitionTree


def find_overlap_parquets(query, partition_index):
    '''
    find out all the overlap partition ids
    '''
    query_lower = [qr[0] for qr in query]
    query_upper = [qr[1] for qr in query]
    query_border = tuple(query_lower + query_upper)
    overlap_pids = list(partition_index.intersection(query_border))

    return overlap_pids


def transform_query_to_sql(query, used_dims, column_name_dict, hdfs_path, querytype=0, pids=None):
    sql = ''
    for i, dim in enumerate(used_dims):
        # if query[i][0] != -1:
        sql += column_name_dict[dim] + '>' + str(query[i]) + ' and '
        # if query[i][1] != -1:
        sql += column_name_dict[dim] + '<' + str(query[len(used_dims) + i]) + ' and '
    sql = sql[0:-4]  # remove the last 'and '
    print("pids:", pids)
    if pids is not None and len(pids) != 0:
        pids = str(set(pids)).replace(" ", "")  # '{1,2,3}'
        hdfs_path = hdfs_path + '/partition_' + pids + ".parquet"

    if querytype == 0:
        sql = "SELECT * FROM parquet.`" + hdfs_path + "`WHERE " + sql
    elif querytype == 1:
        sql = "SELECT COUNT(*) FROM parquet.`" + hdfs_path + "`WHERE " + sql
    elif querytype == 2:
        sql = "SELECT variance(_c0) FROM parquet.`" + hdfs_path + "`WHERE " + sql
    # else:
    # pids = str(set(pids)).replace(" ", "") # '{1,2,3}'
    # sql = "SELECT * FROM parquet.`" + hdfs_path + 'partition_' + pids + ".parquet` WHERE " + sql
    # sql = "SELECT COUNT(*) FROM parquet.`" + hdfs_path + 'partition_' + pids + ".parquet` WHERE " + sql
    # sql = "SELECT variance(_c0) FROM parquet.`" + hdfs_path + 'partition_' + pids + ".parquet` WHERE " + sql
    return sql


def query_with_parquets(query, used_dims, column_name_dict, hdfs_path, querytype=0, partition_tree=None,
                        print_execution_time=False):
    start_time = time.time()

    sql = None
    data_size=0
    if partition_tree == None:
        sql = transform_query_to_sql(query, used_dims, column_name_dict, hdfs_path, querytype)
    else:
        pids = partition_tree.query_single(query)  # find_overlap_parquets(query, rtree_idx)
        sql = transform_query_to_sql(query, used_dims, column_name_dict, hdfs_path, querytype, pids)
        data_size+=sum([partition_tree.nid_node_dict[pid].node_size for pid in pids])
        # print(sql)

    # print("generated sql:", sql)
    end_time_1 = time.time()

    query_result = spark.sql(sql).collect()
    #     query_result = spark.sql(sql) # lazy execution
    #     query_time = spark.time(spark.sql(sql).collect())  # there is no .time in pyspark

    end_time_2 = time.time()

    #     print("result size:", len(query_result))
    #     print("result content:", query_result)

    query_translation_time = end_time_1 - start_time
    query_execution_time = end_time_2 - end_time_1
    # print('query execution time: ', query_execution_time)

    if print_execution_time:
        print('query translation time: ', query_translation_time)
        print('query execution time: ', query_execution_time)

    # return (query_result, query_translation_time, query_execution_time) # this takes too much memory
    return (query_translation_time, query_execution_time, len(query_result),data_size)


def load_query(path):
    query_set = np.genfromtxt(path, delimiter=' ')
    # query_set = query_set.reshape(len(query_set),-1,2)
    return query_set


def kdnode_2_border(kdnode):
    lower = [domain[0] for domain in kdnode[0]]
    upper = [domain[1] for domain in kdnode[0]]
    border = tuple(lower + upper)  # non interleave
    return border


def load_partitions_from_file(path):
    '''
    the loaded stretched_kdnodes: [num_dims, l1,l2,...,ln, u1,u2,...,un, size, id, pid, left_child,id, right_child_id]
    '''
    stretched_kdnodes = np.genfromtxt(path, delimiter=',')
    num_dims = int(stretched_kdnodes[0, 0])
    kdnodes = []
    for i in range(len(stretched_kdnodes)):
        domains = [[stretched_kdnodes[i, k + 1], stretched_kdnodes[i, 1 + num_dims + k]] for k in range(num_dims)]
        row = [domains]
        row.append(stretched_kdnodes[i, 2 * num_dims + 1])
        # to be compatible with qd-tree's partition, that do not have the last 4 attributes
        if len(stretched_kdnodes[i]) > 2 * num_dims + 2:
            row.append(stretched_kdnodes[i, -4])
            row.append(stretched_kdnodes[i, -3])
            row.append(stretched_kdnodes[i, -2])
            row.append(stretched_kdnodes[i, -1])
        kdnodes.append(row)
    return kdnodes


# def prepare_partition_index(partition_path):
#     partitions = load_partitions_from_file(partition_path)

#     p = index.Property()
#     p.leaf_capacity = 32
#     p.index_capacity = 32
#     p.NearMinimumOverlaoFactor = 16
#     p.fill_factor = 0.8
#     p.overwrite = True
#     pidx = index.Index(properties = p)

#     partition_index = index.Index(properties = p)
#     for i in range(len(partitions)):
#         partition_index.insert(i, kdnode_2_border(partitions[i]))

#     return partition_index

def batch_query(queryset, used_dims, column_name_dict, hdfs_path, querytype=0, partition_path=""):
    #     rtree_idx = None
    #     if use_rtree_idx:
    #         rtree_idx = prepare_partition_index(partition_path)

    partition_tree = PartitionTree(len(used_dims))  # newly added
    partition_tree.load_tree(partition_path)

    start_time = time.time()

    # add statistics result
    results = []
    count = 0
    for i in range(0, len(queryset)):
        result = query_with_parquets(queryset[i], used_dims, column_name_dict, hdfs_path, querytype, partition_tree)
        print('finish query', count)
        count += 1
        results.append(result)
        # print("query:",queryset[i])
    #         if i == 0:
    #             break # just analysis top k queries
    end_time = time.time()

    result_size = 0
    block_size=0
    for result in results:
        result_size += result[2]
        block_size+=result[3]
    avg_result_size = int(result_size // len(queryset))
    avg_block_size = int(block_size // len(queryset))

    print('total query response time: ', end_time - start_time)
    print('average query response time: ', (end_time - start_time) / len(queryset))
    print('average result size: ', avg_result_size)
    print('average block size: ', avg_block_size)

if __name__ == '__main__':
    # ==== set environment parameters and generate dataset ====
    scale_factor = 100
    problem_type = 2
    query_path = '/home/liupengju/pycharmProjects/NORA_JOIN_SIMULATION/NORA_experiments/queryset/'
    # scale 100
    training_set = np.genfromtxt(query_path+"prob"+str(problem_type)+"_train.csv", delimiter=',')
    ## scale 50 and 10
    # training_set = np.genfromtxt(query_path+"prob"+str(problem_type)+"_train_scale"+str(scale_factor)+".csv", delimiter=',')
    used_dims = [1,2,3]
    num_dims = 16
    column_names = ['_c'+str(i) for i in range(num_dims)]
    column_name_dict = {}
    for i in range(num_dims):
        column_name_dict[i] = column_names[i]
    # scale 100
    hdfs_path_nora = 'hdfs://10.77.110.133:9001/par_nora/NORA/prob'+str(problem_type)+'/scale100/merged/'
    hdfs_path_qdtree ='hdfs://10.77.110.133:9001/par_nora/QdTree/prob'+str(problem_type)+'/scale100/merged/'
    hdfs_path_paw ='hdfs://10.77.110.133:9001/par_nora/PAW/prob'+str(problem_type)+'/scale100/merged/'
    # hdfs_path_kdtree = 'hdfs://192.168.6.62:9000/user/cloudray/KDTree/prob'+str(1)+'/merged/'

    # scale 50 and 10
    # hdfs_base_path = 'hdfs://192.168.6.62:9000/user/cloudray/'
    # hdfs_path_nora = hdfs_base_path + 'NORA/prob' + str(problem_type) + '/scale' + str(scale_factor) + "/merged/"
    # hdfs_path_qdtree = hdfs_base_path + 'QdTree/prob' + str(problem_type) + '/scale' + str(scale_factor) + "/merged/"
    # hdfs_path_kdtree = hdfs_base_path + 'KDTree/prob' + str(problem_type) + '/scale' + str(scale_factor) + "/merged/"

    # newly added
    querytype = 0 # 0: SELECT *;  2: SELECT variance(_c0)
    partition_base_path = '/home/liupengju/pycharmProjects/NORA_JOIN_SIMULATION/PartitionLayout/'
    # scale 100
    nora_partition_path = partition_base_path + 'prob' + str(problem_type) + '_nora_scale' + str(scale_factor)
    qdtree_partition_path = partition_base_path + 'prob' + str(problem_type) + '_qdtree_scale' + str(scale_factor)
    paw_partition_path = partition_base_path + 'prob' + str(problem_type) + '_paw_scale' + str(scale_factor)

    # scale 50 and 10 / 100
    # nora_partition_path = partition_base_path + 'prob' + str(problem_type) + '_nora_scale' + str(scale_factor)
    # qdtree_partition_path = partition_base_path + 'prob' + str(problem_type) + '_qdtree_scale' + str(scale_factor)
    # kdtree_partition_path = partition_base_path + 'prob' + str(problem_type) + '_kdtree_scale' + str(scale_factor)


    # NORA
    batch_query(training_set, used_dims, column_name_dict, hdfs_path_nora, querytype, nora_partition_path)
    # Qd-Tree
    batch_query(training_set, used_dims, column_name_dict, hdfs_path_qdtree, querytype, qdtree_partition_path)
    # PAW
    batch_query(training_set, used_dims, column_name_dict, hdfs_path_paw, querytype, paw_partition_path)
    # check number of row groups
    # import pyarrow as pa
    # fs = pa.fs.HadoopFileSystem('192.168.6.62', port=9000, user='hdfs', replication=1)
    # path1 = 'hdfs://192.168.6.62:9000/user/cloudray/NORA/prob1/merged/partition_94.parquet'
    # path2 = 'hdfs://192.168.6.62:9000/user/cloudray/KDTree/prob1/merged/partition_98.parquet'
    # fw1 = fs.open_input_file(path1)
    # meta1 = pa.parquet.read_metadata(fw1, memory_map=False)
    # print(meta1)
    # print(meta1.row_group(0))
    # fw1.close()
    #
    # fw2 = fs.open_input_file(path2)
    # meta2 = pa.parquet.read_metadata(fw2, memory_map=False)
    # print(meta2)
    # print(meta2.row_group(0))
    # fw2.close()


