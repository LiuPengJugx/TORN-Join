import random
import time
class JOIN_UNTIL:

    def __init__(self,a_training_set, b_training_set, join_attr, dim_nums):
        self.a_training_set=a_training_set
        self.b_training_set=b_training_set
        self.join_attr=join_attr
        self.dim_nums=dim_nums

    def set_partitioner(self,pa_A,pa_B):
        self.pa_A=pa_A
        self.pa_B=pa_B

    def generate_join_queries(self,a_training_set_for_join,b_training_set_for_join,join_amount=20):
        join_attr=self.join_attr
        dim_nums=self.dim_nums
        # join_amount=20
        def __overlap(q1, q2, dim):
            if q1[dim] <= q2[dim] <= q1[dim + dim_nums] or q2[dim] <= q1[dim] <= q2[dim + dim_nums]:
                return True
            return False
        # a_training_set=self.a_training_set
        # b_training_set=self.b_training_set
        a_training_set=a_training_set_for_join
        b_training_set=b_training_set_for_join

        # pick join query which will be measure
        b_join_index = []
        for _ in range(join_amount):
            b_join_index.append(
                list(set([random.randint(0, len(b_training_set) - 1) for _ in range(random.randint(1, 10))])))
        # remove block id with overlap join attribute range
        b_join_queries = []
        for ids in b_join_index:
            item = []
            for idx in ids:
                flag = True
                for em in item:
                    if __overlap(b_training_set[idx], em, join_attr):
                        flag = False
                        break
                if flag: item.append(b_training_set[idx])
            b_join_queries.append(item)
        a_join_queries = {}
        for bid, item in enumerate(b_join_queries):
            for qb in item:
                a_join_queries[bid] = []
                for qa in a_training_set:
                    if __overlap(qa, qb, join_attr):
                        # remove overlap range queries
                        flag = True
                        for qa2 in a_join_queries[bid]:
                            # if __overlap(qa2,qa,join_attr):
                            if qa2 == qa:
                                flag = False
                                break
                        if flag: a_join_queries[bid].append(qa)
        # for key in a_join_queries.keys():
        #     print(f"{key} : {len(a_join_queries[key])}")
        return a_join_queries,b_join_queries

    def compute_total_shuffle_hyper_cost(self,a_join_queries,b_join_queries,group_type):
        pa_A=self.pa_A
        pa_B=self.pa_B
        dim_nums=self.dim_nums
        join_attr=self.join_attr
        blocks_a_ids = []
        blocks_b_ids = []
        a_join_info = []
        b_join_info = []
        # how to get join attr range base on block id.
        for key, queries in enumerate(b_join_queries):
            map_content = {}
            join_keys = []
            node_vals = []
            for query in queries:
                join_keys += pa_B.partition_tree.query_single_join(query)
                node_vals += pa_B.partition_tree.query_single(query)
            map_content[key] = list(set(node_vals))
            blocks_b_ids.append(map_content)

            join_keys = list(set(join_keys))
            join_info = {"nums": len(join_keys), "length": []}
            for join_id in join_keys:
                node = pa_B.partition_tree.nid_node_dict[join_id]
                join_info["length"].append(node.boundary[join_attr + node.num_dims] - node.boundary[join_attr])
            b_join_info.append(join_info)

        for key in a_join_queries:
            map_content = {}
            join_keys = []
            node_vals = []
            for query in a_join_queries[key]:
                join_keys += pa_A.partition_tree.query_single_join(query)
                node_vals += pa_A.partition_tree.query_single(query)
            map_content[key] = list(set(node_vals))
            blocks_a_ids.append(map_content)

            join_keys = list(set(join_keys))
            join_info = {"nums": len(join_keys), "length": []}
            for join_id in join_keys:
                node = pa_A.partition_tree.nid_node_dict[join_id]
                join_info["length"].append(node.boundary[join_attr + node.num_dims] - node.boundary[join_attr])
            a_join_info.append(join_info)

        # print(sum([len(group_ids[key]) for key, group_ids in enumerate(blocks_a_ids)]))
        # print(sum([len(group_ids[key]) for key, group_ids in enumerate(blocks_b_ids)]))

        join_infos = [a_join_info, b_join_info]
        for join_info in join_infos:
            total_nums, total_length = 0, 0
            for item in join_info:
                total_nums += item['nums']
                total_length += sum(item['length'])
            # print(f"{total_nums} , {round(total_length, 2)}")

        # compute hyper join cost （Use Group 4）
        def is_overlay(aid, bid):
            bucket_a = pa_A.partition_tree.nid_node_dict[aid].boundary
            bucket_b = pa_B.partition_tree.nid_node_dict[bid].boundary
            return __overlap(bucket_a, bucket_b, join_attr)
        def __overlap(q1, q2, dim):
            if q1[dim] <= q2[dim] <= q1[dim + dim_nums] or q2[dim] <= q1[dim] <= q2[dim + dim_nums]:
                return True
            return False
        final_resized_splits = []
        overlap_chunks_for_queries = []
        intersection_reward = 0
        total_hyper_cost = 0
        build_time = 0
        for idx in range(len(blocks_a_ids)):
            # if idx<=2:continue
            join_a_block_ids = []
            for key in blocks_a_ids[idx].keys():
                join_a_block_ids += blocks_a_ids[idx][key]
            join_b_block_ids = []
            for key in blocks_b_ids[idx].keys():
                join_b_block_ids += blocks_b_ids[idx][key]
            # group algorithm
            # step1: generate overlap_chunks
            overlap_chunks = {}
            for aid in join_a_block_ids:
                if aid not in overlap_chunks.keys(): overlap_chunks[aid] = []
                for bid in join_b_block_ids:
                    if is_overlay(aid, bid): overlap_chunks[aid].append(bid)
            # print(f"overlap chunks: ",overlap_chunks)
            overlap_chunks_for_queries.append(overlap_chunks)
            # step2: group
            # print(overlap_chunks)
            # print(join_a_block_ids)
            time0 = time.time()
            if group_type==3:
                resizedSplits = self.group3(overlap_chunks, join_a_block_ids, partition_size=8)
            elif group_type==1:
                resizedSplits = self.group1(overlap_chunks, join_a_block_ids, partition_size=8)

            build_time += time.time() - time0
            for group in resizedSplits:
                all_b_ids = []
                for a_id in group:
                    all_b_ids += overlap_chunks[a_id]
                    # print(overlap_chunks[a_id])
                actual_b_ids = list(set(all_b_ids))
                intersection_reward += len(all_b_ids) - len(actual_b_ids)
                total_hyper_cost += sum([pa_B.partition_tree.nid_node_dict[_].node_size for _ in actual_b_ids])
            # print(f"join-query#{idx}: {resizedSplits}")
            final_resized_splits.append(resizedSplits)
        # print("intersection_reward: ", intersection_reward)
        # print("total_hyper_cost: ", total_hyper_cost)
        # print("average build time: ", build_time / len(blocks_a_ids))

        total_hyper_read_cost = 0  # the dataset size loaded in memory
        total_hyper_block_num = 0
        total_shuffle_read_cost = 0
        total_reference_block_size = 0
        a_hyper_cost,b_hyper_cost=0,0
        shuffle_weight = 3
        for q_no, resizedSplits in enumerate(final_resized_splits):
            cnt = 0
            total_b_ids = []
            total_a_ids = []
            overlap_chunks = overlap_chunks_for_queries[q_no]
            a_hyper_for_q,b_hyper_for_q=0,0
            for group in resizedSplits:
                b_ids = []
                for a_id in group:
                    total_reference_block_size += pa_A.partition_tree.nid_node_dict[a_id].node_size
                    b_ids += overlap_chunks[a_id]
                    # total_a_ids.append(a_id)
                    cnt += 1
                total_a_ids+=group
                b_ids = list(set(b_ids))
                total_b_ids += b_ids
                total_hyper_block_num += len(b_ids)
                for b_id in b_ids:
                    total_hyper_read_cost += pa_B.partition_tree.nid_node_dict[b_id].node_size
                    b_hyper_cost+=pa_B.partition_tree.nid_node_dict[b_id].node_size
                    b_hyper_for_q+=pa_B.partition_tree.nid_node_dict[b_id].node_size
            # print(f"total a ids:{total_a_ids}")
            # print(f"total b ids:{total_b_ids}  {[pa_B.partition_tree.nid_node_dict[_].node_size for _ in total_b_ids]}")
            total_b_ids = list(set(total_b_ids))
            for b_id in total_b_ids:
                total_shuffle_read_cost += shuffle_weight * pa_B.partition_tree.nid_node_dict[b_id].node_size
            for a_id in total_a_ids:
                total_shuffle_read_cost += shuffle_weight * pa_A.partition_tree.nid_node_dict[a_id].node_size
                total_hyper_read_cost += pa_A.partition_tree.nid_node_dict[a_id].node_size
                a_hyper_cost+=pa_A.partition_tree.nid_node_dict[a_id].node_size
                a_hyper_for_q+=pa_A.partition_tree.nid_node_dict[a_id].node_size
            # print(f"query#{q_no}: a:{a_hyper_for_q} b:{b_hyper_for_q} sum:{a_hyper_for_q+b_hyper_for_q}")


        # print('total_hyper_read_cost:', total_hyper_read_cost)
        # print('total_shuffle_read_cost:', total_shuffle_read_cost)
        # print('total_reference_block_size:', total_reference_block_size)
        # print('total_hyper_block_num:', total_hyper_block_num)
        return total_hyper_read_cost,total_shuffle_read_cost,[a_hyper_cost,b_hyper_cost]

    def print_shuffle_hyper_blocks(self,a_join_queries,b_join_queries,group_type):
        pa_A=self.pa_A
        pa_B=self.pa_B
        dim_nums=self.dim_nums
        join_attr=self.join_attr
        blocks_a_ids = []
        blocks_b_ids = []
        a_join_info = []
        b_join_info = []
        # how to get join attr range base on block id.
        for key, queries in enumerate(b_join_queries):
            map_content = {}
            join_keys = []
            node_vals = []
            for query in queries:
                join_keys += pa_B.partition_tree.query_single_join(query)
                node_vals += pa_B.partition_tree.query_single(query)
            map_content[key] = list(set(node_vals))
            blocks_b_ids.append(map_content)

            join_keys = list(set(join_keys))
            join_info = {"nums": len(join_keys), "length": []}
            for join_id in join_keys:
                node = pa_B.partition_tree.nid_node_dict[join_id]
                join_info["length"].append(node.boundary[join_attr + node.num_dims] - node.boundary[join_attr])
            b_join_info.append(join_info)

        for key in a_join_queries:
            map_content = {}
            join_keys = []
            node_vals = []
            for query in a_join_queries[key]:
                join_keys += pa_A.partition_tree.query_single_join(query)
                node_vals += pa_A.partition_tree.query_single(query)
            map_content[key] = list(set(node_vals))
            blocks_a_ids.append(map_content)

            join_keys = list(set(join_keys))
            join_info = {"nums": len(join_keys), "length": []}
            for join_id in join_keys:
                node = pa_A.partition_tree.nid_node_dict[join_id]
                join_info["length"].append(node.boundary[join_attr + node.num_dims] - node.boundary[join_attr])
            a_join_info.append(join_info)

        join_infos = [a_join_info, b_join_info]
        for join_info in join_infos:
            total_nums, total_length = 0, 0
            for item in join_info:
                total_nums += item['nums']
                total_length += sum(item['length'])
            # print(f"{total_nums} , {round(total_length, 2)}")

        # compute hyper join cost （Use Group 4）
        def is_overlay(aid, bid):
            bucket_a = pa_A.partition_tree.nid_node_dict[aid].boundary
            bucket_b = pa_B.partition_tree.nid_node_dict[bid].boundary
            return __overlap(bucket_a, bucket_b, join_attr)
        def __overlap(q1, q2, dim):
            if q1[dim] <= q2[dim] <= q1[dim + dim_nums] or q2[dim] <= q1[dim] <= q2[dim + dim_nums]:
                return True
            return False
        final_resized_splits = []
        overlap_chunks_for_queries = []
        intersection_reward = 0
        total_hyper_cost = 0
        build_time = 0
        for idx in range(len(blocks_a_ids)):
            # if idx<=2:continue
            join_a_block_ids = []
            for key in blocks_a_ids[idx].keys():
                join_a_block_ids += blocks_a_ids[idx][key]
            join_b_block_ids = []
            for key in blocks_b_ids[idx].keys():
                join_b_block_ids += blocks_b_ids[idx][key]
            # group algorithm
            # step1: generate overlap_chunks
            overlap_chunks = {}
            for aid in join_a_block_ids:
                if aid not in overlap_chunks.keys(): overlap_chunks[aid] = []
                for bid in join_b_block_ids:
                    if is_overlay(aid, bid): overlap_chunks[aid].append(bid)
            # print(f"overlap chunks: ",overlap_chunks)
            overlap_chunks_for_queries.append(overlap_chunks)
            # step2: group
            # print(overlap_chunks)
            # print(join_a_block_ids)
            time0 = time.time()
            if group_type==3:
                resizedSplits = self.group3(overlap_chunks, join_a_block_ids, partition_size=8)
            elif group_type==1:
                resizedSplits = self.group1(overlap_chunks, join_a_block_ids, partition_size=8)

            build_time += time.time() - time0
            for group in resizedSplits:
                all_b_ids = []
                for a_id in group:
                    all_b_ids += overlap_chunks[a_id]
                    # print(overlap_chunks[a_id])
                actual_b_ids = list(set(all_b_ids))
                intersection_reward += len(all_b_ids) - len(actual_b_ids)
                total_hyper_cost += sum([pa_B.partition_tree.nid_node_dict[_].node_size for _ in actual_b_ids])
            final_resized_splits.append(resizedSplits)
        # print("total_hyper_cost: ", total_hyper_cost)
        # print("average build time: ", build_time / len(blocks_a_ids))
        a_ids_for_q, b_ids_for_q = [], []
        for q_no, resizedSplits in enumerate(final_resized_splits):
            cnt = 0
            total_b_ids = []
            total_a_ids = []
            overlap_chunks = overlap_chunks_for_queries[q_no]
            for group in resizedSplits:
                b_ids = []
                for a_id in group:
                    b_ids += overlap_chunks[a_id]
                    cnt += 1
                total_a_ids += group
                b_ids = list(set(b_ids))
                if b_ids:
                    total_b_ids.append(b_ids)
            a_ids_for_q.append(total_a_ids)
            b_ids_for_q.append(total_b_ids)
        return a_ids_for_q, b_ids_for_q

    def compute_join_blocks_for_main_table(self,a_join_queries,b_join_queries):
        pa_A=self.pa_A
        pa_B=self.pa_B
        # dim_nums=self.dim_nums
        join_attr=self.join_attr
        blocks_a_ids = []
        blocks_b_ids = []
        a_join_info = []
        b_join_info = []
        # how to get join attr range base on block id.
        for key, queries in enumerate(b_join_queries):
            map_content = {}
            join_keys = []
            node_vals = []
            for query in queries:
                join_keys += pa_B.partition_tree.query_single_join(query)
                node_vals += pa_B.partition_tree.query_single(query)
            map_content[key] = list(set(node_vals))
            blocks_b_ids.append(map_content)

            join_keys = list(set(join_keys))
            join_info = {"nums": len(join_keys), "length": []}
            for join_id in join_keys:
                node = pa_B.partition_tree.nid_node_dict[join_id]
                join_info["length"].append(node.boundary[join_attr + node.num_dims] - node.boundary[join_attr])
            b_join_info.append(join_info)

        for key in a_join_queries:
            map_content = {}
            join_keys = []
            node_vals = []
            for query in a_join_queries[key]:
                join_keys += pa_A.partition_tree.query_single_join(query)
                node_vals += pa_A.partition_tree.query_single(query)
            map_content[key] = list(set(node_vals))
            blocks_a_ids.append(map_content)

            join_keys = list(set(join_keys))
            join_info = {"nums": len(join_keys), "length": []}
            for join_id in join_keys:
                node = pa_A.partition_tree.nid_node_dict[join_id]
                join_info["length"].append(node.boundary[join_attr + node.num_dims] - node.boundary[join_attr])
            a_join_info.append(join_info)

        # print(sum([len(group_ids[key]) for key, group_ids in enumerate(blocks_a_ids)]))
        # print(sum([len(group_ids[key]) for key, group_ids in enumerate(blocks_b_ids)]))

        join_infos = [a_join_info, b_join_info]
        for join_info in join_infos:
            total_nums, total_length = 0, 0
            for item in join_info:
                total_nums += item['nums']
                total_length += sum(item['length'])
            # print(f"{total_nums} , {round(total_length, 2)}")
        # print(blocks_a_ids)
        # print(blocks_b_ids)
        a_hyper_blocks_size=0
        b_hyper_blocks_size=0
        for idx in range(len(blocks_a_ids)):
            # if idx<=2:continue
            for key in blocks_a_ids[idx].keys():
                a_hyper_blocks_size+=sum([pa_A.partition_tree.nid_node_dict[ida].node_size for ida in blocks_a_ids[idx][key]])
            for key in blocks_b_ids[idx].keys():
                b_hyper_blocks_size+=sum([pa_B.partition_tree.nid_node_dict[idb].node_size for idb in blocks_b_ids[idx][key]])
        return a_hyper_blocks_size,0

    # (adaptDB group)
    def group1(self,overlap_chunks, join_a_block_ids, partition_size):
        def get_intersection_size_count(setValues, listValues):
            size = 0
            for lv in listValues:
                if lv in setValues: size += 1
            return size
        resizedSplits = []
        size = len(join_a_block_ids)
        while size > 0:
            cur_splits = []
            chunks = []
            # max block size limit for every split.
            splitAvailableSize = partition_size  # indicate the max B block in every partition, here B=2.
            while size > 0 and splitAvailableSize > 0:
                maxIntersection = -1
                best_offset = -1
                for offset, bid in enumerate(join_a_block_ids):
                    cur_intersection = get_intersection_size_count(chunks, overlap_chunks[bid])
                    if cur_intersection > maxIntersection:
                        maxIntersection = cur_intersection
                        best_offset = offset
                bucket_id = join_a_block_ids[best_offset]
                cur_splits.append(bucket_id)
                chunks += overlap_chunks[bucket_id]
                chunks = list(set(chunks))
                # for rhs in overlap_chunks[bucket_id]:
                #     chunks.append(rhs)
                join_a_block_ids.remove(bucket_id)
                # splitAvailableSize-=pa_A.partition_tree.nid_node_dict[bucket_id].node_size
                splitAvailableSize -= 1
                size -= 1
            resizedSplits.append(cur_splits)
        return resizedSplits

    # best (group algorithm)
    def group3(self,overlap_chunks, join_a_block_ids, partition_size):
        def list_solved_list(l1, l2):
            for item1 in l1:
                if item1 in l2:
                    return True
            return False

        def get_intersection_size_count(setValues, listValues):
            size = 0
            for lv in listValues:
                if lv in setValues: size += 1
            return size

        resizedSplits = []
        size = len(join_a_block_ids)
        # max block size limit for every split.
        splitAvailableSize = partition_size  # indicate the max B block in every partition, here B=2.
        affinity_tab = []
        pre_save_ids = []
        computed_ids_dict = {}
        for bid in join_a_block_ids: computed_ids_dict[bid] = {}
        a_block_len = len(join_a_block_ids)
        for no1 in range(a_block_len):
            bid1 = join_a_block_ids[no1]
            max_intersection = -1
            max_bid = []
            for exist_bid in computed_ids_dict[bid1].keys():
                cur_intersection = computed_ids_dict[bid1][exist_bid]
                if cur_intersection > max_intersection:
                    max_intersection = cur_intersection
                    max_bid = [exist_bid]
            for no2 in range(no1 + 1, a_block_len):
                bid2 = join_a_block_ids[no2]
                cur_intersection = get_intersection_size_count(overlap_chunks[bid1], overlap_chunks[bid2])
                computed_ids_dict[bid1][bid2] = cur_intersection
                computed_ids_dict[bid2][bid1] = cur_intersection
                if cur_intersection > max_intersection:
                    max_intersection = cur_intersection
                    max_bid = [bid2]
            if max_intersection == 0:
                pre_save_ids.append(bid1)
            else:
                affinity_tab.append({'item': [[bid1], max_bid], 'val': max_intersection, 'chunk': overlap_chunks[bid1]})
            # sort computed_ids_dict for bid1
            # computed_ids_dict[bid1]=dict(sorted(computed_ids_dict[bid1].items(), key=lambda k: k[1],reverse=True))
        # print(computed_ids_dict)
        cur_index = 0
        # pre-save these ids which doesn't have any overlap blocks
        while cur_index < len(pre_save_ids):
            if cur_index + partition_size - 1 <= len(pre_save_ids) - 1:
                merge_ids = pre_save_ids[cur_index:cur_index + partition_size]
            else:
                merge_ids = pre_save_ids[cur_index:]
            resizedSplits.append(merge_ids)
            size -= len(merge_ids)
            cur_index += partition_size
        while size > 0:
            affinity_tab.sort(key=lambda item: (item['val'], len(item['item'][0])), reverse=True)
            # print(f"size: {size},   {affinity_tab}")
            sel_tab = affinity_tab.pop(0)
            # note that: because may be len(sel_tab['item'][1])>1, so the length of merge_ids may be > splitAvailableSize
            merge_ids = sel_tab['item'][0] + sel_tab['item'][1]
            merge_ids_length = len(merge_ids)
            is_completed = False
            if merge_ids_length == splitAvailableSize or len(affinity_tab) == 0 or sel_tab['val'] == -1:
                is_completed = True
                resizedSplits.append(merge_ids)
                size -= merge_ids_length
            else:
                # add key=chunk
                new_overlap_chunks = sel_tab['chunk']
                for bid in sel_tab['item'][1]:
                    new_overlap_chunks += overlap_chunks[bid]
                new_tab = {'item': [merge_ids, []], 'val': -1, 'chunk': list(set(new_overlap_chunks))}

            # update affinity_tab
            for tab in reversed(affinity_tab):
                # delete tab
                if list_solved_list(tab['item'][0], sel_tab['item'][1]):
                    affinity_tab.remove(tab)
                    continue
                # update tab
                if list_solved_list(tab['item'][1], merge_ids):
                    if is_completed or len(tab['item'][0]) + merge_ids_length > partition_size:
                        tab['item'][1] = []
                        tab['val'] = -1
                    else:
                        tab['item'][1] = merge_ids
                        tab['val'] = get_intersection_size_count(tab['chunk'], new_tab['chunk'])
            if not is_completed: affinity_tab.append(new_tab)
            # Case: the affinity_tab only has one item.
            if len(affinity_tab) == 1:
                last_tab = affinity_tab.pop(0)
                merge_ids = last_tab['item'][0] + last_tab['item'][1]
                resizedSplits.append(merge_ids)
                size -= len(merge_ids)
            # new round: these ids need to be updated
            for ud_item1 in affinity_tab:
                if ud_item1['val'] == -1:
                    ud1_key = ud_item1['item'][0]
                    flag1 = False
                    if len(ud1_key) == 1: flag1 = True
                    # if flag1:
                    #     single_target_ids=[next(iter(computed_ids_dict[ud1_key[0]]))]
                    #     single_max_intersection=computed_ids_dict[ud1_key[0]][single_target_ids[0]]
                    # print(f"{single_target_ids} --- {single_max_intersection}")
                    # overlap_chunks1=[]
                    # for bid in ud1_key: overlap_chunks1+=overlap_chunks[bid]
                    # overlap_chunks1=list(set(overlap_chunks1))

                    # if flag1:
                    #     overlap_chunks1=overlap_chunks[ud1_key[0]]
                    # else:
                    overlap_chunks1 = ud_item1['chunk']
                    min_allocate_length = splitAvailableSize - len(ud1_key)
                    max_intersection = -1
                    max_target_ids = []
                    for ud_item2 in affinity_tab:
                        ud2_key = ud_item2['item'][0]
                        if ud1_key == ud2_key: continue
                        if len(ud2_key) > min_allocate_length: continue
                        if ud_item2['item'][1] == ud1_key:
                            cur_intersection = ud_item2['val']
                        else:
                            # if flag1 and len(ud2_key)==1:continue
                            flag2 = False
                            if len(ud2_key) == 1: flag2 = True
                            if flag1 and flag2:
                                cur_intersection = computed_ids_dict[ud1_key[0]][ud2_key[0]]
                            else:
                                overlap_chunks2 = ud_item2['chunk']
                                # overlap_chunks2=[]
                                # for bid in ud2_key: overlap_chunks2+=overlap_chunks[bid]
                                # overlap_chunks2=list(set(overlap_chunks2))
                                cur_intersection = get_intersection_size_count(overlap_chunks1, overlap_chunks2)
                        if cur_intersection > max_intersection:
                            max_intersection = cur_intersection
                            max_target_ids = ud2_key
                    # if flag1:
                    #     if single_max_intersection>max_intersection:
                    #         max_intersection=single_max_intersection
                    #         max_target_ids=single_target_ids
                    ud_item1['val'] = max_intersection
                    ud_item1['item'][1] = max_target_ids
        return resizedSplits





