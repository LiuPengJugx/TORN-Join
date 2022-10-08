import time
def get_intersection_size_count(setValues,listValues):
    size=0
    for lv in listValues:
        if lv in setValues: size+=1
    return size
def get_intersection_detail(setValues,listValues):
    size=0
    intersection=[]
    for lv in listValues:
        if lv in setValues:
            size+=1
            intersection.append(lv)
    return size,intersection
# improve the original group algorithm
def group4(overlap_chunks,join_a_block_ids,partition_size):
    def list_solved_list(l1,l2):
        for item1 in l1:
            if item1 in l2:
                return True
        return False
    resizedSplits=[]
    size=len(join_a_block_ids)
    # max block size limit for every split.
    splitAvailableSize = partition_size  # indicate the max B block in every partition, here B=2.
    affinity_tab=[]
    pre_save_ids=[]
    computed_ids_dict={}
    for bid1 in join_a_block_ids:
        max_intersection=-1
        max_bid=[]
        computed_ids_dict[bid1]={}
        for bid2 in join_a_block_ids:
            if bid1==bid2:continue
            cur_intersection=get_intersection_size_count(overlap_chunks[bid1],overlap_chunks[bid2])
            computed_ids_dict[bid1][bid2]=cur_intersection
            if cur_intersection>max_intersection:
                max_intersection=cur_intersection
                max_bid=[bid2]
        if max_intersection==0:
            pre_save_ids.append(bid1)
        else:
            affinity_tab.append({'item':[[bid1],max_bid],'val':max_intersection,'chunk':overlap_chunks[bid1]})
        #sort computed_ids_dict for bid1
        # computed_ids_dict[bid1]=dict(sorted(computed_ids_dict[bid1].items(), key=lambda k: k[1],reverse=True))
    # print(computed_ids_dict)
    cur_index=0
    # pre-save these ids which doesn't have any overlap blocks
    while cur_index<len(pre_save_ids):
        if cur_index+partition_size-1<=len(pre_save_ids)-1:
            merge_ids=pre_save_ids[cur_index:cur_index+partition_size]
        else:
            merge_ids=pre_save_ids[cur_index:]
        resizedSplits.append(merge_ids)
        size-=len(merge_ids)
        cur_index+=partition_size
    while size>0:
        affinity_tab.sort(key=lambda item: (item['val'],len(item['item'][0])), reverse=True)
        # print(f"size: {size},   {affinity_tab}")
        sel_tab=affinity_tab.pop(0)
        # note that: because may be len(sel_tab['item'][1])>1, so the length of merge_ids may be > splitAvailableSize
        merge_ids=sel_tab['item'][0]+sel_tab['item'][1]
        #update affinity_tab
        for tab in reversed(affinity_tab):
            # delete tab
            if list_solved_list(tab['item'][0],sel_tab['item'][1]):
                affinity_tab.remove(tab)
                continue
            # update tab
            if list_solved_list(tab['item'][1],merge_ids):
                # if len(tab['item'][0])==1 and len(tab['item'][1])==1:
                    # print(f"{tab['item'][0][0]} {tab['item'][1][0]}=>{computed_ids_dict[tab['item'][0][0]][tab['item'][1][0]]} is removed!!!!")
                    # computed_ids_dict[tab['item'][0][0]].pop(tab['item'][1][0])
                tab['item'][1]=[]
                tab['val']=-1

        if len(merge_ids)==splitAvailableSize or len(affinity_tab)==0 or sel_tab['val']==-1:
            resizedSplits.append(merge_ids)
            size-=len(merge_ids)
        else:
            #add key=chunk
            new_overlap_chunks=sel_tab['chunk']
            for bid in sel_tab['item'][1]:
                new_overlap_chunks+=overlap_chunks[bid]
            affinity_tab.append({'item':[merge_ids,[]],'val':-1,'chunk':list(set(new_overlap_chunks))})
        # Case: the affinity_tab only has one item.
        if len(affinity_tab)==1:
            last_tab=affinity_tab.pop(0)
            merge_ids=last_tab['item'][0]+last_tab['item'][1]
            resizedSplits.append(merge_ids)
            size-=len(merge_ids)
        # new round: these ids need to be updated
        for ud_item1 in affinity_tab:
            if ud_item1['val']==-1:
                ud1_key=ud_item1['item'][0]
                flag1=False
                if len(ud1_key)==1: flag1=True
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
                overlap_chunks1=ud_item1['chunk']
                min_allocate_length=splitAvailableSize-len(ud1_key)
                max_intersection=-1
                max_target_ids=[]
                for ud_item2 in affinity_tab:
                    ud2_key=ud_item2['item'][0]
                    if ud1_key==ud2_key:continue
                    if len(ud2_key)>min_allocate_length: continue
                    # if flag1 and len(ud2_key)==1:continue
                    flag2=False
                    if len(ud2_key)==1: flag2=True
                    if flag1 and flag2:
                        cur_intersection=computed_ids_dict[ud1_key[0]][ud2_key[0]]
                    else:
                        overlap_chunks2=ud_item2['chunk']
                        # overlap_chunks2=[]
                        # for bid in ud2_key: overlap_chunks2+=overlap_chunks[bid]
                        # overlap_chunks2=list(set(overlap_chunks2))
                        cur_intersection=get_intersection_size_count(overlap_chunks1,overlap_chunks2)
                    if cur_intersection>max_intersection:
                        max_intersection=cur_intersection
                        max_target_ids=ud2_key
                # if flag1:
                #     if single_max_intersection>max_intersection:
                #         max_intersection=single_max_intersection
                #         max_target_ids=single_target_ids
                ud_item1['val']=max_intersection
                ud_item1['item'][1]=max_target_ids
    return resizedSplits

# improve the original group algorithm
def group3(overlap_chunks,join_a_block_ids,partition_size):
    def list_solved_list(l1,l2):
        for item1 in l1:
            if item1 in l2:
                return True
        return False
    resizedSplits=[]
    size=len(join_a_block_ids)
    # max block size limit for every split.
    splitAvailableSize = partition_size  # indicate the max B block in every partition, here B=2.
    affinity_tab=[]
    pre_save_ids=[]
    computed_ids_dict={}
    for bid in join_a_block_ids: computed_ids_dict[bid]={}
    a_block_len=len(join_a_block_ids)
    for no1 in range(a_block_len):
        bid1=join_a_block_ids[no1]
        max_intersection=-1
        max_bid=[]
        for exist_bid in computed_ids_dict[bid1].keys():
            cur_intersection=computed_ids_dict[bid1][exist_bid]
            if cur_intersection>max_intersection:
                max_intersection=cur_intersection
                max_bid=[exist_bid]
        for no2 in range(no1+1,a_block_len):
            bid2=join_a_block_ids[no2]
            cur_intersection=get_intersection_size_count(overlap_chunks[bid1],overlap_chunks[bid2])
            computed_ids_dict[bid1][bid2]=cur_intersection
            computed_ids_dict[bid2][bid1]=cur_intersection
            if cur_intersection>max_intersection:
                max_intersection=cur_intersection
                max_bid=[bid2]
        if max_intersection==0:
            pre_save_ids.append(bid1)
        else:
            affinity_tab.append({'item':[[bid1],max_bid],'val':max_intersection,'chunk':overlap_chunks[bid1]})
        #sort computed_ids_dict for bid1
        # computed_ids_dict[bid1]=dict(sorted(computed_ids_dict[bid1].items(), key=lambda k: k[1],reverse=True))
    # print(computed_ids_dict)
    cur_index=0
    # pre-save these ids which doesn't have any overlap blocks
    while cur_index<len(pre_save_ids):
        if cur_index+partition_size-1<=len(pre_save_ids)-1:
            merge_ids=pre_save_ids[cur_index:cur_index+partition_size]
        else:
            merge_ids=pre_save_ids[cur_index:]
        resizedSplits.append(merge_ids)
        size-=len(merge_ids)
        cur_index+=partition_size
    while size>0:
        affinity_tab.sort(key=lambda item: (item['val'],len(item['item'][0])), reverse=True)
        print(f"size: {size},   {affinity_tab}")
        sel_tab=affinity_tab.pop(0)
        # note that: because may be len(sel_tab['item'][1])>1, so the length of merge_ids may be > splitAvailableSize
        merge_ids=sel_tab['item'][0]+sel_tab['item'][1]
        is_completed=False
        if len(merge_ids)==splitAvailableSize or len(affinity_tab)==0 or sel_tab['val']==-1:
            is_completed=True
            resizedSplits.append(merge_ids)
            size-=len(merge_ids)
        else:
            #add key=chunk
            new_overlap_chunks=sel_tab['chunk']
            for bid in sel_tab['item'][1]:
                new_overlap_chunks+=overlap_chunks[bid]
            new_tab={'item':[merge_ids,[]],'val':-1,'chunk':list(set(new_overlap_chunks))}
        #update affinity_tab
        for tab in reversed(affinity_tab):
            # delete tab
            if list_solved_list(tab['item'][0],sel_tab['item'][1]):
                affinity_tab.remove(tab)
                continue
            # update tab
            if list_solved_list(tab['item'][1],merge_ids):
                if is_completed or len(tab['item'][0])+len(merge_ids)>partition_size:
                    tab['item'][1]=[]
                    tab['val']=-1
                else:
                    tab['item'][1]=merge_ids
                    tab['val']=get_intersection_size_count(tab['chunk'],new_tab['chunk'])
        if not is_completed: affinity_tab.append(new_tab)
        # Case: the affinity_tab only has one item.
        if len(affinity_tab)==1:
            last_tab=affinity_tab.pop(0)
            merge_ids=last_tab['item'][0]+last_tab['item'][1]
            resizedSplits.append(merge_ids)
            size-=len(merge_ids)
        # new round: these ids need to be updated
        for ud_item1 in affinity_tab:
            if ud_item1['val']==-1:
                ud1_key=ud_item1['item'][0]
                flag1=False
                if len(ud1_key)==1: flag1=True
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
                overlap_chunks1=ud_item1['chunk']
                min_allocate_length=splitAvailableSize-len(ud1_key)
                max_intersection=-1
                max_target_ids=[]
                for ud_item2 in affinity_tab:
                    ud2_key=ud_item2['item'][0]
                    if ud1_key==ud2_key:continue
                    if len(ud2_key)>min_allocate_length: continue
                    if ud_item2['item'][1]==ud1_key:
                        cur_intersection=ud_item2['val']
                    else:
                        # if flag1 and len(ud2_key)==1:continue
                        flag2=False
                        if len(ud2_key)==1: flag2=True
                        if flag1 and flag2:
                            cur_intersection=computed_ids_dict[ud1_key[0]][ud2_key[0]]
                        else:
                            overlap_chunks2=ud_item2['chunk']
                            # overlap_chunks2=[]
                            # for bid in ud2_key: overlap_chunks2+=overlap_chunks[bid]
                            # overlap_chunks2=list(set(overlap_chunks2))
                            cur_intersection=get_intersection_size_count(overlap_chunks1,overlap_chunks2)
                    if cur_intersection>max_intersection:
                        max_intersection=cur_intersection
                        max_target_ids=ud2_key
                # if flag1:
                #     if single_max_intersection>max_intersection:
                #         max_intersection=single_max_intersection
                #         max_target_ids=single_target_ids
                ud_item1['val']=max_intersection
                ud_item1['item'][1]=max_target_ids
    return resizedSplits
# improve the original group algorithm, keep the time complex is O(n^2)
def group2(overlap_chunks,join_a_block_ids,partition_size):
    def list_solved_list(l1,l2):
        for item1 in l1:
            if item1 in l2:
                return True
        return False
    resizedSplits=[]
    size=len(join_a_block_ids)
    # max block size limit for every split.
    splitAvailableSize = partition_size  # indicate the max B block in every partition, here B=2.
    affinity_tab=[]
    pre_save_ids=[]
    computed_ids_dict={}
    time0=time.time()
    for bid in join_a_block_ids: computed_ids_dict[tuple([bid])]={}
    block_ids_length=len(join_a_block_ids)
    for no1 in range(block_ids_length):
        bid1=join_a_block_ids[no1]
        for no2 in range(no1+1,block_ids_length):
            bid2=join_a_block_ids[no2]
            cur_intersection,its_content=get_intersection_detail(overlap_chunks[bid1],overlap_chunks[bid2])
            computed_ids_dict[tuple([bid1])][tuple([bid2])]=[cur_intersection,its_content]
            computed_ids_dict[tuple([bid2])][tuple([bid1])]=[cur_intersection,its_content]
        computed_ids_dict[tuple([bid1])] = dict(sorted(computed_ids_dict[tuple([bid1])].items(), key=lambda k: k[1][0], reverse=True))
        computed_ids_dict[tuple([bid1])]["sorted_keys"] = list(computed_ids_dict[tuple([bid1])].keys())
        best_target_ids = computed_ids_dict[tuple([bid1])]['sorted_keys'][0]
        max_intersection=computed_ids_dict[tuple([bid1])][best_target_ids][0]
        if max_intersection==0:
            pre_save_ids.append(bid1)
        else:
            max_bid=list(best_target_ids)
            affinity_tab.append({'item':[[bid1],max_bid],'val':max_intersection})

    print("phase 1:",time.time()-time0)
    cur_index=0
    # pre-save these ids which doesn't have any overlap blocks
    while cur_index<len(pre_save_ids):
        if cur_index+partition_size-1<=len(pre_save_ids)-1:
            merge_ids=pre_save_ids[cur_index:cur_index+partition_size]
        else:
            merge_ids=pre_save_ids[cur_index:]
        resizedSplits.append(merge_ids)
        size-=len(merge_ids)
        cur_index+=partition_size
    phase2_time,phase3_time=0,0
    while size>0:
        time1 = time.time()
        affinity_tab.sort(key=lambda item: (item['val'],len(item['item'][0])), reverse=True)
        # print(f"size: {size},   {affinity_tab}")
        sel_tab=affinity_tab.pop(0)
        # note that: because may be len(sel_tab['item'][1])>1, so the length of merge_ids may be > splitAvailableSize
        merge_ids=sel_tab['item'][0]+sel_tab['item'][1]
        if sel_tab['val']!=-1:
            # delete combined ids from computed_ids_dict
            # computed_ids_dict.pop(tuple(sel_tab['item'][0]))
            # computed_ids_dict.pop(tuple(sel_tab['item'][1]))
            #update affinity_tab
            for tab in reversed(affinity_tab):
                # delete tab
                if list_solved_list(tab['item'][0],sel_tab['item'][1]):
                    affinity_tab.remove(tab)
                    continue
                # update tab
                else:
                    # delete the key which will be combined
                    link_a_id=tuple(tab['item'][0])
                    link_sort_keys=computed_ids_dict[link_a_id]['sorted_keys']
                    item_list=[]
                    is_top=False
                    for sel_item in sel_tab['item']:
                        target_a_id=tuple(sel_item)
                        if target_a_id in link_sort_keys:
                            if link_sort_keys.index(target_a_id)==0: is_top=True
                            link_sort_keys.remove(target_a_id)
                            item=computed_ids_dict[link_a_id].pop(target_a_id)
                            item_list.append(item[1])
                    if len(item_list)>=1:
                        # if: the length of combined key is too long:
                        min_allocate_length=splitAvailableSize-len(link_a_id)
                        if len(merge_ids)<=min_allocate_length:
                            # else: update the combined key
                            new_item=[]
                            for item in item_list: new_item+=item
                            merged_items=list(set(new_item))
                            merged_items_length=len(merged_items)
                            for kid,key in enumerate(link_sort_keys):
                                if merged_items_length>=computed_ids_dict[link_a_id][key][0]:
                                    link_sort_keys.insert(kid,tuple(merge_ids))
                                    break
                            computed_ids_dict[link_a_id][tuple(merge_ids)]=[merged_items_length,merged_items]
                    if is_top:
                    # if list_solved_list(tab['item'][1],merge_ids):
                        if len(link_sort_keys)==0:
                            tab['item'][1]=[]
                            tab['val'] =-1
                        else:
                            best_target_ids=link_sort_keys[0]
                            tab['item'][1]=list(best_target_ids)
                            tab['val']=computed_ids_dict[link_a_id][best_target_ids][0]
            phase2_time+=time.time()-time1

        if len(merge_ids)==splitAvailableSize or len(affinity_tab)==0 or sel_tab['val']==-1:
            resizedSplits.append(merge_ids)
            size-=len(merge_ids)
        else:
            time2 = time.time()
            new_combined_tab={'item':[merge_ids,[]],'val':-1}
            ud1_key=tuple(merge_ids)
            # overlap_chunks1=[]
            # for bid in ud1_key: overlap_chunks1+=overlap_chunks[bid]
            # overlap_chunks1=list(set(overlap_chunks1))
            min_allocate_length=splitAvailableSize-len(ud1_key)
            # max_intersection=-1
            # max_target_ids=[]
            computed_ids_dict[ud1_key]={}
            for ud_item2 in affinity_tab:
                ud2_key=tuple(ud_item2['item'][0])
                if len(ud2_key)>min_allocate_length: continue
                # overlap_chunks2=[]
                # for bid in ud2_key: overlap_chunks2+=overlap_chunks[bid]
                # overlap_chunks2=list(set(overlap_chunks2))
                # cur_intersection,its_content=get_intersection_detail(overlap_chunks1,overlap_chunks2)
                # computed_ids_dict[ud1_key][ud2_key]=[cur_intersection,its_content]
                # print(ud1_key,ud2_key)
                # if ud1_key==tuple([536, 539, 289, 546, 767, 533, 549]):
                #     print(1)
                computed_ids_dict[ud1_key][ud2_key]=computed_ids_dict[ud2_key][ud1_key]
                # if cur_intersection>max_intersection:
                #     max_intersection=cur_intersection
                #     max_target_ids=list(ud2_key)
            computed_ids_dict[ud1_key]=dict(sorted(computed_ids_dict[ud1_key].items(), key=lambda k: k[1][0],reverse=True))
            computed_ids_dict[ud1_key]["sorted_keys"]=list(computed_ids_dict[ud1_key].keys())
            if len(computed_ids_dict[ud1_key]["sorted_keys"])>0:
                best_target_ids = computed_ids_dict[ud1_key]['sorted_keys'][0]
                new_combined_tab['item'][1] = list(best_target_ids)
                new_combined_tab['val'] = computed_ids_dict[ud1_key][best_target_ids][0]
            affinity_tab.append(new_combined_tab)
            phase3_time += time.time() - time2
        # Case: the affinity_tab only has one item.
        # if len(affinity_tab)==1:
        #     last_tab=affinity_tab.pop(0)
        #     merge_ids=last_tab['item'][0]+last_tab['item'][1]
        #     resizedSplits.append(merge_ids)
        #     size-=len(merge_ids)
    print("phase 2:", phase2_time)
    print("phase 3:", phase3_time)
    return resizedSplits
# overlap_chunks={648: [222, 214], 649: [222, 214], 652: [222, 214], 655: [222, 214], 403: [222, 214], 660: [222, 214], 662: [222, 214], 408: [222, 214], 665: [222, 214], 669: [222, 214], 431: [222], 688: [222], 818: [222, 214], 819: [214], 820: [222, 214], 821: [214], 822: [222, 214], 695: [222], 824: [222, 214], 707: [222], 634: [515, 214, 346], 639: [222, 214]}
# join_a_block_ids=[648, 649, 652, 655, 403, 660, 662, 408, 665, 669, 431, 688, 818, 819, 820, 821, 822, 695, 824, 707, 634, 639]
# overlap_chunks={643: [577], 649: [577, 529, 530], 394: [515, 346, 349], 395: [515, 346, 349], 403: [577, 529, 530], 664: [577], 665: [577, 529, 530], 669: [577, 529, 530], 415: [577], 672: [577], 671: [577], 808: [515], 819: [577], 820: [577, 529, 530], 821: [577], 822: [529, 530], 824: [577, 529, 530], 605: [515, 346, 349], 612: [515, 346, 349], 623: [515], 634: [515, 346, 349], 639: [577, 529, 530]}
# join_a_block_ids=[643, 649, 394, 395, 403, 664, 665, 669, 415, 672, 671, 808, 819, 820, 821, 822, 824, 605, 612, 623, 634, 639]
# overlap_chunks={778: [450, 446, 287], 527: [450, 446, 287], 275: [417, 450, 446], 533: [311, 446, 287], 536: [450, 446, 287], 281: [417, 450, 446], 539: [450, 446, 287], 543: [450, 311, 446, 287], 289: [450, 446, 287], 546: [450, 446, 287], 549: [311, 446, 287], 552: [450, 311, 446, 287], 299: [450, 311, 446, 287], 301: [450, 311, 446, 287], 313: [450, 311, 446, 287], 314: [450, 311, 446, 287], 738: [417, 450, 446], 739: [417, 450, 446], 745: [417, 450, 446], 750: [417, 450, 446], 496: [450, 446], 753: [417, 450, 446], 758: [417, 450, 446], 760: [417, 450, 446], 762: [417, 450, 446], 763: [417, 450, 446], 764: [417, 450, 446], 767: [450, 446, 287]}
# join_a_block_ids=[778, 527, 275, 533, 536, 281, 539, 543, 289, 546, 549, 552, 299, 301, 313, 314, 738, 739, 745, 750, 496, 753, 758, 760, 762, 763, 764, 767]
overlap_chunks={519: [287, 271], 523: [287, 271], 524: [287, 271], 525: [271], 527: [271], 528: [287, 271], 529: [271], 531: [287, 271], 276: [271], 532: [287, 271], 279: [271], 535: [271], 539: [287, 271], 283: [287, 271], 299: [287, 271], 559: [290, 287], 305: [271], 306: [287, 271], 577: [290, 287, 435], 323: [290, 287], 711: [271], 716: [271], 720: [271], 1105: [271], 1106: [287, 271], 723: [271], 470: [271], 471: [271], 726: [287, 271], 733: [271], 736: [287, 271], 486: [271], 487: [271], 490: [271], 493: [271], 494: [271], 496: [271], 498: [271], 502: [271], 504: [271], 506: [287, 271], 508: [287, 271]}
join_a_block_ids=[519, 523, 524, 525, 527, 528, 529, 531, 276, 532, 279, 535, 539, 283, 299, 559, 305, 306, 577, 323, 711, 716, 720, 1105, 1106, 723, 470, 471, 726, 733, 736, 486, 487, 490, 493, 494, 496, 498, 502, 504, 506, 508]
resizedSplits=group3(overlap_chunks,join_a_block_ids,partition_size=8)
print(resizedSplits)

# phase 1: 0.0012600421905517578
# phase 2: 0.001665353775024414
# phase 3: 0.0006170272827148438

# [[549, 533, 314, 313, 301, 299, 543, 552], [758, 753, 750, 745, 739, 738, 275, 281], [753, 750, 745, 739, 738, 275, 281, 760], [753, 750, 745, 739, 738, 275, 281, 760]]
