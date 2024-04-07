# -*- coding:utf-8 -*-
# @Author: wangjl
# @Time: 2023/1/4 9:40
# @File: create_mdocred_dataset.py
# @Email: wangjl.nju.2020@gmail.com.
import os
import json
import math
import numpy as np
from string import punctuation


entity_types = ["Location", "Person", "Object", "Organization"]

relation_types = [
    "instance_of_Object_Object",
    "symbol_of_Object_Object",
    "part_of_Object_Object",
    "similar_to_Object_Object",
    "working_on_Person_Object",
    "capital_of_Location_Location",
    "located_in_Location_Location",
    "located_in_Organization_Location",
    "close_to_Location_Location",
    "leader_of_Person_Person",
    "instance_of_Person_Person",
    "industry_peer_Person_Person",
    "relative_of_Person_Person",
    "agree_with_Person_Person",
    "disagree_with_Person_Person",
    "belong_to_Object_Person",
    "belongs_to_Object_Organization",
    "president_of_Person_Organization",
    "founder_of_Person_Organization",
    "member_of_Person_Organization",
    "leader_of_Person_Organization",
]

bi_direction_relations = [
    "similar_to_Object_Object",
    "close_to_Location_Location",
    "industry_peer_Person_Person",
    "relative_of_Person_Person",
    "agree_with_Person_Person",
    "disagree_with_Person_Person",
]


def read_text_label_config():
    # config_path = r"E:\NJUWork\mmre_dataset\text_label\annotation.conf"
    entity_type2id = {et: idx for idx, et in enumerate(entity_types)}
    id2entity_type = {v: k for k, v in entity_type2id.items()}
    relation_type2id = {r: idx for idx, r in enumerate(relation_types)}
    id2relation_type = {v: k for k, v in relation_type2id.items()}

    return entity_type2id, id2entity_type, relation_type2id, id2relation_type


def is_contain_chinese(words):
    for c in words:
        if "\u4e00" <= c <= "\u9fa5":
            return True

    return False


def read_corpus(text_path):
    with open(text_path, "r", encoding="utf-8") as fp:
        data = fp.readlines()

    content = ""
    en_split = False
    split_len = 0
    en_split_len = 0
    cn_split_len = 0
    for line in data:
        line = line.replace("\t", " ").replace("\n", " ")
        if not en_split and is_contain_chinese(line):
            en_split_len = split_len
            en_split = True

        if en_split and not is_contain_chinese(line):
            cn_split_len = split_len
            break

        content += line
        split_len += len(line)

    return en_split_len, content[:en_split_len], cn_split_len, content[en_split_len: cn_split_len]


def is_punctuation(c):
    return c in punctuation


def is_special_words(words):
    if len(words) >= 2 and words[-1] == "." and words[-2] == "Dr":
        return True

    return False


def resplit_word(en_content, en_len):
    idx2word_id = {}
    sents = []
    words = []

    last = 0
    for i in range(en_len + 1):
        # sent_id, word_id
        idx2word_id[i] = (len(sents), len(words))
        if i == en_len or en_content[i] == " " or is_punctuation(en_content[i]):
            if i == en_len or en_content[i] == " ":
                if i < en_len and en_content[i] == " " and is_punctuation(en_content[i - 1]):
                    last = i + 1
                    continue

                words.append(en_content[last: i])
            else:  # punctuation
                words.append(en_content[last: i])
                words.append(en_content[i])

            if (
                    (len(words) > 0 and len(words[-1]) > 0) and not is_special_words(words)
                    and (words[-1][-1] == "." or words[-1][-1] == "?" or words[-1][-1] == "!")
            ):
                sents.append(words)
                words = []

            last = i

    if len(words) > 0:
        sents.append(words)

    return sents, idx2word_id


def parse_text_ann(text_label_path):
    text_path = text_label_path.replace(".ann", ".txt")
    en_len, en_content, cn_len, cn_content = read_corpus(text_path)
    en_sents, idx2word_id = resplit_word(en_content, en_len)

    with open(text_label_path, "r", encoding="utf-8") as fp:
        label_datas = fp.readlines()

    # entities: main entity and auxiliary entity
    en_entity_map = {}
    cn_entity_map = {}
    for line in label_datas:
        if line.startswith("T"):
            splited = line.split()
            e_id, e_type, e_st, e_ed = splited[0], splited[1], splited[2], splited[3]
            e_c = " ".join(splited[4:]) if len(splited[4:]) > 1 else splited[4]
            e_st = int(e_st)
            e_ed = int(e_ed)
            if e_ed <= en_len:
                # 英文实体，需要分词
                sent_id, word_st_idx = idx2word_id[e_st]
                _, word_ed_idx = idx2word_id[e_ed]
                en_entity_map[e_id] = (e_type, word_st_idx, word_ed_idx + 1, sent_id, e_c)
            else:
                # 中文实体，不分词
                cn_entity_map[e_id] = (e_type, e_st - en_len, e_ed - en_len, 0, e_c)

    # relations
    en_main_entities, en_relation_map = get_main_entity(label_datas, en_entity_map)
    en_main_entities = sorted(en_main_entities, key=lambda x: en_entity_map[x][3] * 1000 + en_entity_map[x][1])
    cn_main_entities, cn_relation_map = get_main_entity(label_datas, cn_entity_map)
    cn_main_entities = sorted(cn_main_entities, key=lambda x: cn_entity_map[x][1])
    en_labels, en_vertex_set = construct_entity_label(label_datas, en_main_entities, en_entity_map, en_relation_map)
    cn_labels, cn_vertex_set = construct_entity_label(label_datas, cn_main_entities, cn_entity_map, cn_relation_map)

    # title = en_entity_map[en_main_entities[0]][-1]
    title = en_content[:50].replace(" ", "")
    sample = {
        "title": title,
        "labels": en_labels,
        "vertexSet": en_vertex_set,
        "sents": en_sents,
        "cn_labels": cn_labels,
        "cn_vertexSet": cn_vertex_set,
        "cn_sents": cn_content,
    }
    return sample


def construct_entity_label(label_datas, main_entities, entity_map, relation_map):
    entity2id = {me: idx for idx, me in enumerate(main_entities)}

    # add relation label
    labels = [
        {
            "r": r_type,
            "h": entity2id[e1],
            "t": entity2id[e2],
        } for r_type, e1, e2 in relation_map.values()
    ]

    # add main entity
    vertex_set = [
        [
            {
                "pos": [entity_map[me][1], entity_map[me][2]],
                "type": entity_map[me][0],
                "sent_id": entity_map[me][3],
                "name": entity_map[me][4],
            }
        ] for me in main_entities
    ]

    # add auxiliary entity
    for line in label_datas:
        if line.startswith("R"):
            r_id, r_type, arg1, arg2 = line.split()
            arg1 = arg1.replace("Arg1:", "")
            arg2 = arg2.replace("Arg2:", "")

            if r_type == "same_mention":
                if arg1 in main_entities and arg2 in main_entities:
                    print(f"### Warning: arg1: {arg1}, arg2: {arg2}")
                    continue

                if arg1 not in main_entities and arg2 not in main_entities:
                    continue

                if arg1 not in main_entities:
                    vertex_set[entity2id[arg2]].append(
                        {
                            "pos": [entity_map[arg1][1], entity_map[arg1][2]],
                            "type": entity_map[arg1][0],
                            "sent_id": entity_map[arg1][3],
                            "name": entity_map[arg1][4],
                        }
                    )
                elif arg2 not in main_entities:
                    vertex_set[entity2id[arg1]].append(
                        {
                            "pos": [entity_map[arg2][1], entity_map[arg2][2]],
                            "type": entity_map[arg2][0],
                            "sent_id": entity_map[arg2][3],
                            "name": entity_map[arg2][4],
                        }
                    )

    return labels, vertex_set


def get_main_entity(label_datas, entity_map):
    main_entities = set()
    relation_map = {}
    for line in label_datas:
        if line.startswith("R"):
            r_id, r_type, arg1, arg2 = line.split()
            arg1 = arg1.replace("Arg1:", "")
            arg2 = arg2.replace("Arg2:", "")
            if r_type != "same_mention" and arg1 in entity_map and arg2 in entity_map:
                main_entities.add(arg1)
                main_entities.add(arg2)
                r_type = r_type + "_" + entity_map[arg1][0] + "_" + entity_map[arg2][0]
                relation_map[r_id] = (r_type, arg1, arg2)
                if r_type in bi_direction_relations:
                    relation_map[r_id] = (r_type, arg2, arg1)

    return list(main_entities), relation_map


def get_video_text_label_path():
    video_label_path = r"E:\NJUWork\label_complete"
    authors = os.listdir(video_label_path)
    author_paths = [os.path.join(video_label_path, a) for a in authors]
    label_file_map = {ap: os.listdir(ap) for ap in author_paths}

    text_label_path = r"E:\NJUWork\mmre_dataset\text_label"

    video_label_dirs = []
    text_label_files = []
    for ap, lfs in label_file_map.items():
        text_label_dir = os.path.join(text_label_path, os.path.basename(ap))
        for lf in lfs:
            video_label_dir = os.path.join(ap, lf)
            if os.path.isdir(video_label_dir):
                text_label_file = os.path.join(text_label_dir, lf + ".ann")
                assert os.path.exists(text_label_file), f"{text_label_file} not exists."

                text_label_files.append(text_label_file)
                video_label_dirs.append(video_label_dir)

    return video_label_dirs, text_label_files


def parse_visual_label(video_label_dir):
    visual_anns = os.listdir(video_label_dir)
    visual_anns = filter(lambda x: x.endswith(".json"), visual_anns)
    return list(visual_anns)


def split_dataset(samples, dev_ratio=0.0, test_ratio=0.0, num_dev=None, num_test=None):
    num_data = len(samples)
    num_dev = int(num_data * dev_ratio) if num_dev is None else num_dev
    num_test = int(num_data * test_ratio) if num_test is None else num_test

    import numpy as np
    np.random.shuffle(samples)
    test_set = samples[:num_test]
    dev_set = samples[num_test: num_test + num_dev]
    train_set = samples[num_test + num_dev:]

    return train_set, dev_set, test_set


def write_dataset(data_path, samples):
    with open(data_path, "w") as fp:
        print(f"### write dataset to {data_path}")
        json.dump(samples, fp)


def divide_batch(sources, batch_num, is_shuffle=False):
    data_len = len(sources)
    batch_size = math.ceil(data_len / batch_num)

    idx_arr = list(range(data_len))
    if is_shuffle:
        np.random.shuffle(idx_arr)

    results = []
    for i in range(batch_num):
        cur_indices = idx_arr[i * batch_size: (i + 1) * batch_size]
        samples = [sources[idx] for idx in cur_indices]
        results.append(samples)
    return results


def main(dev_ratio, test_ratio):
    # en_len, en_content, cn_len, cn_content = read_corpus(r"E:\NJUWork\mmre_dataset\text_label\daijy\5729.txt")
    video_label_dirs, text_label_files = get_video_text_label_path()
    data_len = len(video_label_dirs)
    samples = []
    for idx, (video_label_dir, text_label_file) in enumerate(zip(video_label_dirs, text_label_files)):
        print(f"### Process {idx + 1} / {data_len}: {video_label_dir}")
        sample = parse_text_ann(text_label_file)
        sample["video_labels"] = parse_visual_label(video_label_dir)
        sample["visual_label_path"] = video_label_dir

        num_main_entities = len(sample["vertexSet"])
        num_visual_entities = len(sample["video_labels"])
        if num_main_entities != num_visual_entities:
            print(f"text-video annotations not match, {num_main_entities}-{num_visual_entities}-{video_label_dir}")
            if num_main_entities > num_visual_entities:
                last = sample["video_labels"][0]
                sample["video_labels"] = sample["video_labels"] + [last] * (num_main_entities - num_visual_entities)
            else:
                sample["video_labels"] = sample["video_labels"][:num_main_entities]

        samples.append(sample)

    all_json_path = r"D:\Dataset\MDocRED\all.json"
    with open(all_json_path, "w") as fp:
        print(f"### write dataset to {all_json_path}")
        json.dump(samples, fp)

    train_set, dev_set, test_set = split_dataset(samples, dev_ratio=dev_ratio, test_ratio=test_ratio)
    train_annotated_path = r"D:\Dataset\MDocRED\train_annotated.json"
    write_dataset(train_annotated_path, train_set)

    num_train_splits = 10
    divided_train_sets = divide_batch(train_set, num_train_splits, True)
    train_lens = [len(divided_train_set) for divided_train_set in divided_train_sets]
    train_len_path = r"D:\Dataset\MDocRED\train_lens.json"
    write_dataset(train_len_path, train_lens)

    train_data_dir = os.path.dirname(train_annotated_path)
    for idx, divided_train_set in enumerate(divided_train_sets):
        train_annotated_path = os.path.join(train_data_dir, "train_annotated" + str(idx) + ".json")
        write_dataset(train_annotated_path, divided_train_set)

    dev_path = r"D:\Dataset\MDocRED\dev.json"
    write_dataset(dev_path, dev_set)

    test_path = r"D:\Dataset\MDocRED\test.json"
    write_dataset(test_path, test_set)

    rel_info = {rel: rel for rel in relation_types}
    rel_info_path = r"D:\Dataset\MDocRED\rel_info.json"
    write_dataset(rel_info_path, rel_info)


if __name__ == "__main__":
    main(dev_ratio=0.2, test_ratio=0.0)
