from src import reader_utils
import os
import sys

data_path = "data/atomic/"
filename = sys.argv[1]

with open(os.path.join(data_path, filename)) as f:
    data = f.readlines()

edge_dict = {}
for inst in data:
    inst = inst.strip()
    if inst:
        inst = inst.split('\t')
        src, rel, tgt = inst
        src = reader_utils.preprocess_atomic_sentence(src).replace("-", " ")
        tgt = reader_utils.preprocess_atomic_sentence(tgt).replace("-", " ")
        if src and tgt:
            if (src, rel) in edge_dict:
                edge_dict[(src, rel)].add(tgt)
            else:
                edge_dict[(src, rel)] = set([tgt])

out_lines = []
for k, v in edge_dict.items():
    if len(v) > 1 and "none" in v: 
        edge_dict[k].remove("none")
    out_lines.append([k[0] + "\t" + k[1] + "\t" + e2 + "\n" for e2 in edge_dict[k]])

out_lines = [line for sublist in out_lines for line in sublist]

with open(os.path.join(data_path, filename.replace(".txt", ".preprocessed.txt")), 'w') as f:
    f.writelines(out_lines)
