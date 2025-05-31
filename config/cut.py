#!/usr/bin/env python3
import os, sys
from ruamel.yaml import YAML

def split_config(input_file):
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=4, sequence=4, offset=2)

    # 读取为一个列表
    with open(input_file, 'r', encoding='utf-8') as f:
        docs = yaml.load(f)  # 不再是 load_all

    groups = {}
    for doc in docs:
        # doc 应该长这样： {'model': {'model_type': 'VNet2D', 'kwargs': {...}}}
        model = doc.get('model', {})
        mtype = model.get('model_type')
        if not mtype:
            continue
        groups.setdefault(mtype, []).append(doc)

    for mtype, docs_list in groups.items():
        os.makedirs(mtype, exist_ok=True)
        for idx, single_doc in enumerate(docs_list, start=1):
            path = os.path.join(mtype, f"config{idx}.yaml")
            with open(path, 'w', encoding='utf-8') as out_f:
                yaml.dump(single_doc, out_f)
        print(f"Wrote {len(docs_list)} files to '{mtype}'")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python split_configs.py <config.yaml>")
        sys.exit(1)
    split_config(sys.argv[1])
