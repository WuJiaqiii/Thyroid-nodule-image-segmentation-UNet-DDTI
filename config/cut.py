#!/usr/bin/env python3
"""
Split a multi-document YAML config file into per-model-type directories,
renaming `model.type` to `model.model_type` and preserving original formatting.

Usage:
    python split_configs.py path/to/config.yaml

Requires:
    pip install ruamel.yaml
"""
import os
import sys
from ruamel.yaml import YAML

def split_config(input_file):
    yaml = YAML()
    # preserve as much of the original formatting and comments as possible
    yaml.preserve_quotes = True
    yaml.indent(mapping=4, sequence=4, offset=2)

    with open(input_file, 'r', encoding='utf-8') as f:
        docs = list(yaml.load_all(f))

    groups = {}
    for doc in docs:
        if not doc or 'model' not in doc:
            continue
        model = doc['model']
        mtype = model.get('type')
        if not mtype:
            continue
        # rename `type` to `model_type`
        model_type_value = model.pop('type')
        model['model_type'] = model_type_value
        # collect by type
        groups.setdefault(model_type_value, []).append(doc)

    # create directories and write out files
    for mtype, docs_list in groups.items():
        dir_name = mtype
        os.makedirs(dir_name, exist_ok=True)
        for idx, single_doc in enumerate(docs_list, start=1):
            output_path = os.path.join(dir_name, f"config{idx}.yaml")
            with open(output_path, 'w', encoding='utf-8') as out_f:
                yaml.dump(single_doc, out_f)
        print(f"Wrote {len(docs_list)} files to '{dir_name}'")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python split_configs.py <path_to_config.yaml>")
        sys.exit(1)
    split_config(sys.argv[1])
