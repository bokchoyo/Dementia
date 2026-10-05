"""Recreate current default splits without loading models, then report participant overlap."""
import argparse
import ast
import json
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from audit_dataset import summarize, overlap

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT/'data/dataset/dimentia/all_test_parsed_filtered_inference_ready.jsonl')
    parser.add_argument('--output',type=Path,default=ROOT/'results/audits/split_audit.json')
    args=parser.parse_args()
    names={'load_json_records','save_jsonl_records','normalize_source_group','parse_label_list','infer_binary_label_from_record','prepare_binary_dementia_jsonl','get_dialogue_topic_from_record','summarize_binary_records','print_binary_topic_split_summary','split_binary_records_by_dialogue_topic','make_binary_train_dev_test_files','make_binary_8_1_1_tuning_files'}
    source=ROOT/'train_relcoh_binary_hyperparam_tuned.py'
    tree=ast.parse(source.read_text(encoding='utf-8'))
    # Execute only the named repository functions, avoiding Torch/model imports and main().
    selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    if {n.name for n in selected}!=names: raise RuntimeError('Training function names changed; update the audit explicitly.')
    ns=dict(json=json,os=os,random=random,np=np,KFold=KFold,StratifiedKFold=StratifiedKFold,train_test_split=train_test_split)
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(source),'exec'),ns)
    with tempfile.TemporaryDirectory() as tmp:
        a=SimpleNamespace(output_dir=tmp,binary_train_file=str(args.data.resolve()),test_file='',binary_test_ratio=.2,binary_topic_field='dialogue_topic',seed=106524,binary_dev_file='',binary_dev_ratio=0,binary_tune_folds=10,binary_tune_val_fold=9,binary_tune_test_fold=10)
        results={}
        for key,func in [('topic_80_20_default','make_binary_train_dev_test_files'),('fixed_8_1_1_default','make_binary_8_1_1_tuning_files')]:
            files=ns[func](a,['control','patient'])[:3]
            paths=[Path(p) for p in files if p]
            results[key]={'files':[summarize(p) for p in paths],'split_overlap':overlap(paths)}
        results['scope']='Recreated default splits from current source and supplied dataset; not verified historical result splits. Group key is (source_group, patient_id). No training performed.'
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({k:v['split_overlap'] for k,v in results.items() if isinstance(v,dict)},indent=2))


if __name__=='__main__': main()
