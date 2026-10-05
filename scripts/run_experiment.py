"""Portable experiment launcher with commands, versions, and input hashes saved before training."""
import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
BALANCED='train_relcoh_10fold_cv_dementia_binary_topic80_20_balanced.py'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('experiment',choices=['balanced','tuned'])
    p.add_argument('--data',type=Path,default=Path('data/dataset/dimentia/all_test_parsed_filtered_inference_ready.jsonl'))
    p.add_argument('--checkpoint',type=Path,default=Path('data/result/gcdc/model/checkpoint_4/pytorch_model.bin'))
    p.add_argument('--output',type=Path,required=True,help='Use a new output directory for each run')
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--seed',type=int,default=106524)
    p.add_argument('--loss-weighting',choices=['balanced','none'],default='balanced')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    data=(ROOT/a.data).resolve(); checkpoint=(ROOT/a.checkpoint).resolve(); output=(ROOT/a.output).resolve()
    script=BALANCED if a.experiment=='balanced' else 'train_relcoh_binary_hyperparam_tuned.py'
    command=[sys.executable,str(ROOT/script),'--do_binary_train','--do_binary_test',
             '--binary_train_file',str(data),'--checkpoint_file',str(checkpoint),'--output_dir',str(output),
             '--dataset','dementia','--model_type','transformer_sent','--model_name_or_path','roberta-base',
             '--label_list','low,medium,high','--binary_label_list','control,patient',
             '--binary_num_train_epochs',str(a.epochs),'--seed',str(a.seed),'--binary_loss_weighting',a.loss_weighting]
    if a.experiment=='tuned':
        command+=['--do_binary_hyperparameter_tune','--binary_tune_learning_rates','1e-5,3e-5,1e-4,3e-4,1e-3']
    else:
        command+=['--binary_learning_rate','0.001','--binary_dev_ratio','0.125']
    if a.dry_run:
        print(json.dumps(command,indent=2)); return
    for path in (data,checkpoint):
        if not path.is_file(): raise SystemExit(f'Missing required asset: {path}')
        with path.open('rb') as f:
            if f.read(100).startswith(b'version https://git-lfs.github.com/spec'):
                raise SystemExit(f'LFS pointer found instead of data: {path}. Run git lfs pull.')
    if output.exists() and any(output.iterdir()):
        raise SystemExit('Output directory is not empty. Choose a new directory to avoid stale checkpoints.')
    output.mkdir(parents=True,exist_ok=True)
    versions={}
    for name in ['torch','numpy','transformers','tokenizers','scikit-learn','scipy','matplotlib','nltk','stanza','tqdm']:
        try: versions[name]=metadata.version(name)
        except metadata.PackageNotFoundError: versions[name]='not installed'
    revision=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True)
    manifest={'started_utc':datetime.now(timezone.utc).isoformat(),'command':command,
              'git_commit':revision.stdout.strip() if revision.returncode==0 else None,
              'python':platform.python_version(),'platform':platform.platform(),'packages':versions,
              'data_sha256':digest(data),'coherence_checkpoint_sha256':digest(checkpoint),
              'training_source_sha256':digest(ROOT/script),'model_source_sha256':digest(ROOT/'model.py'),
              'note':'Prospective run recipe, not the verified command behind the supplied historical CSV.'}
    target=output/'run_manifest.json'
    target.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    with (output/'console.log').open('w',encoding='utf-8') as log:
        result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    manifest['exit_code']=result.returncode
    manifest['finished_utc']=datetime.now(timezone.utc).isoformat()
    target.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(f'Run finished with exit code {result.returncode}. See {output / "console.log"}')
    raise SystemExit(result.returncode)


if __name__=='__main__': main()
