"""Summarize data and audit actual split files by participant (report sections 2 and 6)."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def load(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def label(r):
    v=str(r.get('source_group',r.get('source_label','unknown'))).lower().strip()
    return {'0':'control','1':'patient'}.get(v,v)


def participant(r):
    # Group prefix prevents accidental collisions between cohort-local IDs.
    v=str(r.get('patient_id','')).strip()
    return (label(r),v) if v else None


def summarize(path):
    rows=load(path)
    return {'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'records':len(rows),
            'labels':dict(Counter(map(label,rows))),
            'topics':dict(sorted(Counter(str(r.get('dialogue_topic','unknown')) for r in rows).items())),
            'participants_by_cohort_and_patient_id':len({participant(r) for r in rows if participant(r)}),
            'participants_by_patient_id_alone':len({str(r['patient_id']) for r in rows if r.get('patient_id')}),
            'missing_patient_id':sum(participant(r) is None for r in rows),
            'empty_sentences':sum(not r.get('sents') for r in rows),
            'duplicate_record_ids':len(rows)-len({r.get('id') for r in rows}),
            'score_values':dict(Counter(str(r.get('score')) for r in rows))}


def overlap(paths):
    groups=[{participant(r) for r in load(p) if participant(r)} for p in paths]
    return [{'left':paths[i].name,'right':paths[j].name,'shared_participants':len(groups[i]&groups[j])}
            for i in range(len(paths)) for j in range(i+1,len(paths))]


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files',type=Path,nargs='+',help='One dataset or train/validation/test JSONL files')
    parser.add_argument('--output',type=Path,default=Path('results/audits/dataset_audit.json'))
    parser.add_argument('--fail-on-overlap',action='store_true')
    args=parser.parse_args()
    result={'participant_key':'(normalized source_group, patient_id); verify IDs persist across visits',
            'files':[summarize(p) for p in args.files],'split_overlap':overlap(args.files)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
    if args.fail_on_overlap and (any(r['shared_participants'] for r in result['split_overlap']) or any(r['missing_patient_id'] for r in result['files'])):
        raise SystemExit(1)
