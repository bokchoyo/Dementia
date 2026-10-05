"""Audit the supplied aggregate results without modifying the source CSV (report section 5)."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def metrics(cc, cp, pc, pp):
    n = cc + cp + pc + pp
    f_control = 2 * cc / (2 * cc + cp + pc) if 2 * cc + cp + pc else 0
    f_patient = 2 * pp / (2 * pp + cp + pc) if 2 * pp + cp + pc else 0
    return dict(n=n, accuracy=(cc + pp) / n, macro_f1=(f_control + f_patient) / 2,
                control_recall=cc / (cc + cp), patient_recall=pp / (pc + pp))


def audit(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.reader(stream))
    overall = metrics(*[int(rows[i][j]) for i, j in [(2,1),(2,2),(3,1),(3,2)]])
    overall['confusion_matrix'] = [[int(rows[i][j]) for j in (1,2)] for i in (2,3)]
    overall['majority_control_accuracy'] = int(rows[2][3]) / overall['n']
    topics=[]
    issues=[]
    total=[0,0,0,0]
    for i in range(7,15):
        r=rows[i]
        topic=r[0]
        c, pred_c, p, pred_p=map(int,r[1:5])
        reported=dict(zip(['accuracy','macro_f1','control_recall','patient_recall'],map(float,r[5:9])))
        # Infer integer diagonal counts only when both rounded recalls and marginal counts agree.
        candidates=[]
        for cc in range(c+1):
            pc=pred_c-cc
            pp=p-pc
            if 0<=pc<=p and c-cc+pp==pred_p and abs(cc/c-reported['control_recall'])<=0.000501 and abs(pp/p-reported['patient_recall'])<=0.000501:
                candidates.append((cc,c-cc,pc,pp))
        entry={'topic':topic,'csv_row':i+1,'reported':reported,'n_control':c,'n_patient':p}
        if len(candidates)==1:
            cells=candidates[0]
            derived=metrics(*cells)
            entry.update(inferred_confusion_matrix=[list(cells[:2]),list(cells[2:])],recomputed=derived)
            total=[a+b for a,b in zip(total,cells)]
            for key,val in reported.items():
                if abs(val-derived[key])>0.000501:
                    issues.append({'topic':topic,'csv_row':i+1,'metric':key,'reported':val,'recomputed':derived[key]})
        else:
            issues.append({'topic':topic,'problem':'Cannot uniquely infer confusion matrix from rounded recalls and counts'})
        topics.append(entry)
    return {'source':'results/reported/final_results.csv','sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'method':'Recompute overall metrics from exact counts; infer topic matrices only when integer counts, margins, and rounded recalls uniquely agree. Preserve original values.',
            'overall':overall,'topics':topics,'topic_sum_matrix':[total[:2],total[2:]],
            'topic_sum_matches_overall':[total[:2],total[2:]]==overall['confusion_matrix'],'issues':issues}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path('results/reported/final_results.csv'))
    parser.add_argument('--output',type=Path,default=Path('results/audits/results_audit.json'))
    args=parser.parse_args()
    result=audit(args.input)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'overall':result['overall'],'issues':result['issues'],'topic_sum_matches_overall':result['topic_sum_matches_overall']},indent=2))
