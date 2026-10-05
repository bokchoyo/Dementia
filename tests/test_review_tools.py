import importlib.util
from pathlib import Path
import tempfile
import unittest
import json

ROOT=Path(__file__).resolve().parents[1]
def module(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py')
    value=importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value
results=module('audit_results')
dataset=module('audit_dataset')

class ReviewToolsTests(unittest.TestCase):
    def test_overall_metrics_and_source_preservation(self):
        path=ROOT/'results/reported/final_results.csv'; original=path.read_bytes()
        result=results.audit(path)
        self.assertEqual(result['overall']['n'],445)
        self.assertAlmostEqual(result['overall']['accuracy'],242/445)
        self.assertTrue(result['topic_sum_matches_overall'])
        tea=next(x for x in result['issues'] if x['topic']=='Tea')
        self.assertAlmostEqual(tea['recomputed'],0.503968253968254)
        self.assertEqual(path.read_bytes(),original)

    def test_overlap_detects_same_participant_across_topics(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths=[Path(tmp)/f'{x}.jsonl' for x in ['train','test']]
            rows=[{'patient_id':'p1','source_group':'control','dialogue_topic':'Cat'},
                  {'patient_id':'p1','source_group':'control','dialogue_topic':'Tea'}]
            for p,r in zip(paths,rows): p.write_text(json.dumps(r)+'\n')
            self.assertEqual(dataset.overlap(paths)[0]['shared_participants'],1)
            rows[1]['source_group']='patient'; paths[1].write_text(json.dumps(rows[1])+'\n')
            self.assertEqual(dataset.overlap(paths)[0]['shared_participants'],0)

if __name__=='__main__': unittest.main()
