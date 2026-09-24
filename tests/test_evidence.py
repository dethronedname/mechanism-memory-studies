from pathlib import Path
import importlib.util
import json
import math
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('rebuild',ROOT/'scripts/rebuild_results.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=module.load_records(ROOT)
        cls.generated=module.generate(ROOT)

    def test_source_manifest(self):
        self.assertGreater(module.verify_sources(ROOT),100)

    def test_all_four_studies_retained(self):
        self.assertEqual(set(module.STAGES),{r['study'] for r in self.rows})

    def test_total_score_rows(self):
        self.assertEqual(len(self.rows),1512) # 135 PI4 + 405 C1 + 486 B1 + 486 T1

    def test_no_missing_methods_or_pairs(self):
        for stage in module.STAGES:
            expected=module.METHODS_5 if stage in ('pi4','c1') else module.METHODS_6
            for dataset in {r['dataset'] for r in self.rows if r['study']==stage}:
                ss=[r for r in self.rows if r['study']==stage and r['dataset']==dataset]
                self.assertEqual({r['method'] for r in ss},expected)
                self.assertEqual(len(ss),len(expected)*3*9)

    def test_null_boundary_not_zero(self):
        boundary=[r for r in self.rows if r['case']=='reset_null']
        self.assertTrue(boundary)
        self.assertTrue(all(r.get('difference_error') is None for r in boundary))

    def test_derived_files_are_exact(self):
        for rel,txt in self.generated.items():
            self.assertEqual((ROOT/rel).read_text(),txt,rel)

    def test_historical_validity_not_overwritten(self):
        statuses=json.loads(self.generated['results/derived/study_statuses.json'])
        row=next(r for r in statuses if r['study']=='pi4')
        self.assertEqual(row['scientific_numeric_status'],'LIMITED_ARCHITECTURE_SIGNAL')
        self.assertEqual(row['original_validity_status'],'INCONCLUSIVE')

    def test_c1_partial_replication_preserved(self):
        raw=json.loads((ROOT/'results/source_records/c1/run/DECISION.json').read_text())
        self.assertEqual(raw['status'],'PARTIAL_REPLICATION')
        self.assertEqual(sum(d['saved']['passed'] for d in raw['datasets']),1)

    def test_t1_no_uniform_dominance(self):
        raw=json.loads((ROOT/'results/source_records/t1/run/DECISION.json').read_text())
        self.assertEqual(raw['status'],'MIXED_TIME_BUDGET_TRADEOFF')
        for dataset in ('t01','t02','t03'):
            ss=[r for r in self.rows if r['study']=='t1' and r['dataset']==dataset]
            x=module.compare(ss,'gru32_behavior')
            self.assertGreater(x['mechanism_ratio'],1)
            self.assertGreater(x['ring4_ratio'],1)
            self.assertGreater(x['ring8_ratio'],1)

    def test_t1_long_horizon_claim_has_denominator(self):
        import csv,io
        rows=list(csv.DictReader(io.StringIO(self.generated['results/derived/t1_directional_counts.csv'])))
        gru=next(r for r in rows if r['baseline']=='gru32_behavior' and r['category']=='long')
        graph=next(r for r in rows if r['baseline']=='graph32_behavior' and r['category']=='long')
        self.assertEqual((gru['candidate_lower_count'],gru['n_paired_comparisons']),('7','9'))
        self.assertEqual((graph['candidate_lower_count'],graph['n_paired_comparisons']),('9','9'))

    def test_no_silent_undefined_ratio(self):
        for a,b in [(1,0),(None,1),(1,None),(float('nan'),1)]:
            with self.assertRaises(ValueError):module.ratio(a,b)

    def test_geomean_not_arithmetic_mean(self):
        self.assertAlmostEqual(module.geomean([.5,2]),1)
        for x in ([],[0,1],[-1,2],[float('inf')]):
            with self.assertRaises(ValueError):module.geomean(x)

    def test_source_row_pointers(self):
        for row in self.rows:
            index=int(row['source_pointer'].split('/')[-1])
            original=json.loads((ROOT/row['source_file']).read_text())['records'][index]
            self.assertEqual(original['method'],row['method'])
            self.assertEqual(original['seed'],row['seed'])

    def test_manifest_detects_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'results').mkdir();(p/'record.json').write_text('{}')
            (p/'results/evidence_manifest.json').write_text(json.dumps({'entries':[{'path':'record.json','size':2,'sha256':'0'*64}]}))
            with self.assertRaises(ValueError):module.verify_sources(p)

if __name__=='__main__':unittest.main()
