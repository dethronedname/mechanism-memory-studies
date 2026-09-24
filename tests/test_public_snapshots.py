from pathlib import Path
import importlib.util,json,shutil,tempfile,unittest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('verify_reference',ROOT/'scripts/verify_reference.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
class PublicSnapshotTests(unittest.TestCase):
    def test_all_stage_imports_verified(self):
        self.assertEqual({r['study'] for r in v.verify()},{'pi4','c1','b1','t1'})
    def test_modified_model_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'reference').mkdir()
            shutil.copytree(ROOT/'reference/pi4',root/'reference/pi4')
            (root/'catalog').mkdir()
            shutil.copyfile(ROOT/'catalog/source_imports.json',root/'catalog/source_imports.json')
            p=root/'reference/pi4/pi2/models.py';p.write_bytes(p.read_bytes()+b'\n# unexpected\n')
            with self.assertRaises(ValueError):v.verify(root,'pi4')
    def test_manifest_escape_rejected(self):
        for name in ('../private','/absolute','a/../../escape'):
            with self.assertRaises(ValueError):v.safe_path(ROOT,name)
    def test_execution_repair_preserved(self):
        root=ROOT/'reference/pi4_t1_gpu02'
        original=(root/'provenance/ORIGINAL_T1_experiment.py.txt').read_text()
        expected=original.replace("    if str(device).startswith('cuda'):torch.cuda.reset_peak_memory_stats(device)\n",
            "    if str(device).startswith('cuda'):\n        torch.cuda.init()\n        torch.cuda.reset_peak_memory_stats(device)\n")
        self.assertNotEqual(original,expected)
        self.assertEqual((root/'tc1/experiment.py').read_text(),expected)
    def test_public_assets_are_present(self):
        catalog=json.loads((ROOT/'catalog/assets.json').read_text())
        for item in catalog['assets']:
            self.assertIn(item['availability'],{'available','external','missing'})
            self.assertIsNone(item['public_asset_url'])
            if item['availability']=='available':
                self.assertTrue((ROOT/item['source_location']['path']).is_dir())
                self.assertFalse(item['contains_weights'])
    def test_original_failed_run_inspection_retained(self):
        record=json.loads((ROOT/'catalog/t1_failed_run_inspection.json').read_text())
        self.assertEqual((record['attempts'],record['parent_errors']),(54,54))
        self.assertEqual(record['recorded_decision'],'INCONCLUSIVE')
        self.assertFalse(record['test_release_lock_present'])
if __name__=='__main__':unittest.main()
