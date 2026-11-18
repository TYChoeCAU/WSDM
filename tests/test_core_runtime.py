"""CPU import, configuration and dry-run checks; no fitting or inference."""
import importlib
import json
from pathlib import Path
import subprocess
import sys
import unittest
sys.dont_write_bytecode=True
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from common.runtime import bootstrap,restore_config

class CoreRuntimeTests(unittest.TestCase):
    def test_import_closure(self):
        bootstrap(True)
        names=['graph_rerank','baselines.dnr.data_adapter','baselines.dnr.backbone_adapter',
               'baselines.prm.trainer','diagnostics.graph_fusion.feature_cache','diagnostics.graph_fusion.train',
               'diagnostics.revision_evidence.pipeline','studies.corrected.native','studies.corrected.external',
               'studies.corrected.corrected_sampling','studies.s1.run','studies.s1.mostpop','studies.s2.cf_experiment',
               'studies.checks.run','studies.checks.a_equivalence','studies.checks.c_tables','studies.checks.c_partial_dependence']
        for name in names:
            with self.subTest(module=name):
                module=importlib.import_module(name)
                self.assertTrue(Path(module.__file__).resolve().is_relative_to(REPO/'src'))
        from graph_rerank import GenericPathReranker, GraphReranker
        self.assertTrue(issubclass(GenericPathReranker, GraphReranker))
        self.assertIn('GenericPathReranker', importlib.import_module('graph_rerank').__all__)
        self.assertNotIn('baselines.dnr.model',sys.modules)
        self.assertNotIn('baselines.dnr.trainer',sys.modules)

    def test_actual_s1_configurations(self):
        paths=sorted((REPO/'configs/s1').glob('*/*/*.json'))
        self.assertEqual(len(paths),54)
        for path in paths:
            saved=json.loads(path.read_text())
            with self.subTest(config=str(path.relative_to(REPO))):
                config=restore_config(saved['backbone_config'],REPO/'data',REPO/'runs/config-check','cpu')
                self.assertEqual(config['seed'],saved['seed'])
                self.assertEqual(config['model'],saved['backbone'])
                self.assertEqual(config['threshold'],{'rating':4})
                self.assertEqual(config['device'].type,'cpu')
                self.assertFalse(config['save_dataset'])
                self.assertFalse(config['save_dataloaders'])

    def test_diagnostic_plan(self):
        result=subprocess.run([sys.executable,str(REPO/'scripts/run_diagnostics.py'),'all','--dry-run'],cwd=REPO,capture_output=True,text=True,check=True)
        plan=json.loads(result.stdout)
        self.assertFalse(plan['training'])
        self.assertEqual(plan['equivalence_studies'],['S1_fresh_U','S3_W'])

    def test_dry_run_coordinate_counts(self):
        for study,expected in [('s1',54),('s2',6)]:
            result=subprocess.run([sys.executable,str(REPO/f'scripts/run_{study}.py'),'all','--dry-run'],cwd=REPO,capture_output=True,text=True,check=True)
            self.assertEqual(json.loads(result.stdout)['coordinates'],expected)

if __name__=='__main__':unittest.main()
