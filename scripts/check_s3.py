#!/usr/bin/env python3
"""CPU-only S3 integrity, statistics, and import checks; no model construction."""
from pathlib import Path
import ast,importlib,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from common.runtime import bootstrap
bootstrap(True)
from studies.s3.cross_common import CONFIG,verify_configuration,identity
from studies.s3.s3_common import configure_runtime,applied_configuration
from studies.s3 import w2w3_statistics as stats
from studies.s3.w2w3_analysis import scopes_from_index
import numpy as np
from scipy.stats import kendalltau

def main():
    manifest=verify_configuration()
    for path in (ROOT/'src/studies/s3').glob('*.py'):ast.parse(path.read_text())
    modules=configure_runtime()
    assert modules['models'].RESIDUAL_SETTINGS['batch_size']==256
    assert stats.classify((.01,.02),(.03,.04),(.012,.018),(.032,.038),.0025)=='transfer'
    assert stats.classify((.01,.02),(-.04,-.03),(.012,.018),(-.038,-.032),.0025)=='reverse'
    assert stats.classify((-.003,.003),(-.003,.003),(-.001,.001),(-.0025,.0025),.0025)=='equivalent'
    assert stats.classify((-.003,.003),(.01,.02),(-.001,.001),(.012,.018),.0025)=='unresolved'
    assert stats.repeat_interval([1,2,3])['sample_sd_raw']==1
    assert stats.paired_tost([1,1],[1,1],.0025)['p_equivalence']is None
    assert abs(stats.holm_planned([.001],16)[0]-.016)<1e-14
    x=np.array([1,1,3,4]);y=np.array([1,2,2,4])
    assert abs(float(stats.tau_b(x,y))-kendalltau(x,y).statistic)<1e-14
    users,aligned,audit=stats.aligned_users({'U':(np.array([3,1,2]),np.array(['c','a','b']),np.array([[30],[10],[20]])),
        'F':(np.array([4,2,3]),np.array(['d','b','c']),np.array([[400],[200],[300]]))})
    np.testing.assert_array_equal(users,[2,3]);np.testing.assert_array_equal(aligned['U'][:,0],[20,30])
    assert audit['excluded_users']=={'U':1,'F':1}
    try:stats.aligned_users({'U':(np.array([1]),np.array(['a']),np.ones((1,1))),'F':(np.array([1]),np.array(['b']),np.ones((1,1)))})
    except ValueError:pass
    else:raise AssertionError('Mismatched full relevance must fail')
    matrix=np.array([[0.,1.],[1.,0.],[.5,.5]])
    a=stats.bootstrap_macros(matrix,np.arange(3),100,8702026,'toy')
    b=stats.bootstrap_macros(matrix,np.arange(3),100,8702026,'toy')
    np.testing.assert_array_equal(a,b);np.testing.assert_allclose(a.sum(1),1)
    try:scopes_from_index([])
    except ValueError:pass
    else:raise AssertionError('Incomplete five-seed R0 cohort must fail')
    profiles={}
    for d in ('yelp2018','ml-1m'):
      for b in ('NFM','MF_BPR'):
       for s in range(2020,2025):
        v=json.loads((CONFIG/'backbones'/d/b/f'seed_{s}.json').read_text())
        assert (v['dataset'],v['backbone'],v['training_seed'])==(d,b,s)
        assert v['profile']['train_batch_size']==4096
        if (d,b)in profiles:assert v['profile']==profiles[d,b]
        profiles[d,b]=v['profile']
    for path in (CONFIG/'applied').rglob('*.json'):
        d,b,seed,protocol=path.relative_to(CONFIG/'applied').parts
        for method in json.loads(path.read_text()):
            app=applied_configuration(d,b,int(seed.removeprefix('seed_')),protocol.removesuffix('.json'),method,
                {'signature':{'sha256':'dry-run-placeholder-no-artifact'}})
            modules['models'].normalized_applied(method,app)
    origins={name:str(Path(modules[name].__file__).relative_to(ROOT))for name in modules}
    print(json.dumps(dict(status='PASS',configuration_files=len(manifest['files']),backbone_settings=20,
        checks='configuration integrity, import closure, exact ties, pairing, bootstrap determinism, TOST, 16-family Holm, complete cohort',
        model_instantiated=False,training=False,inference=False,module_origins=origins),sort_keys=True))
if __name__=='__main__':main()
