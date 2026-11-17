"""Read explicit frozen S1/S2 artifacts, including public reproduction outputs."""
from pathlib import Path
import copy,json,hashlib
import numpy as np,torch
from .paths import coordinate,resolve,s1_spec,S2,DATA_ROOT
from studies.s3.cross_common import load,sha
from studies.corrected import native
from diagnostics.revision_evidence.pipeline import train_degrees
from baselines.dnr.backbone_adapter import FrozenBackboneAdapter,load_checkpoint

def coordinate_input(dataset,backbone,seed,fullpool=False):
    row=coordinate('s1',dataset,backbone,seed)
    key='fullpool_cache' if fullpool else 'top50_cache'
    if key not in row:key='benchmark_cache'
    path=resolve(row[key]);cache=load(path)
    cp_path=resolve(row.get('checkpoint',cache.get('backbone_identity',{}).get('checkpoint')))
    expected=cache.get('backbone_identity',{}).get('checkpoint_sha256',cache.get('checkpoint_sha256'))
    if expected is None:raise ValueError('S1 artifact must record its scorer checkpoint SHA256')
    actual_sha=sha(cp_path)
    if actual_sha!=expected:raise ValueError('S1 cache/checkpoint SHA differs')
    if row.get('checkpoint_sha256') and actual_sha!=row['checkpoint_sha256']:raise ValueError('S1 manifest/checkpoint SHA differs')
    spec=s1_spec(dataset,backbone,seed);config=load_checkpoint(cp_path)['config']
    if (str(config['model']),int(config['seed']),str(config['dataset']))!=(backbone,int(seed),str(spec['backbone_config']['dataset'])):
        raise ValueError('S1 checkpoint model/dataset/seed differs from the requested coordinate')
    if 'model_config' not in cache:
        # Public S1 benchmark_candidates.pt has exactly the same binary histories.
        history={int(u):set(map(int,v)) for u,v in cache['histories'].items()}
        user_num=int(config['user_num']) if 'user_num' in config and config['user_num'] else max(history,default=0)+1
        cache={**cache,'model_config':{'user_num':user_num,'item_num':len(cache['item_degree'])},'evaluation_histories':history}
    return row,cache,cp_path,spec,path

def scorer(cp_path,device,batch_size=32768):
    return FrozenBackboneAdapter(cp_path,device=torch.device(device),data_root=DATA_ROOT,sample_num=100,predict_batch_size=batch_size)

def neighbors(row,dataset,seed,fingerprint,k):
    path=resolve(row['native_neighbors'])
    return native.tables(path,fingerprint,k,'nonparamgc')

def s2_input(dataset,seed,adapter):
    row=coordinate('s2',dataset,'MF_BPR',seed,False) or {}
    bundle_path=resolve(row['split_bundle']) if row.get('split_bundle') else S2/'private/splits'/dataset/f'seed_{seed}.pt'
    selected=S2/'cf'/dataset/f'seed_{seed}'/'selected'
    score_path=resolve(row['test_scores']) if row.get('test_scores') else selected/'F_predictions.pt'
    bundle=load(bundle_path);saved=load(score_path)
    checks=dict(split_membership=native.verify_split_membership({p:native.phase_data(adapter,p) for p in ('train','valid','test')},bundle))
    for kind,field in [('user',adapter.uid_field),('item',adapter.iid_field)]:
        tokens=adapter.dataset.field2id_token[field]
        if 'field2id_token' in bundle:
            same=np.array_equal(tokens,bundle['field2id_token'][kind])
        else:
            same=hashlib.sha256(json.dumps(tokens.tolist(),sort_keys=True).encode()).hexdigest()==bundle['mapping_digests'][kind]
        if not same:raise ValueError('S1/S2 token mapping differs: '+kind)
    checks['mapping_exact']=True
    profile_path=resolve(row['selected_profile']) if row.get('selected_profile') else S2/'cf'/dataset/'selected_profile.json'
    if profile_path.is_file():profile=json.loads(profile_path.read_text())
    else:profile=json.loads((Path(__file__).resolve().parents[3]/'configs/s2'/dataset/f'seed_{seed}.json').read_text())['selected_profile']
    selected_result=resolve(row['selected_result']) if row.get('selected_result') else S2/'cf'/dataset/f'seed_{seed}'/f"lr_{profile['learning_rate']:g}_wd_{profile['weight_decay']:g}"/'result.json'
    result=json.loads(selected_result.read_text());retriever_cp=resolve(row.get('checkpoint') or result.get('checkpoint'))
    actual_sha=sha(retriever_cp)
    actual_profile=result['identity']['profile']
    if float(actual_profile['learning_rate'])!=float(profile['learning_rate']) or float(actual_profile['actual_weight_decay'])!=float(profile['weight_decay']):
        raise ValueError('S2 checkpoint differs from the recorded selected profile')
    selected_cp=load(retriever_cp)
    if selected_cp['identity']!=result['identity']:raise ValueError('S2 selected checkpoint/result identity differs')
    if int(result['identity']['seed'])!=int(seed):raise ValueError('S2 selected checkpoint seed differs')
    for key in ('split_digests','mapping_digests'):
        if result['identity'][key]!=bundle[key]:raise ValueError('S2 selected model/bundle identity differs: '+key)
    del selected_cp
    if 'identity' in saved:
        completion=resolve(row['complete']) if row.get('complete') else score_path.with_name('complete.json')
        parity_path=resolve(row['parity']) if row.get('parity') else score_path.with_name('scorer_parity.json')
        complete=json.loads(completion.read_text());parity=json.loads(parity_path.read_text())
        if saved['identity']!=complete['identity']:raise ValueError('S2 saved score completion identity differs')
        for key in ('split_digests','mapping_digests'):
            if saved['identity'][key]!=bundle[key]:raise ValueError('S2 saved split identity differs')
        if parity['status']!='PASS' or not parity['full_catalog_top20_equals_F_base_top20']:raise ValueError('S2 full catalog prefix check unavailable')
        if saved['identity']['cf_checkpoint_sha256']!=actual_sha:raise ValueError('S2 selected retriever SHA differs')
    else:
        # Public S2 output stores the F split directly. Require explicit byte
        # bindings instead of inventing an absent historic signature/parity test.
        if row.get('test_scores_sha256')!=sha(score_path) or row.get('checkpoint_sha256')!=actual_sha:
            raise ValueError('Public S2 F artifacts require explicit test_scores_sha256 and checkpoint_sha256 in the manifest')
        parity={'status':'NOT_RECORDED','full_catalog_top20_equals_F_base_top20':None,'current_input_byte_bindings_verified':True}
    if 'identity' not in saved:saved={**saved,'identity':{'cf_checkpoint_sha256':actual_sha}}
    return bundle,bundle_path,saved,score_path,retriever_cp,parity,checks
