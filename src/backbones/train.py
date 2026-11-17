"""Execute the original RecBole backbone trainer using actual saved settings."""
from pathlib import Path
from common.runtime import restore_config

def train_backbone(spec, data_root, output_root, device):
    from baselines.dnr.backbone_adapter import checkpoint_for
    from recbole.data import create_dataset, data_preparation
    from recbole.utils import init_seed, get_model, get_trainer
    dataset, model, seed = spec['dataset'], spec['backbone'], spec['seed']
    target = Path(output_root)/dataset/model/f'seed_{seed}'
    existing = checkpoint_for(Path(output_root),dataset,model,seed)
    if existing: return existing
    target.mkdir(parents=True,exist_ok=True)
    config=restore_config(spec['backbone_config'],data_root,target,device)
    init_seed(config['seed'],config['reproducibility'])
    complete=create_dataset(config)
    train,valid,_=data_preparation(config,complete)
    init_seed(config['seed']+config['local_rank'],config['reproducibility'])
    backbone=get_model(config['model'])(config,train._dataset).to(config['device'])
    trainer=get_trainer(config['MODEL_TYPE'],config['model'])(config,backbone)
    trainer.fit(train,valid,saved=True,show_progress=False)
    checkpoint=checkpoint_for(Path(output_root),dataset,model,seed)
    if checkpoint is None: raise RuntimeError('RecBole trainer produced no checkpoint')
    return checkpoint
