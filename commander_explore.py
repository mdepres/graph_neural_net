import os
import json
import yaml
import argparse

import torch
import torch.backends.cudnn as cudnn
from models import get_siamese_model_exp, get_siamese_model_test, get_node_model_exp, get_edge_model_exp, get_edge_model_test
import loaders.data_generator as dg
from loaders.loaders import siamese_loader, node_classif_loader
#from toolbox.optimizer import get_optimizer
import toolbox.utils as utils
from datetime import datetime
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor


def get_config(filename='default_config.yaml') -> dict:
    with open(filename, 'r') as f:
        config = yaml.safe_load(f)
    return config

def custom_name(config):
    l_name = [config['arch']['node_emb']['type'],
        config['data']['train']['generative_model'], config['data']['train']['n_vertices'],
        config['data']['train']['edge_density']]
    name = "_".join([str(e) for e in l_name])
    return name

def check_paths_update(config, name):
    """
    add to the configuration:
        'path_log' = root/experiments/$problem/$name/
            (arch_gnn)_(numb_locks)_(generative_model)_(n_vertices)_(edge_density)/date_time
        'date_time' 
        ['data']['path_dataset'] = root/experiments/qap/data
    save the new configuration at path_log/config.json
    """ 
    now = datetime.now() # current date and time
    date_time = now.strftime("%m-%d-%y-%H-%M")
    dic = {'date_time' : date_time}
    #expe_runs_qap = os.path.join(QAP_DIR,config['name'],'runs/')
    name = custom_name(config)
    name = os.path.join(name, str(date_time))
    path_log = os.path.join(PB_DIR,config['name'], name)
    utils.check_dir(path_log)
    dic['path_log'] = path_log

    utils.check_dir(DATA_PB_DIR)
    config['data'].update({'path_dataset' : DATA_PB_DIR})
    
    #print(path_log)
    config.update(dic)
    with open(os.path.join(path_log, 'config.json'), 'w') as f:
        json.dump(config, f)
    return config

def train(config):
    """ Main func """
    cpu = config['cpu']
    problem = config['problem']
    config_arch = config['arch'] 
    path_log = config['path_log']
    data = config['data']
    max_epochs = config['train']['epochs']
    batch_size = config['train']['batch_size']
    config_optim = config['train']
    log_freq = config_optim['log_freq']

    print("Heading to Training.")
    global best_score, best_epoch
    best_score, best_epoch = -1, -1
    print("Current problem : ", problem)

    use_cuda = not cpu and torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    print('Using device:', device)

    # init random seeds 
    utils.setup_env(cpu)
    
    print("Models saved in ", path_log)
    
    if config['problem'] == 'kcol' : # Coloring problem
        model_pl = get_node_model_exp(config_arch, config_optim, config['k'])
        generator = dg.KCOL_Generator
        data['train']['k'] = config['k']
    elif config['problem'] == 'mbs' : # Min bisection problem
        model_pl = get_edge_model_exp(config_arch, config_optim, 2) #get_node_model_exp(config_arch, config_optim, 2) node or edge mode
        generator = dg.MBS_Generator
    else: # QAP problem
        model_pl = get_siamese_model_exp(config_arch, config_optim) 
        generator = dg.QAP_Generator
    gene_train = generator('train', data['train'], data['path_dataset'])
    gene_train.load_dataset()
    gene_val = generator('val', data['train'], data['path_dataset'])
    gene_val.load_dataset()
    
    if config['problem'] == 'qap':
        train_loader = siamese_loader(gene_train, batch_size,
                                    gene_train.constant_n_vertices)
        val_loader = siamese_loader(gene_val, batch_size,
                                    gene_val.constant_n_vertices, shuffle=False)
    else:
        train_loader = node_classif_loader(gene_train, batch_size,
                                    gene_train.constant_n_vertices)
        val_loader = node_classif_loader(gene_val, batch_size,
                                    gene_val.constant_n_vertices, shuffle=False)
    
    #optimizer, scheduler = get_optimizer(train,model)
    #print("Model #parameters : ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    """ if not train['anew']:
        try:
            utils.load_model(model,device,train['start_model'])
            print("Model found, using it.")
        except RuntimeError:
            print("Model not existing. Starting from scratch.")
 """
    #model.to(device)
    # train model
    if config['observers']['wandb']:
        logger = WandbLogger(project=f"{config['problem']}_{config['name']}", log_model="all", save_dir=path_log)
        logger.experiment.config.update(config)
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        trainer = pl.Trainer(accelerator=device,max_epochs=max_epochs,logger=logger,log_every_n_steps=log_freq,callbacks=[lr_monitor],precision=16)
    else:
        trainer = pl.Trainer(accelerator=device,max_epochs=max_epochs,log_every_n_steps=log_freq,precision=16)
    trainer.fit(model_pl, train_loader, val_loader)

    return trainer

    """ is_best = True
    try:
        for epoch in range(train['epoch']):
            print('Current epoch: ', epoch)
            if not use_dgl:
                trainer.train_triplet(train_loader,model,optimizer,exp_helper,device,epoch,eval_score=True,print_freq=train['print_freq'])
            else:
                trainer.train_triplet_dgl(train_loader,model,optimizer,exp_helper,device,epoch,uncollate_function,
                                            sym_problem=symmetric_problem,eval_score=True,print_freq=train['print_freq'])
            

            if not use_dgl:
                relevant_metric, loss = trainer.val_triplet(val_loader,model,exp_helper,device,epoch,eval_score=True)
            else:
                relevant_metric, loss = trainer.val_triplet_dgl(val_loader,model,exp_helper,device,epoch,uncollate_function,eval_score=True)
            scheduler.step(loss)
            # remember best acc and save checkpoint
            is_best = (relevant_metric > best_score)
            best_score = max(relevant_metric, best_score)
            if True == is_best:
                best_epoch = epoch
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_score': best_score,
                'best_epoch': best_epoch,
                'exp_logger': exp_helper.get_logger(),
                }, is_best,path_log)

            cur_lr = utils.get_lr(optimizer)
            if exp_helper.stop_condition(cur_lr):
                print(f"Learning rate ({cur_lr}) under stopping threshold, ending training.")
                break 
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')
    if test_enabled:
        eval(use_model=model)
    """
def test(config):
    """ Main func.
    """
    cpu = config['cpu']
    #train, 
    #problem =config['problem']
    #config_arch = config['arch'] 
    #test_enabled, 
    #path_log = config['path_log']
    data = config['data']
    #max_epochs = config['train']['epochs']
    batch_size = config['train']['batch_size']
    #config_optim = config['train']
    #log_freq = config_optim['log_freq']

    print("Heading to Test.")

    use_cuda = not cpu and torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    print('Using device:', device)

    # init random seeds 
    utils.setup_env(cpu)
    
    path_data_test = os.path.join(data['path_dataset'], 'test/')
    utils.check_dir(path_data_test)
    
    if config['problem'] == 'kcol' : # Coloring problem
        model_pl = get_node_model_test(config_arch, config_optim, config['k'])
        generator = dg.KCOL_Generator
        data['train']['k'] = config['k']
    elif config['problem'] == 'mbs' : # Min bisection problem
        model_pl = get_edge_model_test(data['test']['path_model'])
        generator = dg.MBS_Generator
    else:
        model = get_siamese_model_test(data['test']['path_model'])
        generator = dg.QAP_Generator

    gene_test = generator('test', data['test'], path_data_test)
    gene_test.load_dataset()
    
    if config['problem'] == 'qap':
        test_loader = siamese_loader(gene_test, batch_size,
                                  gene_test.constant_n_vertices, shuffle=False)
    else:
        test_loader = node_classif_loader(gene_test, batch_size,
                                  gene_test.constant_n_vertices, shuffle=False)

    trainer = pl.Trainer(accelerator=device,precision=16)
    res_test = trainer.test(model_pl, test_loader)
    return res_test

def predict(config):
    """ Function to call on the prediction """
    
    cpu = config['cpu']
    data = config['data']
    batch_size = config['train']['batch_size']

    print("Predicting.")

    use_cuda = not cpu and torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    print('Using device:', device)

    # init random seeds 
    utils.setup_env(cpu)
    
    path_data_test = os.path.join(data['path_dataset'], 'test/')
    utils.check_dir(path_data_test)
    
    if config['problem'] == 'kcol' : # Coloring problem
        model_pl = get_node_model_test(config_arch, config_optim, config['k'])
        generator = dg.KCOL_Generator
        data['train']['k'] = config['k']
    elif config['problem'] == 'mbs' : # Min bisection problem
        model_pl = get_edge_model_test(data['test']['path_model'])
        generator = dg.MBS_Generator
    else:
        model_pl = get_siamese_model_test(data['test']['path_model'])
        generator = dg.QAP_Generator

    gene_test = generator('test', data['test'], path_data_test)
    
    graph = [gene_test.compute_example()]

    if config['problem'] == 'qap':
        pred_loader = siamese_loader(graph, batch_size,
                                  gene_test.constant_n_vertices, shuffle=False)
    else:
        pred_loader = node_classif_loader(graph, batch_size,
                                  gene_test.constant_n_vertices, shuffle=False)
    
    trainer = pl.Trainer(accelerator=device,precision=16)
    res_predict = trainer.predict(model_pl, pred_loader)
    return res_predict

#@ex.automain
def main():
    parser = argparse.ArgumentParser(description='Main file for creating experiments.')
    parser.add_argument('command', metavar='c', choices=['train','test','predict'],
                    help='Command to execute : train, test or predict on an example')
    parser.add_argument('--n_vertices', type=int, default=0)
    parser.add_argument('--noise', type=float, default=0)
    parser.add_argument('--edge_density', type=float, default=0)
    parser.add_argument('--block_init', type=str, default='block')
    parser.add_argument('--block_inside', type=str, default='block_inside')
    parser.add_argument('--node_emb', type=str, default='node_embedding_block')
    parser.add_argument('--config', type=str, default='default_config.yaml')
    args = parser.parse_args()
    if args.command=='train':
        training=True
        default_test = False
    elif args.command=='test':#not implemented yet !!! generator pb with name!
        training=False
        default_test=True
    elif args.command=='predict':
        training=False
        default_test=False

    config = get_config(args.config)
    if args.n_vertices != 0:
        config['data']['train']['n_vertices'] = args.n_vertices
    if args.noise != 0:
        config['data']['train']['noise'] = args.noise
    if args.edge_density != 0:
        config['data']['train']['edge_density'] = args.edge_density
    if args.block_init != 'block':
        config['arch']['node_emb']['block_init'] = args.block_init
        print(f"block_init override: {args.block_init}")
    if args.block_inside != 'block_inside':
        config['arch']['node_emb']['block_inside'] = args.block_inside
        print(f"block_inside override: {args.block_inside}")
    if args.node_emb != 'node_embedding_block':
        config['arch']['node_emb']['type'] = args.node_emb
        print(f"node_embedding override: {args.node_emb}")


    global ROOT_DIR 
    ROOT_DIR = Path.home()
    global EXPE_DIR #= os.path.join(ROOT_DIR,'experiments-gnn/')
    EXPE_DIR = os.path.join(ROOT_DIR,'experiments-gnn/')
    global PB_DIR #= os.path.join(EXPE_DIR, config['problem'])
    PB_DIR = os.path.join(EXPE_DIR, config['problem'])
    global DATA_PB_DIR #= os.path.join(PB_DIR,'data/')
    DATA_PB_DIR = os.path.join(PB_DIR,'data/')
    name = custom_name(config)
    config = check_paths_update(config, name)
    trainer=None
    if training:
        trainer = train(config)
    if default_test: #or config['test_enabled']:
        res_test = test(config)
    if not training and not default_test:
        res_predict = predict(config)

if __name__=="__main__":
    pl.seed_everything(3787, workers=True)
    main()
