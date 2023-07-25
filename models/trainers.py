import pytorch_lightning as pl
from toolbox.losses import triplet_loss
from toolbox.metrics import accuracy_max
from toolbox.utils import mbs_pretty_print
from models.blocks_emb import *
from models.utils import *

get_node_emb = {'node_embedding_full': node_embedding_full,
    'node_embedding': node_embedding,
    'node_embedding_block': node_embedding_block,
    'node_embedding_iter': node_embedding_iter,
    'node_embedding_rec': node_embedding_rec
    }

get_block_init = {
    'block': block,
    'block_diag': block_diag,
    'block_sym': block_sym,
    'block_sym_diag': block_sym_diag,
    'block_emb': block_emb
}
    
get_block_inside = {'block_inside': block_inside, 
    'block_diag_inside': block_diag_inside,
    'block_sym_inside': block_sym_inside,
    'block_sym_diag_inside': block_sym_diag_inside,
    'block_multi_mlp': block_multi_mlp,
    'block_multi' : block_multi,
    'block_mix' : block_mix,
    'block_diag_mix' : block_diag_mix,
    'block_res': block_res,
    'block_att_inside': block_att_inside,
    'block': block
    }

class Siamese_Node_Exp(pl.LightningModule):
    def __init__(self, original_features_num, node_emb, lr=1e-3, scheduler_decay=0.5, scheduler_step=3, lr_stop = 1e-5):
        """
        take a batch of pair of graphs as
        (bs, original_features, n_vertices, n_vertices)
        and return a batch of "node similarities (i.e. dot product)"
        with shape (bs, n_vertices, n_vertices)
        graphs must NOT have same size inside the batch when maskedtensors are used
        """
        super().__init__()

        #original_features_num = node_emb['original_features_num']
        try:
            node_emb_type = get_node_emb[node_emb['type']]
        except KeyError:
            raise NotImplementedError(f"node embedding {node_emb['type']} is not implemented")
        try:
            block_inside = get_block_inside[node_emb['block_inside']]
            node_emb['block_inside'] = block_inside
        except KeyError:
            raise NotImplementedError(f"block inside {node_emb['block_inside']} is not implemented")
        try:
            block_init = get_block_init[node_emb['block_init']]
            node_emb['block_init'] = block_init
        except KeyError:
            raise NotImplementedError(f"block init {node_emb['block_init']} is not implemented")

    # self. = node_emb['type']
        #self.num_blocks = num_blocks
        #self.in_features = in_features
        #self.out_features = out_features
        #self.depth_of_mlp = depth_of_mlp
        self.node_embedder_dic = {
            'input': (None, []), 
            'ne': node_emb_type(original_features_num, **node_emb)#, block_init= block_init, block_inside=block_inside) 
                }
        self.node_embedder = Network(self.node_embedder_dic)
        
        self.loss = triplet_loss()
        self.metric = accuracy_max
        self.lr = lr
        self.scheduler_decay = scheduler_decay
        self.scheduler_step = scheduler_step
        self.lr_stop = lr_stop

        self.save_hyperparameters()

    def forward(self, x1, x2):
        """
        Data should be given with the shape (b,2,f,n,n)
        """
        x1 = self.node_embedder(x1)['ne/suffix']
        x2 = self.node_embedder(x2)['ne/suffix']
        # einsum not supported with maskedtensors
        #raw_scores = torch.einsum('bfi,bfj-> bij', x1, x2)
        raw_scores = torch.matmul(torch.transpose(x1,1,2),x2)
        return raw_scores
    
    def training_step(self, batch, batch_idx):
        #g, target = batch
        raw_scores = self(batch[0], batch[1])
        #raw_scores = x.squeeze(-1)
        loss = self.loss(raw_scores)
        self.log('train_loss', loss)
        (acc,n) = self.metric(raw_scores)
        #print(acc)
        self.log("train_acc", acc/n)
        #self.log_metric('train', data=g, raw_scores=raw_scores, target=target)
        return loss

    def validation_step(self, batch, batch_idx):
        raw_scores = self(batch[0], batch[1])
        #raw_scores = x.squeeze(-1)
        loss = self.loss(raw_scores)
        self.log('val_loss', loss)
        (acc,n) = self.metric(raw_scores)
        self.log("val_acc", acc/n)

    def test_step(self, batch, batch_idx):
        raw_scores = self(batch[0], batch[1])
        #raw_scores = x.squeeze(-1)
        loss = self.loss(raw_scores)
        self.log('test_loss', loss)
        (acc,n) = self.metric(raw_scores)
        self.log("test_acc", acc/n)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr,
                            amsgrad=False)
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=self.scheduler_decay, patience=self.scheduler_step, verbose=True, min_lr=self.lr_stop),
            "monitor": "val_loss",
            "frequency": 1
            # If "monitor" references validation metrics, then "frequency" should be set to a
            # multiple of "trainer.check_val_every_n_epoch".
        },
    }

import torchmetrics

class Graph_Classif_Exp(pl.LightningModule):
    def __init__(self, original_features_num, num_blocks, in_features,
                out_features, depth_of_mlp, block=block, constant_n_vertices=True, input_embed=False,
                n_classes = 10, classifier=None,
                    lr=1e-3, scheduler_decay=0.5, scheduler_step=5):
        """
        take a batch of graphs as
        (bs, original_features, n_vertices, n_vertices)
        and return a batch of graph features (bs, n_classes)
        graphs must NOT have same size inside the batch when maskedtensors are used
        """
        super().__init__()

        self.original_features_num = original_features_num
        self.num_blocks = num_blocks
        self.in_features = in_features
        self.out_features = out_features
        self.depth_of_mlp = depth_of_mlp
        self.graph_embedder_dic = graph_embedding_block(1,original_features_num-1, num_blocks=num_blocks, 
            out_features=out_features, depth_of_mlp=depth_of_mlp, constant_n_vertices=constant_n_vertices)
        #self.graph_embedder_dic = graph_embedding_block(1,original_features_num-1, num_blocks, 
        #    in_features,out_features, depth_of_mlp, block, constant_n_vertices=constant_n_vertices)
        #self.graph_embedder_dic = graph_embedding_transformer(1,original_features_num-1, num_blocks, 
        #    in_features,out_features, depth_of_mlp, block, constant_n_vertices=constant_n_vertices)
        self.graph_embedder = Network(self.graph_embedder_dic)

        if classifier is None:
            #self.classifier = nn.Sequential(nn.Linear((num_blocks+1)*out_features, n_classes), nn.LogSoftmax(dim=1))
            self.classifier = nn.Sequential(nn.Linear(2*out_features, n_classes), nn.LogSoftmax(dim=-1))
        else:
            self.classifier = classifier

        self.loss = nn.NLLLoss()
        self.lr = lr
        self.scheduler_decay = scheduler_decay
        self.scheduler_step = scheduler_step
        self.accuracy = torchmetrics.Accuracy(task='multiclass')

        self.save_hyperparameters()

    def forward(self, x):
        x = self.graph_embedder(x)['suffix']
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        #g, target = batch
        logp = self(batch)
        loss = self.loss(logp, batch['target'])
        self.log('train_loss', loss)
        acc = self.accuracy(logp.tensor.rename(None), batch['target'])
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        target = batch['target']
        logp = self(batch)
        loss = self.loss(logp, target)
        self.log('val_loss', loss)
        acc = self.accuracy(logp.tensor.rename(None), target)
        self.log("val_acc", acc)
        

    def test_step(self, batch, batch_idx):
        target = batch['target']
        logp = self(batch)
        loss = self.loss(logp, target)
        self.log('test_loss', loss)
        acc = self.accuracy(logp.tensor.rename(None), target)
        self.log("test_acc", acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr,
                            amsgrad=False)
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=self.scheduler_decay, patience=self.scheduler_step, verbose=True, min_lr=0.0001) ,
            "monitor": "val_loss",
            "frequency": 1
            # If "monitor" references validation metrics, then "frequency" should be set to a
            # multiple of "trainer.check_val_every_n_epoch".
        },
    }

class Node_Classif_Exp(pl.LightningModule):
    def __init__(self, original_features_num, num_blocks, in_features,
                out_features, depth_of_mlp, block=block, constant_n_vertices=True, input_embed=False,
                n_classes = 10, classifier=None,
                    lr=1e-3, scheduler_decay=0.5, scheduler_step=5):
        """
        Class for node classification-like experiment, using GNN-produced node embeddings.
        take a batch of graphs as (bs, original_features, n_vertices, n_vertices)
        and return a batch of graph features (bs, n_classes)
        graphs must NOT have same size inside the batch when maskedtensors are used
        """
        super().__init__()

        self.original_features_num = original_features_num
        self.num_blocks = num_blocks
        self.in_features = in_features
        self.out_features = out_features
        self.depth_of_mlp = depth_of_mlp
        self.node_embedder_dic = node_embedding_block(original_features_num, num_blocks=num_blocks, 
            out_features=out_features, depth_of_mlp=depth_of_mlp, num_heads=4, constant_n_vertices=constant_n_vertices)
        self.node_embedder = Network(self.node_embedder_dic)

        if classifier is None:
            #self.classifier = nn.Sequential(nn.Linear((num_blocks+1)*out_features, n_classes), nn.LogSoftmax(dim=1))
            self.classifier = nn.Sequential(nn.Linear(out_features, n_classes), nn.LogSoftmax(dim=-1))
        else:
            self.classifier = classifier

        self.loss = nn.NLLLoss()
        self.lr = lr
        self.scheduler_decay = scheduler_decay
        self.scheduler_step = scheduler_step
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes = n_classes)

        self.save_hyperparameters()

    def forward(self, x):
        x = self.node_embedder(x)['suffix']
        return self.classifier(x.permute(0,2,1)) # x arrives with dimension (bs, output_dim, n_vertices)

    def training_step(self, batch, batch_idx):
        logp = self(batch).permute(0,2,1)
        loss = self.loss(logp, batch['target'].long())
        self.log('train_loss', loss)
        acc = self.accuracy(logp.tensor.rename(None), batch['target'])
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        target = batch['target']
        logp = self(batch).permute(0,2,1)
        loss = self.loss(logp, target.long())
        self.log('val_loss', loss)
        acc = self.accuracy(logp.tensor.rename(None), target)
        self.log("val_acc", acc)

    def test_step(self, batch, batch_idx):
        target = batch['target']
        logp = self(batch)
        loss = self.loss(logp, target)
        self.log('test_loss', loss)
        acc = self.accuracy(logp.tensor.rename(None), target)
        self.log("test_acc", acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr,
                            amsgrad=False)
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=self.scheduler_decay, patience=self.scheduler_step, verbose=True, min_lr=0.0001) ,
            "monitor": "val_loss",
            "frequency": 1
            # If "monitor" references validation metrics, then "frequency" should be set to a
            # multiple of "trainer.check_val_every_n_epoch".
        },
    }

class Edge_Classif_Exp(pl.LightningModule):
    def __init__(self, original_features_num, num_blocks, in_features,
                out_features, depth_of_mlp, block=block, constant_n_vertices=True, input_embed=False,
                n_classes = 10, classifier=None,
                    lr=1e-3, scheduler_decay=0.5, scheduler_step=5):
        """
        Class for edge classification-like experiment, using GNN-produced edge embeddings.
        take a batch of graphs as (bs, original_features, n_vertices, n_vertices)
        and return a batch of graph features (bs, n_classes)
        graphs must NOT have same size inside the batch when maskedtensors are used
        """
        super().__init__()

        self.original_features_num = original_features_num
        self.num_blocks = num_blocks
        self.in_features = in_features
        self.out_features = out_features
        self.depth_of_mlp = depth_of_mlp
        self.node_embedder_dic = node_embedding_block(original_features_num, num_blocks=num_blocks, 
            out_features=out_features, depth_of_mlp=depth_of_mlp, num_heads=4, constant_n_vertices=constant_n_vertices)
        self.node_embedder = Network(self.node_embedder_dic)

        if classifier is None:
            self.classifier = nn.Sequential(nn.Linear(out_features, n_classes), nn.LogSoftmax(dim=-1))
        else:
            self.classifier = classifier

        self.loss = nn.NLLLoss()
        self.lr = lr
        self.scheduler_decay = scheduler_decay
        self.scheduler_step = scheduler_step
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes = n_classes)

        self.save_hyperparameters()

    def forward(self, x):
        x = self.node_embedder(x)['bm/block4/mlp3']
        x = x.permute(0,2,3,1) # Putting the output dimension as last dimension
        #x = x.view(x.shape[0], x.shape[1]*x.shape[2],x.shape[3]) 
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        logp = self(batch).permute(0,3,1,2) #Putting output dimension before data dimensions
        loss = self.loss(logp.contiguous(), batch['target'].long())
        self.log('train_loss', loss)
        acc = self.accuracy(logp.tensor.rename(None), batch['target'])
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        target = batch['target']
        logp = self(batch).permute(0,3,1,2)
        loss = self.loss(logp, target.long())
        self.log('val_loss', loss)
        acc = self.accuracy(logp.tensor.rename(None), target)
        self.log("val_acc", acc)

    def test_step(self, batch, batch_idx):
        target = batch['target']
        logp = self(batch).permute(0,3,1,2)
        #loss = self.loss(logp, target.long())
        #self.log('test_loss', loss)
        acc = self.accuracy(logp.tensor.rename(None), target)
        self.log("test_acc", acc)
        
    def predict_step(self, batch, batch_idx):
        logp = self(batch).permute(0,3,1,2)
        edge_classif = torch.argmax(logp.tensor.rename(None), dim=1)
        mbs_pretty_print(batch['input'][0][0],edge_classif,batch['target'][0])
        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr,
                            amsgrad=False)
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=self.scheduler_decay, patience=self.scheduler_step, verbose=True, min_lr=0.0001) ,
            "monitor": "val_loss",
            "frequency": 1
            # If "monitor" references validation metrics, then "frequency" should be set to a
            # multiple of "trainer.check_val_every_n_epoch".
        },
    }
