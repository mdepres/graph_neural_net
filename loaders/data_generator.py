import os
import random
import itertools
import networkx
#from networkx.algorithms.approximation.clique import max_clique
from numpy import diag_indices
import numpy as np
import torch
import torch.utils
import toolbox.utils as utils
#from toolbox.searches import mcp_beam_method
from sklearn.decomposition import PCA
#from numpy import pi,angle,cos,sin
from numpy.random import default_rng
import tqdm
from numpy import mgrid as npmgrid

from numpy import indices as npindices, argpartition as npargpartition, array as nparray


rng = default_rng(41)

GENERATOR_FUNCTIONS = {}
ADJ_UNIQUE_TENSOR = torch.Tensor([0.,1.])

def is_adj(matrix):
    return torch.all((matrix==0) + (matrix==1))

class TimeOutException(Exception):
    pass

def generates(name):
    """ Register a generator function for a graph distribution """
    def decorator(func):
        GENERATOR_FUNCTIONS[name] = func
        return func
    return decorator

@generates("ErdosRenyi")
def generate_erdos_renyi_netx(p, N):
    """ Generate random Erdos Renyi graph """
    g = networkx.erdos_renyi_graph(N, p)
    W = networkx.adjacency_matrix(g).todense()
    return g, torch.as_tensor(W, dtype=torch.float)

@generates("BarabasiAlbert")
def generate_barabasi_albert_netx(p, N):
    """ Generate random Barabasi Albert graph """
    m = int(p*(N -1)/2)
    g = networkx.barabasi_albert_graph(N, m)
    W = networkx.adjacency_matrix(g).todense()
    return g, torch.as_tensor(W, dtype=torch.float)

@generates("Regular")
def generate_regular_graph_netx(p, N):
    """ Generate random regular graph """
    d = p * N
    d = int(d)
    # Make sure N * d is even
    if N * d % 2 == 1:
        d += 1
    g = networkx.random_regular_graph(d, N)
    W = networkx.adjacency_matrix(g).todense()
    return g, torch.as_tensor(W, dtype=torch.float)

NOISE_FUNCTIONS = {}

def noise(name):
    """ Register a noise function """
    def decorator(func):
        NOISE_FUNCTIONS[name] = func
        return func
    return decorator

@noise("ErdosRenyi")
def noise_erdos_renyi(g, W, noise, edge_density):
    n_vertices = len(W)
    pe1 = noise
    pe2 = (edge_density*noise)/(1-edge_density)
    _,noise1 = generate_erdos_renyi_netx(pe1, n_vertices)
    _,noise2 = generate_erdos_renyi_netx(pe2, n_vertices)
    W_noise = W*(1-noise1) + (1-W)*noise2
    return W_noise

def is_swappable(g, u, v, s, t):
    """
    Check whether we can swap
    the edges u,v and s,t
    to get u,t and s,v
    """
    actual_edges = g.has_edge(u, v) and g.has_edge(s, t)
    no_self_loop = (u != t) and (s != v)
    no_parallel_edge = not (g.has_edge(u, t) or g.has_edge(s, v))
    return actual_edges and no_self_loop and no_parallel_edge

def do_swap(g, u, v, s, t):
    g.remove_edge(u, v)
    g.remove_edge(s, t)
    g.add_edge(u, t)
    g.add_edge(s, v)

@noise("EdgeSwap")
def noise_edge_swap(g, W, noise, edge_density): #Permet de garder la regularite
    g_noise = g.copy()
    edges_iter = list(itertools.chain(iter(g.edges), ((v, u) for (u, v) in g.edges)))
    for u,v in edges_iter:
        if random.random() < noise:             
            for s, t in edges_iter:
                if random.random() < noise and is_swappable(g_noise, u, v, s, t):
                    do_swap(g_noise, u, v, s, t)
    W_noise = networkx.adjacency_matrix(g_noise).todense()
    return torch.as_tensor(W_noise, dtype=torch.float)

def adjacency_matrix_to_tensor_representation(W):
    """ Create a tensor B[0,:,:] = W and B[1,i,i] = deg(i)"""
    degrees = W.sum(1)
    B = torch.zeros((2,len(W), len(W)))
    B[0, :, :] = W
    indices = torch.arange(len(W))
    B[1, indices, indices] = degrees
    return B

class Base_Generator(torch.utils.data.Dataset):
    def __init__(self, name, path_dataset, num_examples):
        self.path_dataset = path_dataset
        self.name = name
        self.num_examples = num_examples

    def load_dataset(self, use_dgl= False):
        """
        Look for required dataset in files and create it if
        it does not exist
        """
        filename = self.name + '.pkl'
        filename_dgl = self.name + '_dgl.pkl'
        path = os.path.join(self.path_dataset, filename)
        path_dgl = os.path.join(self.path_dataset, filename_dgl)
        if os.path.exists(path):
            if use_dgl:
                print('Reading dataset at {}'.format(path_dgl))
                data = torch.load(path_dgl)
            else:
                print('Reading dataset at {}'.format(path))
                data = torch.load(path)
            self.data = list(data)
        else:
            print('Creating dataset at {}'.format(path))
            l_data = self.create_dataset()
            print('Saving dataset at {}'.format(path))
            torch.save(l_data, path)
            self.data = l_data
    
    def remove_file(self):
        os.remove(os.path.join(self.path_dataset, self.name + '.pkl'))
    
    def create_dataset(self):
        l_data = []
        for _ in tqdm.tqdm(range(self.num_examples)):
            example = self.compute_example()
            l_data.append(example)
        return l_data

    def __getitem__(self, i):
        """ Fetch sample at index i """
        return self.data[i]

    def __len__(self):
        """ Get dataset length """
        return len(self.data)

class QAP_Generator(Base_Generator):
    """
    Build a numpy dataset of pairs of (Graph, noisy Graph)
    """
    def __init__(self, name, args, path_dataset):
        self.generative_model = args['generative_model']
        self.noise_model = args['noise_model']
        self.edge_density = args['edge_density']
        self.noise = args['noise']
        num_examples = args['num_examples_' + name]
        n_vertices = args['n_vertices']
        vertex_proba = args['vertex_proba']
        subfolder_name = 'QAP_{}_{}_{}_{}_{}_{}_{}'.format(self.generative_model,
                                                     self.noise_model,
                                                     num_examples,
                                                     n_vertices, vertex_proba,
                                                     self.noise, self.edge_density)
        path_dataset = os.path.join(path_dataset, subfolder_name)
        super().__init__(name, path_dataset, num_examples)
        self.data = []
        self.constant_n_vertices = (vertex_proba == 1.)
        self.n_vertices_sampler = torch.distributions.Binomial(n_vertices, vertex_proba)
        
        
        utils.check_dir(self.path_dataset)

    def compute_example(self):
        """
        Compute pairs (Adjacency, noisy Adjacency)
        """
        n_vertices = int(self.n_vertices_sampler.sample().item())
        try:
            g, W = GENERATOR_FUNCTIONS[self.generative_model](self.edge_density, n_vertices)
        except KeyError:
            raise ValueError('Generative model {} not supported'
                             .format(self.generative_model))
        try:
            W_noise = NOISE_FUNCTIONS[self.noise_model](g, W, self.noise, self.edge_density)
        except KeyError:
            raise ValueError('Noise model {} not supported'
                             .format(self.noise_model))
        B = adjacency_matrix_to_tensor_representation(W)
        B_noise = adjacency_matrix_to_tensor_representation(W_noise)
        #data = torch.cat((B.unsqueeze(0),B_noise.unsqueeze(0)))
        return (B, B_noise)
    
def make_laplacian(W):
    D = W @ torch.ones(W.shape[-1])
    return torch.diag(1/torch.sqrt(D)) @ W @ torch.diag(1/torch.sqrt(D))

def make_spectral_feature(L,n=4):
    out = torch.zeros((n,*L.shape))
    scale = 1#L.shape[-1]
    L_prev = torch.eye(L.shape[-1])
    for i in range(n):
        L_prev = L_prev @ L
        out[i,:,:] = scale*L_prev 
    return out

class QAP_spectralGenerator(Base_Generator):
    """
    Build a numpy dataset of pairs of (Graph, noisy Graph)
    """
    def __init__(self, name, args, path_dataset):
        self.generative_model = args['generative_model']
        self.noise_model = args['noise_model']
        self.edge_density = args['edge_density']
        self.noise = args['noise']
        num_examples = args['num_examples_' + name]
        n_vertices = args['n_vertices']
        vertex_proba = args['vertex_proba']
        subfolder_name = 'QAPspectral_{}_{}_{}_{}_{}_{}_{}'.format(self.generative_model,
                                                     self.noise_model,
                                                     num_examples,
                                                     n_vertices, vertex_proba,
                                                     self.noise, self.edge_density)
        path_dataset = os.path.join(path_dataset, subfolder_name)
        super().__init__(name, path_dataset, num_examples)
        self.data = []
        self.constant_n_vertices = (vertex_proba == 1.)
        self.n_vertices_sampler = torch.distributions.Binomial(n_vertices, vertex_proba)
        utils.check_dir(self.path_dataset)

    def compute_example(self):
        """
        Compute pairs (Adjacency, noisy Adjacency)
        """
        n_vertices = int(self.n_vertices_sampler.sample().item())
        try:
            g, W = GENERATOR_FUNCTIONS[self.generative_model](self.edge_density, n_vertices)
        except KeyError:
            raise ValueError('Generative model {} not supported'
                             .format(self.generative_model))
        try:
            W_noise = NOISE_FUNCTIONS[self.noise_model](g, W, self.noise, self.edge_density)
        except KeyError:
            raise ValueError('Noise model {} not supported'
                             .format(self.noise_model))
        L = make_laplacian(W)
        L_noise = make_laplacian(W_noise)
        F = make_spectral_feature(L)
        F_noise = make_spectral_feature(L_noise)
        return (F, F_noise)

class GCP_Generator(Base_Generator):
    """
    Build a numpy dataset of graphs and colorings
    """
    def __init__(self, name, args, path_dataset):
        self.generative_model = args['generative_model']
        self.edge_density = args['edge_density']
        num_examples = args['num_examples_' + name]
        n_vertices = args['n_vertices']
        vertex_proba = args['vertex_proba']
        self.num_colors_low = args['num_colors_low']
        self.num_colors_high = args['num_colors_high']
        subfolder_name = 'Color_{}_{}_{}_{}_{}'.format(self.generative_model,
                                                     num_examples,
                                                     n_vertices, vertex_proba, self.edge_density)
        path_dataset = os.path.join(path_dataset, subfolder_name)
        super().__init__(name, path_dataset, num_examples)
        self.data = []
        self.constant_n_vertices = (vertex_proba == 1.)
        self.n_vertices_sampler = torch.distributions.Binomial(n_vertices, vertex_proba)
        utils.check_dir(self.path_dataset)
    
    def generate_k(self):
        return torch.randint(self.num_colors_low, self.num_colors_high+1, (1,)).item()

    def compute_example(self):
        """
        Compute adjacencies and associated colorings
        """
        n_vertices = int(self.n_vertices_sampler.sample().item())
        k = self.generate_k
        G = networkx.Graph()
        for i in range(n_vertices):
            G.add_node(i)
        coloring = torch.randint(0, k, (n_vertices,)) # Draw a coloring
        
        for i in range(n_vertices):
            for j in range(i+1, n_vertices):
                if coloring[i]!=coloring[j] and torch.rand(1).item()<self.edge_density:
                    G.add_edge(i,j)
        W = networkx.adjacency_matrix(G)
        W = W.todense()
        W = torch.as_tensor(W, dtype=torch.float)
        data = adjacency_matrix_to_tensor_representation(W)
        #c = networkx.greedy_color(G)
        #print(max(c.values()),k)
        return (data,coloring)

class KCOL_Generator(Base_Generator):
    def __init__(self, name, args, path_dataset):
        self.generative_model = args['generative_model']
        self.edge_density = args['edge_density']
        num_examples = args['num_examples_' + name]
        n_vertices = args['n_vertices']
        vertex_proba = args['vertex_proba']
        self.k = args['k']
        subfolder_name = 'Color_{}_{}_{}_{}_{}'.format(self.generative_model,
                                                     num_examples,
                                                     n_vertices, vertex_proba, self.edge_density)
        path_dataset = os.path.join(path_dataset, subfolder_name)
        super().__init__(name, path_dataset, num_examples)
        self.data = []
        self.constant_n_vertices = (vertex_proba == 1.)
        self.n_vertices_sampler = torch.distributions.Binomial(n_vertices, vertex_proba)
        utils.check_dir(self.path_dataset)
    
    def compute_example(self):
        """
        Compute adjacencies and associated colorings
        """
        n_vertices = int(self.n_vertices_sampler.sample().item())
        G = networkx.Graph()
        for i in range(n_vertices):
            G.add_node(i)
        coloring = torch.randint(0, self.k, (n_vertices,)) # Draw a coloring
        
        for i in range(n_vertices):
            for j in range(i+1, n_vertices):
                if coloring[i]!=coloring[j] and torch.rand(1).item()<self.edge_density:
                    G.add_edge(i,j)
        W = networkx.adjacency_matrix(G)
        W = W.todense()
        W = torch.as_tensor(W, dtype=torch.float)
        data = adjacency_matrix_to_tensor_representation(W)
        
        # If one-hot vectors are needed
        #target = torch.zeros((n_vertices,self.k))
        #for i in range(n_vertices):
            #target[i][coloring[i]] = 1
        return (data,coloring)

class MBS_Generator(Base_Generator):
    def __init__(self, name, args, path_dataset):
        self.generative_model = args['generative_model']
        self.edge_density = args['edge_density']
        self.connection_density = args['connection_density']
        num_examples = args['num_examples_' + name]
        n_vertices = args['n_vertices']
        vertex_proba = args['vertex_proba']
        subfolder_name = 'MBS_{}_{}_{}_{}_{}_{}'.format(self.generative_model,
                                                     num_examples,
                                                     n_vertices, vertex_proba, self.edge_density, self.connection_density)
        path_dataset = os.path.join(path_dataset, subfolder_name)
        super().__init__(name, path_dataset, num_examples)
        self.data = []
        self.constant_n_vertices = (vertex_proba == 1.)
        self.n_vertices_sampler = torch.distributions.Binomial(n_vertices, vertex_proba)
        utils.check_dir(self.path_dataset)

    def compute_example_predict(self):
        """
        Compute adjacencies and planted assignement
        """
        n_vertices = int(self.n_vertices_sampler.sample().item())
        G = networkx.Graph()
        for i in range(n_vertices):
            G.add_node(i)
        # Draw an assignement
        nodes = torch.randperm(n_vertices)[:n_vertices//2]
        groups = torch.zeros((n_vertices,)) 
        groups[nodes] = 1
        edge_target = torch.zeros((n_vertices,n_vertices))
        
        nb_internal_edges = 0
        nb_external_edges = 0
        for i in range(n_vertices):
            for j in range(i+1, n_vertices):
                if groups[i]==groups[j] and torch.rand(1).item()<self.edge_density:
                    G.add_edge(i,j)
                    nb_internal_edges+=1
                elif groups[i]!=groups[j] and torch.rand(1).item()<self.connection_density:
                    G.add_edge(i,j)
                    edge_target[i,j] = 1
                    nb_external_edges+=1
        W = networkx.adjacency_matrix(G)
        W = W.todense()
        W = torch.as_tensor(W, dtype=torch.float)
        data = adjacency_matrix_to_tensor_representation(W)
        
        return (data, edge_target, groups.detach().numpy())
    
    def compute_example(self):
        data, edge_target, groups = self.compute_example_predict()
        return (data, edge_target)

class DC_Generator(Base_Generator):
    """ Detect cycle, a simple problem """
    def __init__(self, name, args, path_dataset):
        self.generative_model = args['generative_model']
        self.edge_density = args['edge_density']
        self.connection_density = args['connection_density']
        num_examples = args['num_examples_' + name]
        n_vertices = args['n_vertices']
        vertex_proba = args['vertex_proba']
        subfolder_name = 'Cycle_{}_{}_{}_{}_{}'.format(self.generative_model,
                                                     num_examples,
                                                     n_vertices, vertex_proba, self.edge_density)
        path_dataset = os.path.join(path_dataset, subfolder_name)
        super().__init__(name, path_dataset, num_examples)
        self.data = []
        self.constant_n_vertices = (vertex_proba == 1.)
        self.n_vertices_sampler = torch.distributions.Binomial(n_vertices, vertex_proba)
        utils.check_dir(self.path_dataset)

    def compute_example(self):
        """
        Compute adjacencies and planted assignement
        """
        n_vertices = int(self.n_vertices_sampler.sample().item())
        G = networkx.Graph()
        for i in range(n_vertices):
            G.add_node(i)
        # Draw a cycle
        cycle_length = np.random.randint(n_vertices/10,n_vertices-5)
        # Draw the corresponding nodes
        cycle = np.random.choice(n_vertices, size=cycle_length, replace=False)
        target = torch.zeros(n_vertices)
        for node in cycle:
            target[node] = 1
        
        # Add the cycle's edges
        for i in range(len(cycle)):
            if i+1>=len(cycle):
                G.add_edge(cycle[i],cycle[0])
            else:
                G.add_edge(cycle[i],cycle[i+1]) 
            
        # Add other edges
        for i in range(n_vertices):
            for j in range(i+1, n_vertices):
                if torch.rand(1).item()<self.edge_density:
                    G.add_edge(i,j)
        W = networkx.adjacency_matrix(G)
        W = W.todense()
        W = torch.as_tensor(W, dtype=torch.float)
        data = adjacency_matrix_to_tensor_representation(W)

        return (data,target)
