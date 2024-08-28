import pygmo as pg
from src.fitting.pygmo_problems import *
from src.fitting.fastnumpyio import save, load

def get_partition(n_items: int, rank: int, size: int)-> tuple[int, int]: 
    """
    Compute the partition

    Returns (start, end) of partition
    """
    chunk_size = n_items // size
    if n_items % size:
        chunk_size += 1
    start = rank * chunk_size
    if rank + 1 == size:
        end = n_items
    else:
        end = start + chunk_size
    return (start, end)

def _ipy_bfe_func(dv_path, prob, rank, size):
    # The function that will be invoked
    # by the individual processes/nodes of mp/ipy bfe.
    import pickle

    all_dvs = load(dv_path)

    # pick dvs based on engine rank
    n_items = all_dvs.shape[0]
    start, end = get_partition(n_items, rank, size)
    dvs = all_dvs[start:end,:]

    return pickle.dumps(list(map(prob.fitness, dvs)))

def _ipy_bfe_func2(dv_path, prob, rank):
    # The function that will be invoked
    # by the individual processes/nodes of mp/ipy bfe.
    import pickle

    dvs = load(dv_path+'/dvs_'+str(rank)+'.b')
    print([str(rank), ])

    return pickle.dumps(list(map(prob.fitness, dvs)))

class pickleless_bfe(pg.ipyparallel_bfe):
    def __init__(self, client_kwargs, view_kwargs, temp_dv_path, prob:dict):
        self.client_kwargs = client_kwargs
        self.view_kwargs = view_kwargs
        self.temp_dv_path = temp_dv_path
        self.client_size = None
        self.prob = prob
        super().__init__()

    def init_view(self,client_args=[], client_kwargs={}, view_args=[], view_kwargs={}):
        self.client_kwargs = client_kwargs
        self.view_kwargs = view_kwargs
        with pg.ipyparallel_bfe._view_lock:
            if pg.ipyparallel_bfe._view is None:
                # Create the new view.
                from ipyparallel import Client, Reference
                rc = Client(*client_args, **self.client_kwargs)
                rc[:].use_cloudpickle()
                pg.ipyparallel_bfe._view = rc.broadcast_view(*view_args, **self.view_kwargs)
                pg.ipyparallel_bfe._view.is_coalescing = False
                self.client_size = len(rc.ids)
                pg.ipyparallel_bfe._view.scatter("rank", rc.ids, flatten=True)
                for k,v in self.prob.items():
                    prob_id = "prob_"+k
                    pg.ipyparallel_bfe._view.push({prob_id: v}, block = True)
                    pg.ipyparallel_bfe._view.apply_sync(lambda x: x.extract(SBMLGlobalFit_Multi_Fly)._setup_rr(), Reference(prob_id))

    def __call__(self, prob, dvs, mode = 'train'):
        import ipyparallel as ipp
        import pickle
        import numpy as np
        prob_id = "prob_"+mode
        # Fetch the dimension and the fitness
        # dimension of the problem.
        ndim = prob.get_nx()
        nf = prob.get_nf()

        # Compute the total number of decision
        # vectors represented by dvs.
        ndvs = len(dvs) // ndim
        # Reshape dvs so that it represents
        # ndvs decision vectors of dimension ndim
        # each.
        dvs.shape = (ndvs, ndim)

        # Save dvs in a binary file
        for i in range(self.client_size):
            # pick dvs based on engine rank
            start, end = get_partition(ndvs, i, self.client_size)
            save(self.temp_dv_path+'/dvs_'+str(i)+'.b', dvs[start:end,:])

        with pg.ipyparallel_bfe._view_lock:

            ar = pg.ipyparallel_bfe._view.apply(_ipy_bfe_func2, self.temp_dv_path, ipp.Reference(prob_id), ipp.Reference("rank"))
            # ret = ipp.AsyncMapResult(pg.ipyparallel_bfe._view.client, ar._children, ipp.client.map.Map())
            
        # Build the vector of fitness vectors as a 2D numpy array.
        fvs = np.array(sum([pickle.loads(fv) for fv in ar.get()],[]))
        # Reshape it so that it is 1D.
        fvs.shape = (ndvs*nf,)
        del ar, dvs, ipp
        return fvs

def _ipy_bfe_metrics(dv_path, prob, rank):
    # The function that will be invoked
    # by the individual processes/nodes of mp/ipy bfe.
    import pickle

    dvs = load(dv_path+'/dvs_'+str(rank)+'.b')
    print([str(rank), ])

    return pickle.dumps(list(map(prob.fitness, dvs)))

class mkcook_bfe(pg.ipyparallel_bfe):
    def __init__(self, client_kwargs, view_kwargs, temp_dv_path, prob):
        self.client_kwargs = client_kwargs
        self.view_kwargs = view_kwargs
        self.temp_dv_path = temp_dv_path
        self.client_size = None
        self.prob = prob
        super().__init__()

    def init_view(self,client_args=[], client_kwargs={}, view_args=[], view_kwargs={}):
        self.client_kwargs = client_kwargs
        self.view_kwargs = view_kwargs
        with pg.ipyparallel_bfe._view_lock:
            if pg.ipyparallel_bfe._view is None:
                # Create the new view.
                from ipyparallel import Client, Reference
                rc = Client(*client_args, **self.client_kwargs)
                rc[:].use_cloudpickle()
                pg.ipyparallel_bfe._view = rc.broadcast_view(*view_args, **self.view_kwargs)
                pg.ipyparallel_bfe._view.is_coalescing = False
                self.client_size = len(rc.ids)
                pg.ipyparallel_bfe._view.scatter("rank", rc.ids, flatten=True)
                pg.ipyparallel_bfe._view.push({'prob': self.prob}, block = True)
                pg.ipyparallel_bfe._view.apply_sync(lambda x: x.extract(SBML_Barebone_Multi_Fly)._setup_rr(), Reference('prob'))

    def __call__(self, prob, dvs):
        import ipyparallel as ipp
        import pickle
        import numpy as np
        # Fetch the dimension and the fitness
        # dimension of the problem.
        ndim = prob.get_nx()
        nf = prob.get_nf()

        # Compute the total number of decision
        # vectors represented by dvs.
        ndvs = len(dvs) // ndim
        # Reshape dvs so that it represents
        # ndvs decision vectors of dimension ndim
        # each.
        dvs.shape = (ndvs, ndim)

        # Save dvs in a binary file
        for i in range(self.client_size):
            # pick dvs based on engine rank
            start, end = get_partition(ndvs, i, self.client_size)
            save(self.temp_dv_path+'/dvs_'+str(i)+'.b', dvs[start:end,:])

        with pg.ipyparallel_bfe._view_lock:

            ar = pg.ipyparallel_bfe._view.apply(_ipy_bfe_metrics, self.temp_dv_path, ipp.Reference('prob'), ipp.Reference("rank"))
            # ret = ipp.AsyncMapResult(pg.ipyparallel_bfe._view.client, ar._children, ipp.client.map.Map())
            
        # Build the vector of fitness vectors as a 2D numpy array.
        fvs = np.array(sum([pickle.loads(fv) for fv in ar.get()],[]))
        # Reshape it so that it is 1D.
        fvs.shape = (ndvs*nf,)
        del ar, dvs, ipp
        return fvs