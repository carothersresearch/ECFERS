import pygmo as pg
from src.fitting.pygmo_problems import *

def _ipy_bfe_func(all_dvs, prob):
    # The function that will be invoked
    # by the individual processes/nodes of mp/ipy bfe.
    import pickle

    n_items = len(all_dvs.shape)
    if n_items > 1:
        fs = list(map(prob.fitness, all_dvs))
    else:
        fs = prob.fitness(all_dvs)
    return pickle.dumps(fs)

class pickleless_bfe(pg.ipyparallel_bfe):
    def __init__(self, client_kwargs, view_kwargs, prob):
        self.client_kwargs = client_kwargs
        self.view_kwargs = view_kwargs
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
                pg.ipyparallel_bfe._view.push({"prob": self.prob}, block = True)
                pg.ipyparallel_bfe._view.apply_sync(lambda x: x.extract(SBMLGlobalFit_Multi_Fly)._setup_rr(), Reference("prob"))

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

        pg.ipyparallel_bfe._view.scatter("all_dvs", dvs, block = True)

        with pg.ipyparallel_bfe._view_lock:
            if pg.ipyparallel_bfe._view is None:
                pg.ipyparallel_bfe._view = self.init_view(self.client_kwargs, self.view_kwargs)

            ar = pg.ipyparallel_bfe._view.apply(_ipy_bfe_func, ipp.Reference("all_dvs"), ipp.Reference("prob"))
            ret = ipp.AsyncMapResult(pg.ipyparallel_bfe._view.client, ar._children, ipp.client.map.Map())
            
        # Build the vector of fitness vectors as a 2D numpy array.
        fvs = np.array(sum([pickle.loads(fv) for fv in ret.get()],[]))
        # Reshape it so that it is 1D.
        fvs.shape = (ndvs*nf,)

        return fvs