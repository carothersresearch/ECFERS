import os
import tellurium as te
import numpy as np
import pygmo as pg
from copy import deepcopy

class TimeoutError(Exception):
    pass

import signal
import time
class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)

class SBMLGlobalFit:

    def __init__(self, model, data, parameter_labels, lower_bounds, upper_bounds, settings, variables = {}, scale = False):
        self.model = model
        self.data = data
        self.parameter_labels = parameter_labels # only the ones that are going to be fitted
        self.settings = settings
        self.upperb = upper_bounds
        self.lowerb = lower_bounds
        self.scale = scale
        self.variables = variables # dict of labels and values

    def fitness(self, x):
        if self.scale: x = self._unscale(x)
        r, results = self._simulate(x)
        obj = self._residual(results,self.data)
        return [obj]
            
    def _simulate(self, x):
        from roadrunner import Config, RoadRunner
        Config.setValue(Config.ROADRUNNER_DISABLE_PYTHON_DYNAMIC_PROPERTIES, False)
        Config.setValue(Config.LOADSBMLOPTIONS_RECOMPILE, False) 
        Config.setValue(Config.LLJIT_OPTIMIZATION_LEVEL, 4)

        id = str(os.getpid())
        try:
            r = RoadRunner()
            r.loadState('/mmfs1/gscratch/cheme/dalba/repos/ECFERS/models/binaries/model_state_'+id+'.b')
        except Exception as e:
            print(e)
            r = te.loadSBMLModel(self.model)
            r.saveState('/mmfs1/gscratch/cheme/dalba/repos/ECFERS/models/binaries/model_state_'+id+'.b')

        # update parameters
        for label, value in zip(self.parameter_labels,x):
            try:
                r[label] = value
            except Exception as e:
                print(e)
                return r, self.data*(-np.inf)

        # set any variable
        for label, value in self.variables.items():
            try:
                r[label] = value
            except Exception as e:
                print(e)
                return r, self.data*(-np.inf)
        try:
            results = r.simulate(**self.settings['simulation'])[:,1:].__array__()
        except:
            results = self.data*(-np.inf)

        return r, results
    
    # def _residual(self,results,data,points):
    #     md = (np.nanmax(data,1,keepdims=True)-np.nanmin(data,1,keepdims=True))/2
    #     mr = (np.nanmax(results,1,keepdims=True)-np.nanmin(results,1,keepdims=True))/2
    #     denom = np.ones(data.shape)
    #     return [np.nansum(((data-results)/(points*(md**2+mr**2)**0.5*denom))**2)]

    def _residual(self,results,data):
        cols = self.settings['fit_to_cols']
        rows = self.settings['fit_to_rows']

        error = (data[:,cols][rows,:]-results[:,cols][rows,:])
        RMSE = np.sqrt(np.nansum(error**2, axis=0)/len(rows))
        NRMSE = RMSE/(np.nanmax(data[:,cols][rows,:], axis=0) - np.nanmin(data[:,cols][rows,:], axis=0) + 1e6)
        return np.nansum(NRMSE)
    
    def _unscale(self, x):
        unscaled = self.lowerb + (self.upperb - self.lowerb) * x
        unscaled[unscaled<0] = self.lowerb[unscaled<0]
        unscaled[unscaled>1] = self.lowerb[unscaled>1]
        return unscaled
    
    def _scale(self, x):
        return (x - self.lowerb) / (self.upperb - self.lowerb)

    def get_bounds(self):
        if self.scale:
            lowerb = [0 for i in self.lowerb]
            upperb = [1 for i in self.upperb]
        else:
            upperb = self.upperb
            lowerb = self.lowerb    
        return (lowerb, upperb)
    
    def get_name(self):
        return 'Global Fitting of Multiple SBML Models'

    def gradient(self, x):
        return pg.estimate_gradient_h(lambda x: self.fitness(x),x)


class SBMLGlobalFit_Constrained:

    def __init__(self, model, data, parameter_labels, lower_bounds, upper_bounds, settings, logKeq_r, variables = {}, scale = False, n_constraints = 0):
        self.model = model
        self.data = data
        self.parameter_labels = parameter_labels # only the ones that are going to be fitted
        self.settings = settings
        self.upperb = upper_bounds
        self.lowerb = lower_bounds
        self.scale = scale
        self.n_constraints = n_constraints
        self.variables = variables # dict of labels and values
        self.logKeq_r = logKeq_r

    def batch_fitness(self, xs):
        view = pg.ipyparallel_bfe().init_view(client_kwargs={'profile':'cheme-ecfers'})
        fs = view.map_sync(self.fitness,xs)
        del view
        return fs

    def fitness(self, x):
        if self.scale: x = self._unscale(x)
        r, results = self._simulate(x)
        obj = self._residual(results,self.data)
        if self.n_constraints > 0:
            constraints = self._haldane(r)
            return [obj, *constraints]
        else:
            constraints = self._haldane(r)
            return [obj + constraints/r.getNumReactions()/10e6]
            
    def _simulate(self, x):
        from roadrunner import Config, RoadRunner
        Config.setValue(Config.ROADRUNNER_DISABLE_PYTHON_DYNAMIC_PROPERTIES, False)
        Config.setValue(Config.LOADSBMLOPTIONS_RECOMPILE, False) 
        Config.setValue(Config.LLJIT_OPTIMIZATION_LEVEL, 4)

        id = str(os.getpid())
        try:
            r = RoadRunner()
            r.loadState('/mmfs1/gscratch/cheme/dalba/repos/ECFERS/models/binaries/model_state_'+id+'.b')
        except Exception as e:
            print(e)
            r = te.loadSBMLModel(self.model)
            r.saveState('/mmfs1/gscratch/cheme/dalba/repos/ECFERS/models/binaries/model_state_'+id+'.b')

        # r = te.loadSBMLModel(self.model)
        # update parameters
        for label, value in zip(self.parameter_labels,x):
            try:
                r[label] = value
            except Exception as e:
                print(e)
                return r, self.data*(-np.inf)

        # set any variable
        for label, value in self.variables.items():
            try:
                r[label] = value
            except Exception as e:
                print(e)
                return r, self.data*(-np.inf)
        try:
            results = r.simulate(**self.settings['simulation'])[:,1:].__array__()
        except:
            results = self.data*(-np.inf)

        return r, results
    
    # def _residual(self,results,data,points):
    #     md = (np.nanmax(data,1,keepdims=True)-np.nanmin(data,1,keepdims=True))/2
    #     mr = (np.nanmax(results,1,keepdims=True)-np.nanmin(results,1,keepdims=True))/2
    #     denom = np.ones(data.shape)
    #     return [np.nansum(((data-results)/(points*(md**2+mr**2)**0.5*denom))**2)]

    def _residual(self,results,data):
        cols = self.settings['fit_to_cols']
        rows = self.settings['fit_to_rows']

        error = (data[:,cols][rows,:]-results[:,cols][rows,:])
        RMSE = np.sqrt(np.nansum(error**2, axis=0)/len(rows))
        NRMSE = RMSE/(np.nanmax(data[:,cols][rows,:], axis=0) - np.nanmin(data[:,cols][rows,:], axis=0) + 1e6)
        return np.nansum(NRMSE)
    
    def _unscale(self, x):
        unscaled = self.lowerb + (self.upperb - self.lowerb) * x
        unscaled[unscaled<0] = self.lowerb[unscaled<0]
        unscaled[unscaled>1] = self.lowerb[unscaled>1]
        return unscaled
    
    def _scale(self, x):
        return (x - self.lowerb) / (self.upperb - self.lowerb)

    def _haldane(self, r):
        matrix = r.getFullStoichiometryMatrix();
        species_labels = np.array(matrix.rownames)
        metabolites_index = ['EC' not in label for label in species_labels]
        metabolites_labels = species_labels[metabolites_index]
        reaction_labels = np.array(matrix.colnames)
        stoich = matrix[metabolites_index,:]

        kms = np.zeros(shape=stoich.shape)
        for i in range(stoich.shape[0]):
            for j in range(stoich.shape[1]):
                m_id = metabolites_labels[i]
                r_id = reaction_labels[j]
                if stoich[i,j] != 0:
                    kms[i,j] = stoich[i,j]*np.log(r['Km_'+m_id+'_'+r_id])
        
        nlogKm_r = kms.sum(axis=0)
        logKcats_r = np.array([np.log(r['Kcat_F_'+r_id]/r['Kcat_R_'+r_id]) for r_id in r.getReactionIds()])
        logKeq_r = self.logKeq_r # may be a better way to pass this
        return np.sum(((logKeq_r-logKcats_r-nlogKm_r)/logKeq_r)**2)

    def get_bounds(self):
        if self.scale:
            lowerb = [0 for i in self.lowerb]
            upperb = [1 for i in self.upperb]
        else:
            upperb = self.upperb
            lowerb = self.lowerb    
        return (lowerb, upperb)
    
    def get_name(self):
        return 'Global Fitting of Multiple SBML Models'

    def gradient(self, x):
        return pg.estimate_gradient_h(lambda x: self.fitness(x),x)
    
    def get_nec(self):
        return self.n_constraints
    

class SBMLGlobalFit_Multi:

    def __init__(self, model, data, parameter_labels, lower_bounds, upper_bounds, metadata, variables = {}, scale = False):
        self.model = model
        self.path_to_models = model[:model.find('/models')]
        self.parameter_labels = parameter_labels # only the ones that are going to be fitted
        self.upperb = upper_bounds
        self.lowerb = lower_bounds
        self.scale = scale
        self.data = data
        self.metadata = metadata
        self.variables = variables # dict of labels and values
        self.cvode_timepoints = 1000

        r = te.loadSBMLModel(self.model)
        self.species_labels = np.array(r.getFullStoichiometryMatrix().rownames)
        self.global_parameter_labels = np.array(r.getGlobalParameterIds())
        self.cols = {}
        self.rows = {}
        self.data_cols = {}
        for s in self.metadata['sample_labels']:
            measurements = []
            self.data_cols[s] = []
            for i,m in enumerate(self.metadata['measurement_labels']):
                try:    # model may not contain measurement species
                    measurements.append(np.where(self.species_labels==m)[0][0])
                    self.data_cols[s].append(i)
                except:
                    pass
            self.cols[s] = measurements
            self.rows[s] = [np.where(np.linspace(0, self.metadata['timepoints'][s][-1], self.cvode_timepoints) < (t+0.1))[0][-1] for t in self.metadata['timepoints'][s]]


    def fitness(self, x):
        if self.scale: x = self._unscale(x)
        res_dict = self._simulate(x)
        obj = np.nanmean([self._residual(results, self.data[sample], sample) for sample, results in res_dict.items()])
        return [obj]
            
    def _simulate(self, x):
        from roadrunner import Config, RoadRunner
        Config.setValue(Config.ROADRUNNER_DISABLE_PYTHON_DYNAMIC_PROPERTIES, False)
        Config.setValue(Config.LOADSBMLOPTIONS_RECOMPILE, False) 
        Config.setValue(Config.LLJIT_OPTIMIZATION_LEVEL, 4)

        # load model
        id = str(os.getpid())
        if os.path.isfile(self.path_to_models+'/models/binaries/model_state_'+id+'.b'):
            with open(self.path_to_models+'/models/binaries/model_state_'+id+'.b', 'rb') as f:
                r = RoadRunner()
                r.loadStateS(f.read())
        else:
            r = te.loadSBMLModel(self.model)
            r.integrator.absolute_tolerance = 1e-8
            r.integrator.relative_tolerance = 1e-8
            r.integrator.maximum_num_steps = 2000
            with open(self.path_to_models+'/models/binaries/model_state_'+id+'.b', 'wb') as f:
                f.write(r.saveStateS())
       
        # set new parameters
        for l, v in zip(self.parameter_labels,x):
            if 'Km' in l:
                r.setValue('init('+l+')',v)
            else:
                r.setValue('init('+l+')',v)

        # set variables and simulate
        results = {sample:self.data[sample]*(-np.inf) for sample in self.metadata['sample_labels']}
        rb = r.saveStateS()
        for sample in self.metadata['sample_labels']:
            r2 = RoadRunner()
            r2.loadStateS(rb)
            # set any variable
            for label, value in self.variables[sample].items():
                if not np.isnan(value):
                    if label not in self.species_labels:
                        r2.setValue('init('+label+')', value) # we might need new assignment rules for heterologous enzymes
                    else:
                        if 'EC' in label:
                            r2.setValue('['+label+']', value*r2.getValue(self.parameter_labels[-1])) # this is a bit obtuse
                        else:
                        # r2.removeInitialAssignment(label) theres some bug here? it seems to also mess up with over valuesS
                            r2.setValue('['+label+']', value)
            try:
                results[sample] = r2.simulate(0,self.metadata['timepoints'][sample][-1],self.cvode_timepoints)[:,1:].__array__()
            except Exception as e:
                print(e)
                # break # stop if any fail
        return results
                
    def _residual(self,results,data,sample):
        cols = self.cols[sample]
        rows = self.rows[sample]
        dcols = self.data_cols[sample]
        
        if data.shape == results.shape:
            error = (data[:,dcols]-results[:,dcols])
        else:
            error = (data[:,dcols]-results[:,cols][rows,:])

        RMSE = np.sqrt(np.nansum(error**2, axis=0)/len(rows))
        NRMSE = RMSE/(np.nanmax(data[:,dcols], axis=0) - np.nanmin(data[:,dcols], axis=0) + 1e-6)
        return np.nansum(NRMSE)
    
    def _unscale(self, x):
        unscaled = self.lowerb + (self.upperb - self.lowerb) * x
        # unscaled[unscaled<0] = self.lowerb[unscaled<0]
        # unscaled[unscaled>1] = self.lowerb[unscaled>1]
        return unscaled
    
    def _scale(self, x):
        return (x - self.lowerb) / (self.upperb - self.lowerb)

    def get_bounds(self):
        if self.scale:
            lowerb = [0 for i in self.lowerb]
            upperb = [1 for i in self.upperb]
        else:
            upperb = self.upperb
            lowerb = self.lowerb    
        return (lowerb, upperb)
    
    def get_name(self):
        return 'Global Fitting of Multiple SBML Models'

class SBMLGlobalFit_Multi_Fly:
    class ModelStuff:
        def __init__(self, model, metadata, cvode_timepoints, parameter_labels, variables):
            r = te.loadSBMLModel(model)
            self.species_labels = np.array(r.getFullStoichiometryMatrix().rownames)
            self.r_parameter_labels = np.array(r.getGlobalParameterIds())
            self.parameter_order = np.int32(np.squeeze(np.array([np.where(p == self.r_parameter_labels) for p in parameter_labels if p in self.r_parameter_labels])))
            self.variable_order = {sample:np.int32(np.squeeze(np.array([np.where(p == self.r_parameter_labels) for p in var.keys() if p in self.r_parameter_labels]))) for sample,var in variables.items()}
            self.cols = {}
            self.rows = {}
            self.data_cols = {}
            for s in metadata['sample_labels']:
                measurements = []
                self.data_cols[s] = []
                for i,m in enumerate(metadata['measurement_labels']):
                    try:    # model may not contain measurement species
                        measurements.append(np.where(self.species_labels==m)[0][0])
                        self.data_cols[s].append(i)
                    except:
                        pass
                self.cols[s] = measurements
                self.rows[s] = [np.where(np.linspace(0, metadata['timepoints'][s][-1], cvode_timepoints) < (t+0.1))[0][-1] for t in metadata['timepoints'][s]]
            del r

    def __init__(self, model:list, data:list, parameter_labels, lower_bounds, upper_bounds, metadata:list, variables:list, scale = False):
        self.model = model # now a list of models
        self.parameter_labels = parameter_labels # all parameters across all models, only the ones that are going to be fitted
        self.set_bounds(upper_bounds, lower_bounds)
        self.scale = scale
        self.data = data # now a list of data
        self.metadata = metadata # now a list of metadas
        self.variables = variables # # now a list of dict of labels and values

        self.cvode_timepoints = 1000

        self.model_stuff = [self.ModelStuff(m, md, self.cvode_timepoints, self.parameter_labels, var) for m,md,var in zip(self.model, self.metadata, self.variables)]

    def fitness(self, x):
        if self.scale: x = self._unscale(x)
        res = self._simulate(x)
        obj = np.nansum([[self._residual(results, data[sample], sample, ms) for sample, results in resdict.items()] for data,ms,resdict in zip(self.data,self.model_stuff,res)])
        del res
        return [obj]
    
    def _setup_rr(self): # run on engine
        from roadrunner import Config, RoadRunner, Logger
        Logger.disableLogging()
        Config.setValue(Config.ROADRUNNER_DISABLE_PYTHON_DYNAMIC_PROPERTIES, True)
        Config.setValue(Config.LOADSBMLOPTIONS_RECOMPILE, False) 
        Config.setValue(Config.LLJIT_OPTIMIZATION_LEVEL, 4)
        Config.setValue(Config.LLVM_SYMBOL_CACHE, True)
        Config.setValue(Config.LOADSBMLOPTIONS_OPTIMIZE_GVN, True)
        Config.setValue(Config.LOADSBMLOPTIONS_OPTIMIZE_CFG_SIMPLIFICATION, True)
        Config.setValue(Config.LOADSBMLOPTIONS_OPTIMIZE_INSTRUCTION_COMBINING, True)
        Config.setValue(Config.LOADSBMLOPTIONS_OPTIMIZE_DEAD_INST_ELIMINATION, True)
        Config.setValue(Config.LOADSBMLOPTIONS_OPTIMIZE_DEAD_CODE_ELIMINATION, True)
        Config.setValue(Config.LOADSBMLOPTIONS_OPTIMIZE_INSTRUCTION_SIMPLIFIER, True)
        Config.setValue(Config.SIMULATEOPTIONS_COPY_RESULT, True)
        self.r = []
        for m in self.model:
            r = te.loadSBMLModel(m)
            r.integrator.absolute_tolerance = 1e-8
            r.integrator.relative_tolerance = 1e-8
            r.integrator.maximum_num_steps = 2000
            self.r.append(r)
            
    def _simulate(self, x):
        from roadrunner import Config, RoadRunner, Logger
        Logger.disableLogging()
        Config.setValue(Config.ROADRUNNER_DISABLE_PYTHON_DYNAMIC_PROPERTIES, True)
        Config.setValue(Config.LOADSBMLOPTIONS_RECOMPILE, False) 
        Config.setValue(Config.LLJIT_OPTIMIZATION_LEVEL, 4)
        Config.setValue(Config.LLVM_SYMBOL_CACHE, True)
        Config.setValue(Config.LOADSBMLOPTIONS_OPTIMIZE_GVN, True)
        Config.setValue(Config.LOADSBMLOPTIONS_OPTIMIZE_CFG_SIMPLIFICATION, True)
        Config.setValue(Config.LOADSBMLOPTIONS_OPTIMIZE_INSTRUCTION_COMBINING, True)
        Config.setValue(Config.LOADSBMLOPTIONS_OPTIMIZE_DEAD_INST_ELIMINATION, True)
        Config.setValue(Config.LOADSBMLOPTIONS_OPTIMIZE_DEAD_CODE_ELIMINATION, True)
        Config.setValue(Config.LOADSBMLOPTIONS_OPTIMIZE_INSTRUCTION_SIMPLIFIER, True)
        Config.setValue(Config.SIMULATEOPTIONS_COPY_RESULT, True)

        all_results = []
        for r,ms, data, metadata, variables in zip(self.r, self.model_stuff, self.data, self.metadata, self.variables):
            # this is suboptimal as it only needs to happen once...
            variables = {sample:{k:v for k,v in var.items() if k in ms.r_parameter_labels} for sample,var in variables.items()}

            results = {sample:data[sample]*(-np.inf) for sample in metadata['sample_labels']}
            for sample in metadata['sample_labels']:
                r.model.setGlobalParameterValues([*ms.parameter_order, *ms.variable_order[sample]], [*x, *variables[sample].values()])
                r.reset()
                try:
                    results[sample] = r.simulate(0,metadata['timepoints'][sample][-1],self.cvode_timepoints)[:,1:].__array__()
                except Exception as e:
                    print(e)
                    # break # stop if any fail
                r.resetToOrigin()
            all_results.append(results)
        del Config, RoadRunner, Logger, results
        return all_results
                
    def _residual(self,results,data,sample,modelstuff):
        cols = modelstuff.cols[sample]
        rows = modelstuff.rows[sample]
        dcols = modelstuff.data_cols[sample]
        
        if data.shape == results.shape:
            error = (data[:,dcols]-results[:,dcols])
            d_error = (np.diff(data[:,dcols],axis=0)-np.diff(results[:,dcols],axis=0))
        else:
            error = (data[:,dcols]-results[:,cols][rows,:])
            d_error = (np.diff(data[:,dcols],axis=0)-np.diff(results[:,cols][rows,:],axis=0))

        RMSE = np.concatenate([np.sqrt(np.nansum(error**2, axis=0)/np.count_nonzero(~np.isnan(error))), np.sqrt(np.nansum(d_error**2, axis=0)/np.count_nonzero(~np.isnan(d_error)))])
        # NRMSE = RMSE/(np.nanmax(data[:,dcols], axis=0) - np.nanmin(data[:,dcols], axis=0) + 1e-6)
        return RMSE
    
    def _unscale(self, x):
        unscaled = self.lowerb + (self.upperb - self.lowerb) * x
        # unscaled[unscaled<0] = self.lowerb[unscaled<0]
        # unscaled[unscaled>1] = self.lowerb[unscaled>1]
        return unscaled
    
    def _scale(self, x):
        return (x - self.lowerb) / (self.upperb - self.lowerb)

    def get_bounds(self):
        if self.scale:
            lowerb = [0 for i in self.lowerb]
            upperb = [1 for i in self.upperb]
        else:
            upperb = self.upperb
            lowerb = self.lowerb    
        return (lowerb, upperb)
    
    def get_name(self):
        return 'Global Fitting of Multiple SBML Models'
    
    def set_bounds(self, upper_bounds, lower_bounds):
        self.upperb = upper_bounds # for all parameters
        self.lowerb = lower_bounds # fol all parameters