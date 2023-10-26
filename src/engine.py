import numpy as np

def updateRRParameters(rr, new_parameters: dict):

    # update kinetic parameters
    p_ids = rr.model.getGlobalParameterIds()
    p_values = rr.model.getGlobalParameterValues()
    new_p = [new_parameters[p] if p in new_parameters.keys() else p_values[i] for i,p in enumerate(p_ids)]
    rr.model.setGlobalParameterValues(new_p)

    # update species intial concentrations
    s_ids = rr.model.getFloatingSpeciesInitConcentrationIds()
    s_values = rr.model.getFloatingSpeciesInitConcentrations()
    new_s = [new_parameters[p] if p in new_parameters.keys() else s_values[i] for i,p in enumerate(s_ids)]
    rr.model.setFloatingSpeciesInitConcentrations(new_s)

    return

def simulateRR(rr, start = 0, end = 10, points = 50, selections = None):
    try:
        if selections: rr.timeCourseSelections(selections)
        result = rr.simulate(start,end, points)
    except Exception as e:
        print(e)
        result = None
    return result

def runRR(rr, parameters: dict, options = {}):
    rr.resetToOrigin()
    updateRRParameters(rr, parameters)
    result = simulateRR(rr, **options)
    return result

def transformResults(RRresults, data_to_fit):
    # some function that gets the simulated species and time points we are interested in based on the data
    # data to fit is a np matrix  
    pass

def lossFunction(transformed_results, data_to_fit, method = 'MAE', method_options = {}):
    residuals = (data_to_fit - transformed_results)/data_to_fit

    if method == 'MAE':
        loss = np.sum(np.abs(residuals))

    elif method == 'MSE':
        loss = np.sum((residuals) ** 2)

    elif method == 'Huber':
        delta = method_options['delta']
        huber_mse = 0.5*(residuals)**2
        huber_mae = delta * (np.abs(residuals) - 0.5 * delta)
        loss = np.sum(np.where(np.abs(residuals) <= delta, huber_mse,  huber_mae))

    elif method == 'Pseudo-Huber':
        delta = method_options['delta']
        loss = delta**2 * ((1+(residuals/delta)**2)**0.5 -1)

    else:
        Exception(method + ' not implemented.')
    
    return loss

def fit(rr, parameters, data_to_fit, method):
    pass