
import pandas as pd
import numpy as np
import tellurium as te
from odbm.odbm_main import ModelBuilder
import matplotlib.pyplot as plt

def rxn_plot(model:ModelBuilder, sim, rxn_idx, figsize = None, titles = None):
    """
    Plots the substrate and product concentrations over time for the specified reactions 

    Parameters
    ----------
    model : ModelBuilder
    sim : NamedArray
        simulation results
    rxn_idx : list,
        list of indices (ints) corresponding to reactions in model definition to plot
    figsize : tuple, optional, by default None
    titles : list, optional, by default None

    Returns
    -------
    figure, axis

    """
    if figsize is None:
        figsize = (len(rxn_idx),3)

    f,ax = plt.subplots(1, len(rxn_idx), figsize = figsize, sharey=False)
    for k,r in enumerate(rxn_idx):
        for j in model.get_substrates(id = r):
            if j.upper() in sim.colnames: # this needs to be format, not just upper
                #if species is not in simulation output, it is a boundary species
                ax[k].plot(sim['time']/60,sim[j], label = j)
            else:
                #assumes boundary species are defined with a "$", plots horizontal line
                boundary_species = float(model.species[model.species['Label'] == '$'+j]['StartingConc'])
                ax[k].plot([0,(sim['time']/60)[-1]], [boundary_species, boundary_species], label = j)

        for j in model.get_products(r):
            if j != '':
                if j.upper() in sim.colnames:
                    #if species is not in simulation output, it is a boundary species
                    ax[k].plot(sim['time']/60,sim[j],'--', label = j)
                else:
                    pass
                    # #assumes boundary species are defined with a "$", plots horizontal line
                    # boundary_species = float(model.species[model.species['Label'] == '$'+j]['StartingConc'])
                    # ax[k].plot([0,(sim['time']/60)[-1]], [boundary_species,boundary_species], '--', label = j)


        ax[k].legend()
        if titles: ax[k].set_title(titles[k])

        ax[k].set_xlabel('time (min)')
        ax[k].set_ylabel('Concentraion (mM)')
    f.tight_layout()
    return f, ax

def species_plot(model:ModelBuilder, sim, species, figsize = None, colors = None, markers = None, linestyles = None):
    """
    Plots the specified species concentrations over time 

    Parameters
    ----------
    model : ModelBuilder
    sim : NamedArray
        simulation results
    species : list
        list of species to plot
    figsize : tuple, optional, by default None

    Returns
    -------
    figure, axis

    """
    if figsize is None:
        figsize = (5,3)
    if colors is None:
        colors = [None]*len(species)
    if markers is None:
        markers = [None]*len(species)
    if linestyles is None:
        linestyles = [None]*len(species)

    f,ax = plt.subplots(1,1,figsize = figsize)
    for k,(s,c,m,l) in enumerate(zip(species, colors, markers, linestyles)):
        if '['+s+']'.upper() in sim.colnames: # this needs to be format, not just upper
            #if species is not in simulation output, it is a boundary species
            ax.plot(sim['time']/60,sim['['+s+']'], color = c, marker = m, linestyle = l)
        else:
            #assumes boundary species are defined with a "$", plots horizontal line
            boundary_species = float(model.species[model.species['Label'] == '$'+s]['StartingConc'])
            ax.plot([0,(sim['time']/60)[-1]], [boundary_species, boundary_species], color = c, marker = m, linestyle = l)

        ax.set_xlabel('time (min)')
        ax.set_ylabel('Concentraion (mM)')
    f.tight_layout()
    return f, ax