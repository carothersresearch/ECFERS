FROM mcr.microsoft.com/devcontainers/anaconda:0-3

RUN apt-get update 

# Update environment
RUN conda install -n base conda-libmamba-solver
RUN conda config --set solver libmamba

COPY environment.yml .
RUN conda env create -f environment.yml
RUN echo "source activate myenv" > ~/.bashrc
ENV PATH /opt/conda/envs/myenv/bin:$PATH

SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]
RUN pip install git+https://gitlab.com/wurssb/Modelling/symbolicsbml.git#egg=symbolicSBML
RUN pip install git+https://gitlab.com/wurssb/Modelling/parameter-balancer.git#egg=parameter-balancer

# Copy everything else in folder
# COPY . /usr/src/ECFERS

RUN python -m ipykernel install --user --name=myenv 
# might need to run this after docker inits

# ENTRYPOINT ["python", "-m ipykernel install --user --name=myenv"]

# command to build image
# docker build -f .devcontainer/Dockerfile --tag=dalbabur/ecfers:latest .