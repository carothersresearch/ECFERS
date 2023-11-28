FROM mcr.microsoft.com/devcontainers/anaconda:0-3

RUN apt-get update 

# Copy everything in folder
COPY . /usr/src/ECFERS
RUN ls -la /usr/src/ECFERS

# Update environment
RUN conda install -n base conda-libmamba-solver
RUN conda config --set solver libmamba
RUN conda env create -f /usr/src/ECFERS/environment.yml
RUN echo "source activate myenv" > ~/.bashrc
ENV PATH /opt/conda/envs/myenv/bin:$PATH

SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]
RUN pip install git+https://gitlab.com/wurssb/Modelling/symbolicsbml.git#egg=symbolicSBML
RUN pip install git+https://gitlab.com/wurssb/Modelling/parameter-balancer.git#egg=parameter-balancer

WORKDIR  /usr/src/ECFERS
RUN python -m ipykernel install --user --name=myenv 
# might need to run this after docker inits

# ENTRYPOINT ["python", "-m", "ipykernel", "install", "--user", "--name=myenv"]

