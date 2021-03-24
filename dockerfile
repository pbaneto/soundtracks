FROM continuumio/miniconda3

COPY thesis_env.yml .
RUN conda env create -f thesis_env.yml

# The code to run when container is started:
COPY run.py .
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "thesis_env", "python", "code/load_data.py"]


