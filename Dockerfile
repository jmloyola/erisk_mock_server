# Dockerfile based on this: https://pythonspeed.com/articles/activate-conda-dockerfile/
FROM continuumio/miniconda3

WORKDIR /app

# Create the environment.
COPY environment.yml .
RUN conda env create -f environment.yml

# Make RUN commands use the new environment.
SHELL ["conda", "run", "-n", "erisk_mock_server", "/bin/bash", "-c"]

# Demonstrate the environment is activated.
RUN echo "Make sure fastapi is installed:"
RUN python -c "import fastapi"

# Copy the dataset folder.
# If you have changed the path to the datasets, remember to change it here.
COPY datasets datasets

# Create the directory for the database.
RUN mkdir -p /app/database

# Copy the source code.
COPY mock_server.py .
COPY config.py .
COPY performance_measures.py .
# Command to run when container is started.
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "erisk_mock_server", "uvicorn", "mock_server:app", "--host", "0.0.0.0", "--port", "8000"]
