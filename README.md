# eRisk mock server
Mock server for the [eRisk challenge](https://erisk.irlab.org/).

This server resembles the original eRisk server. It allows teams to:
    - Get writings for a given task.
    - Post the results of a run.

Besides those, the mock servers can:
    - Create new teams (used mainly to experiment with different models and parameters).
    - List the teams in the system.
    - List the teams that have finished processing the input.
    - Get information of a given team.
    - Automatically calculate time aware measures for the teams that have finished processing the input.
    - List the results for the teams that have finished.
    - Get the results of a given team.
    - Graph the final separation plot for a given team.
    - Graph separation plot for a given team and time.
    - Graph score evolution for a given user, team and run.
    - Graph score evolution of a random users for a given team and run.

## Environment set up
To set up the environment we used [miniconda](https://docs.conda.io/en/latest/miniconda.html).
Once you have miniconda installed, run:
```bash
conda env create -f environment.yml
```

## Project configuration
Before running the server, make sure to configure the server editing the `config.py` file.

## Corpus structure
We assume that each corpus has the following structure:
    - One line per user.
    - Each line is divided in two parts separated by a tab (`\t`).
        The first token represents the class the current user belongs to:
        _"positive"_ or _"negative"_. Then, all the user documents follow.
    - Each user document is separated by a special token declared in the `config.py` file.

## Run the server
To run the server use:
```bash
uvicorn mock_server:app --host 0.0.0.0 --port 8000
```
