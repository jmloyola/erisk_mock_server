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

## Docker image
If you want to use a Docker image, you can follow these steps.

Build the docker image with:
```bash
docker build --tag mock_server_image .
```

Note that if you changed the location of the datasets, you will have to change
the path in the `Dockerfile`.

To check the available docker images, use:
```bash
docker image ls
```

In case you want to remove the created image you can use:
```bash
docker image rm mock_server_image
```

Finally, to run the mock server with a persistent database, use:
```bash
docker run --detach --name mock_server --publish 8000:8000 --volume mock_server_db:/app/database/ mock_server_image
```

The `--detach` option makes the container run in the background. If you want to
see the terminal output directly, you can remove it.

Another way to look at the logs of the container when using the `--detach` option
is to use:
```bash
docker logs mock_server
```

To check the status of all the containers, use:
```bash
docker ps --all
```

To check the resources used by the container, use:
```bash
docker stats mock_server
```

To stop the container, use:
```bash
docker stop --time 2 mock_server
```

To remove the container, use:
```bash
docker rm mock_server
```

In case a container failed (it is stopped now) and you want to run it again, you will have to
remove it. You can remove with the previous commando or, you can prune all stopped instances with:
```bash
docker container prune
```
