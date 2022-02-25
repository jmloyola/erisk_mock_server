"""
Mock server for the eRisk challenge (https://erisk.irlab.org/).
Copyright (C) 2022 Juan Martín Loyola

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License version 3 as
published by the Free Software Foundation.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""


import os
import json
import sqlite3
import datetime
from enum import Enum
from typing import List, Optional
from random import choice
from io import BytesIO

from fastapi import FastAPI, status, HTTPException, Query
from databases import Database
from pydantic import BaseModel, create_model
from starlette.responses import StreamingResponse, HTMLResponse
import matplotlib.pyplot as plt
import arviz as az
import numpy as np
import pandas as pd

from performance_measures import erde_final, f_latency, value_p, precision_at_k, ndcg
import config


# Global variable to store the users' writings.
WRITINGS = {task_name: [] for task_name in config.challenges_list}
# Global variable to store the true label of the users.
SUBJECTS = {task_name: {} for task_name in config.challenges_list}
# Global variable to store the median number of posts from the datasets.
MEDIAN_NUMBER_POSTS = {task_name: None for task_name in config.challenges_list}
# Dictionary with the path to the datasets.
DATASET_PATHS = {
    task_name: os.path.join(
        config.dataset_root_path, f"{task_name}{config.dataset_file_suffix}"
    )
    for task_name in config.challenges_list
}


# Since by default sqlite3 does not check FOREIGN KEYS, set that option when
# connecting to the database.
# Create a subclass of the sqlite3.Connection class to execute the pragma
# https://github.com/encode/databases/issues/169#issuecomment-644816412
#
# Also, to avoid being locked of the database by an operation that takes too
# long, set the `busy_timeout` pragma. This is related to the timeout parameter
# from sqlite3 in Python.
# Since the package databases does not passes any options to database, these
# need to be set using pragmas when possible.
# The timeout parameter specifies how long the connection should wait for the
# lock to go away until raising an exception.
# See
#   https://github.com/python/cpython/blob/b2077117d125925210148294eefee28797b7ff4c/Modules/_sqlite/connection.c#L212
#   https://www.sqlite.org/pragma.html#pragma_busy_timeout
class Connection(sqlite3.Connection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.execute("""PRAGMA foreign_keys=ON""")
        self.execute("""PRAGMA busy_timeout=120000""")


app = FastAPI()
database = Database(f"sqlite:///database/{config.database_name}.db", factory=Connection)


TaskName = Enum(
    "TaskName", {task_name: task_name for task_name in config.challenges_list}
)


# Data Models
class TeamBase(BaseModel):
    # Since the column team_id is an INTEGER PRIMARY KEY, sqlite automatically
    # set its value incrementally if not indicated. Thus, It isn't necessary to
    # include it in the model.
    # team_id: int
    name: str
    token: str
    number_runs: int


class TeamIn(TeamBase):
    pass


class TeamOut(TeamBase):
    team_id: int


class UsersWritings(BaseModel):
    id: int
    number: int
    nick: str
    redditor: int
    title: str
    content: str
    date: str


class ResponseData(BaseModel):
    nick: str
    decision: int
    score: float


class BaseExperimentResult(BaseModel):
    team_id: int
    task_id: int
    run_id: int
    erde_5: float
    erde_50: float
    f_latency: float


structure = {}
for i in config.chosen_delays_for_ranking:
    structure[f"precision_at_10_{i}"] = (float, ...)
    structure[f"ndcg_at_10_{i}"] = (float, ...)
    structure[f"ndcg_at_100_{i}"] = (float, ...)
ExperimentResult = create_model(
    "ExperimentResult",
    **structure,
    __base__=BaseExperimentResult,
)


CREATE_TABLE_TEAMS = """
    CREATE TABLE IF NOT EXISTS teams(
        team_id INTEGER PRIMARY KEY,
        name VARCHAR(40) UNIQUE,
        token VARCHAR(42) UNIQUE,
        number_runs INTEGER
    );"""
CREATE_TABLE_TASK = """
    CREATE TABLE IF NOT EXISTS tasks(
        task_id INTEGER PRIMARY KEY,
        task VARCHAR(20) UNIQUE
    );"""
CREATE_TABLE_RUNS_STATUS = """
    CREATE TABLE IF NOT EXISTS runs_status(
        team_id INTEGER,
        task_id INTEGER,
        current_post_number INTEGER,
        has_finished INTEGER,
        PRIMARY KEY (team_id, task_id),
        FOREIGN KEY (team_id) REFERENCES teams(team_id),
        FOREIGN KEY (task_id) REFERENCES tasks(task_id)
    );"""
CREATE_TABLE_GET_WRITINGS_REQUESTS = """
    CREATE TABLE IF NOT EXISTS get_writings_requests(
        team_id INTEGER,
        task_id INTEGER,
        current_post_number INTEGER,
        request_time REAL,
        PRIMARY KEY (team_id, task_id, current_post_number),
        FOREIGN KEY (team_id) REFERENCES teams(team_id),
        FOREIGN KEY (task_id) REFERENCES tasks(task_id)
    );"""
CREATE_TABLE_RESPONSES = """
    CREATE TABLE IF NOT EXISTS responses(
        team_id INTEGER,
        task_id INTEGER,
        run_id INTEGER,
        current_post_number INTEGER,
        json_response BLOB,
        response_time REAL,
        PRIMARY KEY (team_id, task_id, run_id, current_post_number),
        FOREIGN KEY (team_id) REFERENCES teams(team_id),
        FOREIGN KEY (task_id) REFERENCES tasks(task_id)
    );"""
CREATE_TABLE_RESULTS = (
    """
    CREATE TABLE IF NOT EXISTS results(
        team_id INTEGER,
        task_id INTEGER,
        run_id INTEGER,
        erde_5 REAL,
        erde_50 REAL,
        f_latency REAL,
    """
    + "".join(
        [
            f"precision_at_10_{i} REAL,\nndcg_at_10_{i} REAL,\nndcg_at_100_{i} REAL,\n"
            for i in config.chosen_delays_for_ranking
        ]
    )
    + """
        PRIMARY KEY (team_id, task_id, run_id),
        FOREIGN KEY (team_id) REFERENCES teams(team_id),
        FOREIGN KEY (task_id) REFERENCES tasks(task_id)
    );"""
)
CREATE_TABLE_GET_WRITINGS_SERVER_TIMES = """
    CREATE TABLE IF NOT EXISTS get_writings_server_times(
        team_id INTEGER,
        task_id INTEGER,
        current_post_number INTEGER,
        start_time REAL,
        elapsed_time REAL,
        PRIMARY KEY (team_id, task_id, current_post_number),
        FOREIGN KEY (team_id) REFERENCES teams(team_id),
        FOREIGN KEY (task_id) REFERENCES tasks(task_id)
    );"""
CREATE_TABLE_RESPONSES_SERVER_TIMES = """
    CREATE TABLE IF NOT EXISTS responses_server_times(
        team_id INTEGER,
        task_id INTEGER,
        run_id INTEGER,
        current_post_number INTEGER,
        start_time REAL,
        elapsed_time REAL,
        PRIMARY KEY (team_id, task_id, run_id, current_post_number),
        FOREIGN KEY (team_id) REFERENCES teams(team_id),
        FOREIGN KEY (task_id) REFERENCES tasks(task_id)
    );"""


TABLES_CREATION_QUERIES = [
    CREATE_TABLE_TEAMS,
    CREATE_TABLE_TASK,
    CREATE_TABLE_RUNS_STATUS,
    CREATE_TABLE_GET_WRITINGS_REQUESTS,
    CREATE_TABLE_RESPONSES,
    CREATE_TABLE_RESULTS,
    CREATE_TABLE_GET_WRITINGS_SERVER_TIMES,
    CREATE_TABLE_RESPONSES_SERVER_TIMES,
]


async def get_task_id(task: TaskName):
    """Get the task_id."""
    query = """SELECT task_id FROM tasks WHERE task=:task"""
    result = await database.fetch_one(query=query, values={"task": task.value})
    # This will always return a task_id.
    return result["task_id"]


async def get_team_information(token: str):
    """Get the team id, name and number of runs of the team with the given token."""
    query = """SELECT team_id, name, number_runs FROM teams WHERE token=:token"""
    result = await database.fetch_one(query=query, values={"token": token})
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Invalid token '{token}'."
        )
    team_id = result["team_id"]
    name = result["name"]
    number_runs = result["number_runs"]
    return team_id, name, number_runs


@app.on_event("startup")
async def startup():
    # If the folder for the database does not exists, create it.
    os.makedirs("database", exist_ok=True)

    await database.connect()

    result = await database.fetch_one(query="""PRAGMA foreign_keys""")
    if result[0]:
        print("PRAGMA foreign_keys is set. FOREIGN KEYs will be checked.")
    else:
        print("PRAGMA foreign_keys is not set. FOREIGN KEYs will not be checked.")

    result = await database.fetch_one(query="""PRAGMA busy_timeout""")
    print(f"PRAGMA busy_timeout = {result[0]} milliseconds.")

    print(
        "Checking if there exist any tables. If not, create all of them and initialize some."
    )
    query = """SELECT COUNT(*) as number_tables FROM sqlite_schema WHERE name='teams'"""
    result = await database.fetch_one(query=query)
    if result["number_tables"] == 0:
        print(
            "The database is empty. Creating the tables and initializing some of them."
        )
        for q in TABLES_CREATION_QUERIES:
            await database.execute(query=q)

        # Insert the tasks for the eRisk challenge.
        query = """INSERT INTO tasks(task) VALUES (:task)"""
        values = [{"task": t.value} for t in TaskName]
        await database.execute_many(query=query, values=values)

        # Insert random teams.
        values = [
            {"name": "UNSL", "token": "777", "number_runs": 5},
            {"name": "CONICET", "token": "333", "number_runs": 4},
            {"name": "IMASL", "token": "444", "number_runs": 1},
        ]
        for v in values:
            t = TeamIn(**v)
            await create_team(t)
    print("Initializing the SUBJECTS and WRITINGS dictionaries.")
    for t in TaskName:
        load_writings(t)


def median(num_posts):
    """Median of the numbers' list."""
    num_posts.sort()
    m = len(num_posts) // 2
    if (len(num_posts) % 2) == 0:
        return (num_posts[m - 1] + num_posts[m]) / 2
    else:
        return num_posts[m]


def load_writings(task: TaskName):
    """Load the users writings from a file."""
    global WRITINGS, SUBJECTS, DATASET_PATHS, MEDIAN_NUMBER_POSTS
    writings = WRITINGS[task.value]
    subjects = SUBJECTS[task.value]
    dataset_path = DATASET_PATHS[task.value]

    with open(dataset_path, "r", encoding="utf-8") as f:
        j = 0
        for i, line in enumerate(f):
            subject_id = f"subject{i}"
            label, document = line.split(maxsplit=1)
            label = 1 if label == "positive" else 0

            subject_writings = document.split(config.end_of_post_token)

            subjects[subject_id] = {
                "label": label,
                "num_posts": len(subject_writings),
            }

            for w_idx, writing in enumerate(subject_writings):
                if len(writings) <= w_idx:
                    writings.append([])
                writings[w_idx].append(
                    {
                        "id": j,
                        "number": w_idx,
                        "nick": subject_id,
                        "redditor": i,
                        "title": "",
                        "content": writing,
                        "date": "",
                    }
                )
                j += 1
    print(f"The number of users for {task.value} is {len(subjects)}.")
    print(f"The maximum number of posts for {task.value} is {len(writings)}.")

    # Get median number of posts.
    num_posts = [v["num_posts"] for k, v in subjects.items()]
    MEDIAN_NUMBER_POSTS[task.value] = median(num_posts)
    print(
        f"The median number of posts for {task.value} is {MEDIAN_NUMBER_POSTS[task.value]}."
    )


@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()


@app.get(
    "/teams/list",
    tags=["teams"],
    response_description="The list of registered teams.",
    response_model=List[TeamOut],
)
async def get_all_teams():
    """
    Get the list of teams registered.

    Example curl command:
    ```bash
    curl -X GET "localhost:8000/teams/list"
    ```
    """
    query = """SELECT * FROM teams"""
    results = await database.fetch_all(query=query)

    result_list = [TeamOut(**r) for r in results]
    return result_list


@app.get(
    "/teams/finished/{task}",
    tags=["teams"],
    response_description="List the teams that have finished the task.",
    response_model=List[TeamOut],
)
async def get_all_finished_teams(task: TaskName):
    """
    Get the list of teams that have finished the task.

    Example curl commands:
    ```bash
    curl -X GET "localhost:8000/teams/finished/gambling"
    ```
    """
    query = """
        SELECT teams.*
        FROM teams, runs_status, tasks
        WHERE teams.team_id=runs_status.team_id AND
              tasks.task=:task AND
              runs_status.task_id=tasks.task_id AND
              runs_status.has_finished=1
    """
    results = await database.fetch_all(query=query, values={"task": task.value})

    result_list = [TeamOut(**r) for r in results] if results is not None else []
    return result_list


@app.post(
    "/teams/new",
    status_code=status.HTTP_200_OK,
    tags=["teams"],
    response_description="The registered team.",
    response_model=TeamOut,
)
async def create_team(team: TeamIn):
    """
    Register a new team.

    Example curl command:
    ```bash
    curl -X POST -H  "accept: application/json" -H "Content-Type: application/json" \
        -d '{"name":"TESTING", "token":"1234", "number_runs":1}' localhost:8000/teams/new
    ```
    """
    query = """INSERT INTO teams(name, token, number_runs) VALUES (:name, :token, :number_runs)"""
    values = team.dict()

    try:
        await database.execute(query=query, values=values)
    except sqlite3.IntegrityError as e:
        print(e)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"The team {team} is already in the database.",
        )
    else:
        # If the team creation is successful, initialize the run status.
        # For that, obtain the id of the tasks and the id of the team.
        query = """SELECT task_id FROM tasks"""
        results = await database.fetch_all(query=query)
        task_id_list = [i["task_id"] for i in results]

        query = """SELECT * FROM teams WHERE token=:token"""
        result = await database.fetch_one(query=query, values={"token": team.token})
        team_out = TeamOut(**result)
        team_id = result["team_id"]

        for task_id in task_id_list:
            # Initialize the current_post_number for each run of the team to -1.
            query = """
                INSERT INTO runs_status(team_id, task_id, current_post_number, has_finished)
                VALUES (:team_id, :task_id, :current_post_number, :has_finished)
            """
            values = {
                "team_id": team_id,
                "task_id": task_id,
                "current_post_number": -1,
                "has_finished": 0,
            }
            await database.execute(query=query, values=values)
    return team_out


@app.get(
    "/teams/{token}",
    status_code=status.HTTP_200_OK,
    tags=["teams"],
    response_description="The information of the team.",
    response_model=TeamOut,
)
async def get_team(token: str):
    """
    Get the information of the team with given token.

    Example curl command:
    ```bash
    curl -X GET "localhost:8000/teams/777"
    ```
    """
    query = """SELECT * FROM teams WHERE token=:token"""
    result = await database.fetch_one(query=query, values={"token": token})

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"The team with token {token} is not in the database.",
        )
    else:
        return TeamOut(**result)


@app.get(
    "/{task}/getwritings/{token}",
    status_code=status.HTTP_200_OK,
    tags=["challenge"],
    response_description="The current users writings for the team.",
    response_model=List[UsersWritings],
)
async def get_writings(task: TaskName, token: str):
    """
    Get the current users writings for the given task and team.

    For this, first get the task_id of the task.
    Then, validate the team token. In case it is correct, get the team_id.
    Get the current number of posts that the team_id needs for task_id.
    Check if the teams has sent the response for the previous time step.
    Get the users posts.
    Update the runs_status table with the new current post number
    Finally, return the list of users posts. In case there is no users with that
    number of posts, the endpoint returns an empty list.

    Example curl commands:
    ```bash
    curl -X GET "localhost:8000/gambling/getwritings/777"
    ```
    """
    start_timestamp = datetime.datetime.now().timestamp()

    # Get the task_id.
    task_id = await get_task_id(task)

    # Get the team_id and number of runs.
    team_id, _, number_runs = await get_team_information(token)

    # Get the current number of post.
    query = """SELECT current_post_number FROM runs_status WHERE team_id=:team_id AND task_id=:task_id"""
    result = await database.fetch_one(
        query=query, values={"team_id": team_id, "task_id": task_id}
    )
    # Since, when a team is created the status of the runs is updated, every
    # team should have this information. If not, it indicates that the
    # database was modified outside of the system.
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"The tuple team_id and task_id ({team_id}, "
            f"{task_id}) has no run information. This "
            "means that the database was altered outside"
            "this program.",
        )
    current_post_number = result["current_post_number"]
    writings = WRITINGS[task.value]

    # If the current post number is equal to the number of writings for the task, it means that the team has
    # ended processing the input.
    if current_post_number == len(writings):
        print(f"The team {team_id} has ended processing the writings for {task.value}.")
        return []

    is_last_response_complete = True
    # Check if the team has sent the last responses before giving a new one.
    if current_post_number != -1:
        query = """
            SELECT run_id
            FROM responses
            WHERE team_id=:team_id AND
                  task_id=:task_id AND
                  current_post_number=:current_post_number
        """
        values = {
            "team_id": team_id,
            "task_id": task_id,
            "current_post_number": current_post_number,
        }
        results = await database.fetch_all(query=query, values=values)

        # When fetch_all is empty the result is not None, instead it is an empty list.
        # It isn't necessary to check if results is None.
        if len(results) != number_runs:
            is_last_response_complete = False
    new_post_number = (
        current_post_number + 1 if is_last_response_complete else current_post_number
    )

    if new_post_number < len(writings):
        response = writings[new_post_number]
    else:
        response = []

    # Update the current_post_number value for the (team_id, task_id) if a new set of writings was given.
    if is_last_response_complete:
        # If the team just ended processing all the writings, set a flag in the database.
        just_finished = int(new_post_number == len(writings))
        if just_finished:
            print(
                f"The team {team_id} has just ended processing the writings for {task.value}."
            )
        else:
            print(
                f"A new set of writings was given to team {team_id} for {task.value}."
            )
            # Insert the time of the request in the database.
            end_timestamp = datetime.datetime.now().timestamp()
            query = """
                INSERT INTO get_writings_requests(
                    team_id, task_id, current_post_number, request_time
                )
                VALUES (:team_id, :task_id, :current_post_number, :request_time)
            """
            values = {
                "team_id": team_id,
                "task_id": task_id,
                "current_post_number": new_post_number,
                "request_time": end_timestamp,
            }
            await database.execute(query=query, values=values)

            # Save the time it took the server to process the request.
            query = """
                INSERT INTO get_writings_server_times(
                    team_id, task_id, current_post_number, start_time, elapsed_time
                )
                VALUES (:team_id, :task_id, :current_post_number, :start_time, :elapsed_time)
            """
            values = {
                "team_id": team_id,
                "task_id": task_id,
                "current_post_number": new_post_number,
                "start_time": start_timestamp,
                "elapsed_time": end_timestamp - start_timestamp,
            }
            await database.execute(query=query, values=values)
        query = """
            UPDATE runs_status
            SET current_post_number=:current_post_number, has_finished=:has_finished
            WHERE team_id=:team_id AND task_id=:task_id;
        """
        values = {
            "team_id": team_id,
            "task_id": task_id,
            "current_post_number": new_post_number,
            "has_finished": just_finished,
        }
        await database.execute(query=query, values=values)
    else:
        print(
            "No new data was retrieved since the last run from team "
            f"{team_id} on task {task.value} wasn't completed."
        )

    return response


@app.get(
    "/calculate_results/{task}/{token}",
    status_code=status.HTTP_200_OK,
    tags=["results"],
    response_description="Calculate the performance results for a team.",
)
async def calculate_results(task: TaskName, token: str):
    """Calculate the performance results for a team."""
    global SUBJECTS, MEDIAN_NUMBER_POSTS
    subjects = SUBJECTS[task.value]
    median_number_post = MEDIAN_NUMBER_POSTS[task.value]

    # Get the task_id.
    task_id = await get_task_id(task)

    # Get the team_id and number of runs.
    team_id, _, number_runs = await get_team_information(token)

    # Check if the team has ended processing the input.
    query = """
        SELECT has_finished
        FROM runs_status
        WHERE team_id=:team_id AND
              task_id=:task_id
    """
    result = await database.fetch_one(
        query=query, values={"team_id": team_id, "task_id": task_id}
    )
    has_finished = result["has_finished"]

    if not has_finished:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"The team {team_id} has not finished processing all the input for "
            f"{task.value}.",
        )

    for i in range(number_runs):
        internal_run_id = i + 1
        query = """
            SELECT json_response, current_post_number
            FROM responses
            WHERE team_id=:team_id AND
                task_id=:task_id AND
                run_id=:run_id
        """
        values = {
            "team_id": team_id,
            "task_id": task_id,
            "run_id": internal_run_id,
        }

        subjects_predictions = {}
        async for row in database.iterate(query=query, values=values):
            encoded_json_response = row["json_response"]
            json_response = decode_bytes_response(encoded_json_response)
            current_post_number = row["current_post_number"]
            # Since the API externally starts counting from 1, sum one.
            current_post_number = current_post_number + 1

            for response_data in json_response:
                if response_data.nick not in subjects_predictions:
                    # Initialize the subject label as negative.
                    subjects_predictions[response_data.nick] = {
                        "label": 0,
                    }

                # When the user is first classified as positive, set the corresponding label and delay.
                if (response_data.decision == 1) and (
                    subjects_predictions[response_data.nick]["label"] == 0
                ):
                    subjects_predictions[response_data.nick]["label"] = 1
                    subjects_predictions[response_data.nick][
                        "delay"
                    ] = current_post_number

                # If the user has not been label as positive and her posts are finished,
                # set the corresponding delay.
                if (subjects_predictions[response_data.nick]["label"] == 0) and (
                    subjects[response_data.nick]["num_posts"] == current_post_number
                ):
                    subjects_predictions[response_data.nick]["label"] = 0
                    subjects_predictions[response_data.nick][
                        "delay"
                    ] = current_post_number

                if current_post_number in config.chosen_delays_for_ranking:
                    subjects_predictions[response_data.nick][
                        f"score_{current_post_number}"
                    ] = response_data.score
        predictions = []
        true_labels = []
        delays = []
        scores_dict = {f"score_{j}": [] for j in config.chosen_delays_for_ranking}
        for nick in subjects.keys():
            predictions.append(subjects_predictions[nick]["label"])
            true_labels.append(subjects[nick]["label"])
            delays.append(subjects_predictions[nick]["delay"])
            for j in config.chosen_delays_for_ranking:
                scores_dict[f"score_{j}"].append(
                    subjects_predictions[nick][f"score_{j}"]
                )

        precision_at_10 = [
            precision_at_k(scores=scores_dict[f"score_{j}"], y_true=true_labels, k=10)
            for j in config.chosen_delays_for_ranking
        ]
        ndcg_10 = [
            ndcg(scores=scores_dict[f"score_{j}"], y_true=true_labels, p=10)
            for j in config.chosen_delays_for_ranking
        ]
        ndcg_100 = [
            ndcg(scores=scores_dict[f"score_{j}"], y_true=true_labels, p=100)
            for j in config.chosen_delays_for_ranking
        ]

        c_fp = sum(true_labels) / len(true_labels)
        erde_5 = erde_final(
            labels_list=predictions,
            true_labels_list=true_labels,
            delay_list=delays,
            c_fp=c_fp,
            o=5,
        )
        erde_50 = erde_final(
            labels_list=predictions,
            true_labels_list=true_labels,
            delay_list=delays,
            c_fp=c_fp,
            o=50,
        )
        p = value_p(k=median_number_post)
        f_latency_result = f_latency(
            labels=predictions, true_labels=true_labels, delays=delays, penalty=p
        )

        # Insert the results in the database.
        query = (
            """
            INSERT INTO results(team_id, task_id, run_id, erde_5, erde_50, f_latency"""
            + "".join(
                [
                    f", precision_at_10_{i}, ndcg_at_10_{i}, ndcg_at_100_{i}"
                    for i in config.chosen_delays_for_ranking
                ]
            )
            + """)
            VALUES (:team_id, :task_id, :run_id, :erde_5, :erde_50, :f_latency"""
            + "".join(
                [
                    f", :precision_at_10_{i}, :ndcg_at_10_{i}, :ndcg_at_100_{i}"
                    for i in config.chosen_delays_for_ranking
                ]
            )
            + ")"
        )
        values = {
            "team_id": team_id,
            "task_id": task_id,
            "run_id": internal_run_id,
            "erde_5": erde_5,
            "erde_50": erde_50,
            "f_latency": f_latency_result,
        }
        for j, delay in enumerate(config.chosen_delays_for_ranking):
            values[f"precision_at_10_{delay}"] = precision_at_10[j]
            values[f"ndcg_at_10_{delay}"] = ndcg_10[j]
            values[f"ndcg_at_100_{delay}"] = ndcg_100[j]
        await database.execute(query=query, values=values)
    return {"message": "The function `calculate_results` ended successfully."}


@app.get(
    "/results/{task}/all",
    tags=["results"],
    response_description="The results of all finished experiments.",
    response_model=List[ExperimentResult],
)
async def get_all_results(task: TaskName):
    """
    Get the list of all the results for the given task.

    Example curl commands:
    ```bash
    curl -X GET "localhost:8000/results/gambling/all"
    ```
    """
    # Get the task_id.
    task_id = await get_task_id(task)

    query = """SELECT * FROM results WHERE task_id=:task_id"""
    results = await database.fetch_all(query=query, values={"task_id": task_id})

    result_list = [ExperimentResult(**r) for r in results]
    return result_list


@app.get(
    "/results/{task}/{token}",
    status_code=status.HTTP_200_OK,
    tags=["results"],
    response_description="Team's results.",
    response_model=List[ExperimentResult],
)
async def get_team_results(task: TaskName, token: str):
    """
    Get the list of results for the given team and task.

    Example curl commands:
    ```bash
    curl -X GET "localhost:8000/results/gambling/1234"
    ```
    """
    # Get the task_id.
    task_id = await get_task_id(task)

    # Get the team_id and number of runs.
    team_id, _, number_runs = await get_team_information(token)

    query = """SELECT * FROM results WHERE team_id=:team_id AND task_id=:task_id"""
    results = await database.fetch_all(
        query=query, values={"team_id": team_id, "task_id": task_id}
    )
    # When fetch_all is empty the result is not None, instead it is an empty list.
    if results == []:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"The team with token '{token}' has not yet finished processing the input.",
        )
    result_list = [ExperimentResult(**r) for r in results]
    assert len(result_list) == number_runs
    if len(result_list) != number_runs:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"The team with token '{token}' has results for some of its runs. This means that "
            "the database was modified outside this program.",
        )
    return result_list


def encode_response_as_bytes(response: List[ResponseData]):
    """Encode team response as bytes."""
    r = [data.dict() for data in response]
    return json.dumps(r).encode("utf-8")


def decode_bytes_response(encoded_response: bytes):
    """Decode team response as a list of ResponseData."""
    r = json.loads(encoded_response)
    response = [ResponseData(**data) for data in r]
    return response


def check_if_response_complete(response: List[ResponseData], task: TaskName):
    """Check if the response has an entry for every user for the task."""
    subjects = SUBJECTS[task.value]

    number_subjects_in_response = 0
    subjects_already_counted = []
    for r in response:
        r = r.dict()
        nick = r["nick"]
        if nick in subjects:
            if nick not in subjects_already_counted:
                number_subjects_in_response += 1
                subjects_already_counted.append(nick)
            else:
                print(
                    f'The response has multiple entries for nick "{nick}" in {task.value}.'
                )
        else:
            print(
                f'The nick "{nick}" does not correspond to a subject in {task.value}.'
            )
    return number_subjects_in_response == len(subjects)


@app.post(
    "/{task}/submit/{token}/{run_id}",
    status_code=status.HTTP_200_OK,
    tags=["challenge"],
    response_description="Run responses.",
    response_model=List[ResponseData],
)
async def post_response(
    task: TaskName, token: str, run_id: int, response: List[ResponseData]
):
    """
    Post the response of the team and run for the selected task.

    Note that internally the run_id has values from 1 to infinite, but the API
    for eRisk starts at 0.
    The mapping was made internally.

    Example curl command:
    ```bash
    curl -X POST -H  "accept: application/json" -H "Content-Type: application/json" \
        -d '[{"nick":"subject0", "decision":1, "score":1.2}, {"nick":"subject1", "decision":0, "score":0.2}]' \
        localhost:8000/gambling/submit/777/0
    ```
    """
    start_timestamp = datetime.datetime.now().timestamp()

    # Get the task_id.
    task_id = await get_task_id(task)

    # Get the team_id and number of runs.
    team_id, _, number_runs = await get_team_information(token)

    # Increment the run_id value to map from API to database.
    internal_run_id = run_id + 1

    if not (1 <= internal_run_id <= number_runs):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Invalid run_id {run_id} for token '{token}'.",
        )

    # Get the current_post_number.
    query = """
        SELECT current_post_number, has_finished
        FROM runs_status
        WHERE team_id=:team_id AND
              task_id=:task_id
    """
    result = await database.fetch_one(
        query=query, values={"team_id": team_id, "task_id": task_id}
    )
    current_post_number = result["current_post_number"]
    has_finished = result["has_finished"]

    # If the team has already finished sending responses for the task, don't
    # do anything.
    if has_finished:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"The team {team_id} has already finished sending all the responses for "
            f"{task.value}.",
        )

    # If the team has not asked for writings, they can't send a response.
    if current_post_number == -1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"The team {team_id} does not have any writings yet for {task.value}.",
        )

    # Check if this run has sent responses for that post number.
    query = """
        SELECT 1
        FROM responses
        WHERE team_id=:team_id AND
              task_id=:task_id AND
              run_id=:run_id AND
              current_post_number=:current_post_number
    """
    values = {
        "team_id": team_id,
        "task_id": task_id,
        "run_id": internal_run_id,
        "current_post_number": current_post_number,
    }
    result = await database.fetch_one(query=query, values=values)

    if result is None:
        # There is no response yet.
        is_response_complete = check_if_response_complete(response, task)
        if is_response_complete:
            # Insert the value in the database.
            end_timestamp = datetime.datetime.now().timestamp()
            query = """
                INSERT INTO responses(
                    team_id, task_id, run_id, current_post_number, json_response, response_time
                )
                VALUES (:team_id, :task_id, :run_id, :current_post_number, :json_response, :response_time)
            """
            values = {
                "team_id": team_id,
                "task_id": task_id,
                "run_id": internal_run_id,
                "current_post_number": current_post_number,
                "json_response": encode_response_as_bytes(response),
                "response_time": end_timestamp,
            }
            await database.execute(query=query, values=values)

            # Save the time it took the server to process the request.
            query = """
                INSERT INTO responses_server_times(
                    team_id, task_id, run_id, current_post_number, start_time, elapsed_time
                )
                VALUES (:team_id, :task_id, :run_id, :current_post_number, :start_time, :elapsed_time)
            """
            values = {
                "team_id": team_id,
                "task_id": task_id,
                "run_id": internal_run_id,
                "current_post_number": current_post_number,
                "start_time": start_timestamp,
                "elapsed_time": end_timestamp - start_timestamp,
            }
            await database.execute(query=query, values=values)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"The current response for {task.value} is not complete.",
            )
    else:
        # There is a complete response already. Don't do anything with the request.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"The team {team_id} has already sent a response for "
            f"{task.value} for the current post ({current_post_number}).",
        )
    return response


@app.get(
    "/graph/{task}/separation_plot/{token}",
    status_code=status.HTTP_200_OK,
    tags=["graphs"],
    response_description="Graph the team's models final separation plots.",
)
async def graph_final_separation_plot(task: TaskName, token: str):
    """
    Graph the final separation plot of a team for a task.

    Example curl command:
    ```bash
    curl -X GET "localhost:8000/graph/depression/separation_plot/777" --output "elapsed_time_team.png"
    ```
    """
    global WRITINGS
    writings = WRITINGS[task.value]
    max_number_posts = len(writings)

    return await graph_separation_plot(task, token, max_number_posts)


@app.get(
    "/graph/{task}/separation_plot/{token}/{time}",
    status_code=status.HTTP_200_OK,
    tags=["graphs"],
    response_description="Graph the team's models separation plots at a given time.",
)
async def graph_separation_plot(task: TaskName, token: str, time: int):
    """
    Graph the separation plot of a team for a task in a given time.

    Example curl command:
    ```bash
    curl -X GET "localhost:8000/graph/depression/separation_plot/777/30" --output "elapsed_time_team_at_time.png"
    ```
    """
    global SUBJECTS, WRITINGS
    subjects = SUBJECTS[task.value]
    writings = WRITINGS[task.value]
    max_number_posts = len(writings)

    # Get the team information.
    team_id, name, number_runs = await get_team_information(token)

    # Get the task_id.
    task_id = await get_task_id(task)

    # Check if the time given is valid.
    if time > max_number_posts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"The task {task.value} has {max_number_posts} posts at most. Thus, the given time "
            f"({time}) is not valid.",
        )
    is_final_separtion_plot = time == max_number_posts

    true_labels = [subjects[nick]["label"] for nick in subjects.keys()]
    scores = []
    for i in range(number_runs):
        internal_run_id = i + 1
        query = """
            SELECT json_response
            FROM responses
            WHERE team_id=:team_id AND
                task_id=:task_id AND
                run_id=:run_id AND
                current_post_number=:current_post_number
        """
        values = {
            "team_id": team_id,
            "task_id": task_id,
            "run_id": internal_run_id,
            "current_post_number": time - 1,
        }

        subjects_scores = {}
        async for row in database.iterate(query=query, values=values):
            encoded_json_response = row["json_response"]
            json_response = decode_bytes_response(encoded_json_response)

            for response_data in json_response:
                subjects_scores[response_data.nick] = response_data.score
        # If there was no data for that run_id, raise an error.
        if not subjects_scores:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"The team with token '{token}' has not sent the response for {task.value} time "
                f"{time} for the run {i}.",
            )
        run_scores = [subjects_scores[nick] for nick in subjects.keys()]
        scores.append(run_scores)

    # Graph the results.
    true_labels = np.asarray(true_labels)
    scores = np.asarray(scores)
    fig, ax = plt.subplots(nrows=number_runs, ncols=1)
    for i in range(number_runs):
        az.plot_separation(
            y=true_labels, y_hat=scores[i], y_hat_line=True, legend=False, ax=ax[i]
        )
        ax[i].set_ylabel(f"#{i}", fontsize=10)

    figure_title = (
        f"Final separation plot - {task.value}\n{name}"
        if is_final_separtion_plot
        else f"Separation plot at time {time} - {task.value}\n{name}"
    )
    fig.suptitle(figure_title, fontsize=17)

    # Create a buffer to store image data.
    buf = BytesIO()
    fig.savefig(buf, dpi=300, format="png")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")


async def get_model_response_for_user(
    team_id: int, task: TaskName, internal_run_id: int, user_id: str
):
    """Get the score, label and delay from a particular run of a team given to a user."""
    global SUBJECTS
    subjects = SUBJECTS[task.value]

    # Get the task_id.
    task_id = await get_task_id(task)

    query = """
        SELECT json_response, current_post_number
        FROM responses
        WHERE team_id=:team_id AND
            task_id=:task_id AND
            run_id=:run_id
    """
    values = {
        "team_id": team_id,
        "task_id": task_id,
        "run_id": internal_run_id,
    }

    scores = []
    label = 0
    delay = -1
    finished_processing_user_posts = False
    async for row in database.iterate(query=query, values=values):
        encoded_json_response = row["json_response"]
        json_response = decode_bytes_response(encoded_json_response)
        current_post_number = row["current_post_number"]
        # Since the API externally starts counting from 1, sum one.
        current_post_number = current_post_number + 1

        for response_data in json_response:
            if response_data.nick == user_id:
                scores.append(response_data.score)
                # When the user is first classified as positive, set the corresponding label and delay.
                if (response_data.decision == 1) and (label == 0):
                    label = 1
                    delay = current_post_number

                # If the user has not been label as positive and her posts are finished,
                # set the corresponding delay.
                if (label == 0) and (
                    subjects[response_data.nick]["num_posts"] == current_post_number
                ):
                    label = 0
                    delay = current_post_number
                    finished_processing_user_posts = True
                # When the user is found, it isn't necessary to look for the others.
                break
        if finished_processing_user_posts:
            break
    # If there was no data for that run_id, raise an error.
    if not scores:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"The team has not sent any response for {task.value} for the run {internal_run_id}.",
        )
    return scores, label, delay


@app.get(
    "/graph/{task}/random_user/{token}/{run_id}",
    status_code=status.HTTP_200_OK,
    tags=["graphs"],
    response_description="Graph the model's score given to a random user.",
)
async def graph_score_random_user(task: TaskName, token: str, run_id: int):
    """
    Graph a random user's score evolution of a team's run for a task.

    Example curl command:
    ```bash
    curl -X GET "localhost:8000/graph/depression/random_user/777/1" --output "score_random_user.png"
    ```
    """
    global SUBJECTS
    subjects = SUBJECTS[task.value]

    # Get a random user.
    users_nicks = [k for k in subjects.keys()]
    user_id = choice(users_nicks)

    return await graph_score_user(task, user_id, token, run_id)


@app.get(
    "/graph/{task}/{user_id}/{token}/{run_id}",
    status_code=status.HTTP_200_OK,
    tags=["graphs"],
    response_description="Graph the model's score given to a user.",
)
async def graph_score_user(task: TaskName, user_id: str, token: str, run_id: int):
    """
    Graph a user's score evolution of a team's run for a task.

    Example curl command:
    ```bash
    curl -X GET "localhost:8000/graph/depression/30/777/1" --output "score_user.png"
    ```
    """
    global SUBJECTS
    subjects = SUBJECTS[task.value]
    team_id, name, number_runs = await get_team_information(token)

    # Increment the run_id value to map from API to database.
    internal_run_id = run_id + 1
    # Check if the number of run is valid.
    if not (1 <= internal_run_id <= number_runs):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Invalid run_id {run_id} for token '{token}'.",
        )

    true_label = subjects[user_id]["label"]
    num_posts = subjects[user_id]["num_posts"]

    scores, label, delay = await get_model_response_for_user(
        team_id, task, internal_run_id, user_id
    )

    # Graph the results.
    fig, ax = plt.subplots(nrows=1, ncols=1)
    x = list(range(1, num_posts + 1))
    ax.plot(x, scores[:num_posts])
    c = "green" if true_label == label else "red"
    ax.axvline(
        x=delay,
        color=c,
        linestyle="dashed",
        ymin=0,
        ymax=1,
        label=f"decision delay = {delay}",
        alpha=0.8,
    )
    ax.legend()
    ax.set_title(f'Score for user "{user_id}" in {task.value}\n{name} - run: {run_id}')
    ax.set_xlabel("Number of publications")
    ax.set_ylabel("Model score")

    # Create a buffer to store image data.
    buf = BytesIO()
    fig.savefig(buf, dpi=300, format="png")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")


@app.get(
    "/graph/{task}/teams_elapsed_time",
    status_code=status.HTTP_200_OK,
    tags=["graphs"],
    response_description="Graph the elapsed time from all the teams that had finished processing the input.",
)
async def graph_teams_elapsed_time(task: TaskName):
    """
    Graph the elapsed time from all the teams that had finished processing the input.

    Example curl command:
    ```bash
    curl -X GET "localhost:8000/graph/depression/teams_elapsed_time" --output "teams_elapsed_time.png"
    ```
    """
    global WRITINGS
    writings = WRITINGS[task.value]
    max_number_posts = len(writings)

    # Get the task_id.
    task_id = await get_task_id(task)

    # Get the teams that had finished processing the input.
    finished_teams = await get_all_finished_teams(task)

    # If there is no team that have finished processing the input, raise an error.
    if not (finished_teams):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Currently, there is no team that have finished processing the input for {task.value}.",
        )

    # Get the difference of elapsed times from the last POST and the first GET
    # for every team that have finished processing the input.
    # Note that it was necessary to "hack" the way the teams that had finished
    # were indicated in the SQL query.
    query = f"""
        SELECT t.team_id, name, (last_post_time - first_get_time) as elapsed_time
        FROM teams as t, (
            SELECT g.team_id, request_time as first_get_time, MAX(response_time) as last_post_time
            FROM get_writings_requests as g, responses as r
            WHERE g.team_id=r.team_id AND
                g.task_id=r.task_id AND
                g.task_id=:task_id AND
                g.current_post_number=0 AND
                r.current_post_number=:last_post_number AND
                g.team_id IN ({', '.join([":team"+str(i) for i in range(len(finished_teams))])})
            GROUP BY g.team_id
        ) as e
        WHERE t.team_id=e.team_id
    """
    finished_teams_dict = {
        "team" + str(i): finished_teams[i].team_id for i in range(len(finished_teams))
    }
    values = {
        "task_id": task_id,
        "last_post_number": max_number_posts - 1,
    }
    values = {**values, **finished_teams_dict}

    results = await database.fetch_all(query=query, values=values)
    names = [i["name"] for i in results]
    elapsed_times = [r["elapsed_time"] for r in results]

    # Graph the results.
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.bar(names, elapsed_times)

    fig.suptitle(
        f"Elapsed time from teams that have finished\n{task.value}", fontsize=17
    )
    ax.set_xlabel("Teams name")
    ax.set_ylabel("Total time (seconds)")

    # Create a buffer to store image data.
    buf = BytesIO()
    fig.savefig(buf, dpi=300, format="png")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")


@app.get(
    "/graph/{task}/elapsed_time/{token}",
    status_code=status.HTTP_200_OK,
    tags=["graphs"],
    response_description="Graph the elapsed time from all the selected runs of a team.",
)
async def graph_runs_elapsed_time(
    task: TaskName, token: str, run: Optional[List[int]] = Query(None)
):
    """
    Graph the elapsed time from all the selected runs of a team.

    If `run is None`, that is, no run ids are given, graph the elapsed
    time of all the team's runs.

    If you want to indicate the run ids, you can do it using query parameters.
    That is, at the end of the endpoint add the values of run id you want to
    graph.
    If you want to graph the elapsed time of runs 0 and 2, you would use:
    `?run=0&run=2` at the end of the URL.

    Example curl command:
    ```bash
    curl -X GET "localhost:8000/graph/depression/elapsed_time/777?run=1&run=2" --output "runs_elapsed_time.png"
    ```
    """
    # Get the task_id.
    task_id = await get_task_id(task)

    # Check if the given team has finished processing the input.
    query = """
        SELECT teams.team_id, name, teams.number_runs
        FROM teams, runs_status
        WHERE teams.team_id=runs_status.team_id AND
              teams.token=:token AND
              runs_status.task_id=:task_id AND
              runs_status.has_finished=1
    """
    values = {
        "token": token,
        "task_id": task_id,
    }
    result = await database.fetch_one(query=query, values=values)

    # If the team didn't finished processing the input, raise an error.
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"The team with token '{token}' have not yet finished processing the input for {task.value}.",
        )
    team_id = result["team_id"]
    team_name = result["name"]
    number_runs = result["number_runs"]

    # If there is list of run ids, check if they are valid.
    # If any of them is invalid, raise an error.
    # Also check if any of the run ids are repeated.
    if run:
        if len(set(run)) != len(run):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="There is a repeated run_id in the request.",
            )
        for run_id in run:
            if not (0 <= run_id < number_runs):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Invalid run_id {run_id} for token '{token}'.",
                )
    else:
        run = list(range(number_runs))

    # Increment the run_id value to map from API to database.
    runs_list = [x + 1 for x in run]

    # Get the difference of elapsed times from each POST and each GET.
    # Note that it was necessary to "hack" the way the teams that had finished
    # were indicated in the SQL query.
    query = f"""
        SELECT (response_time - request_time) as elapsed_time
        FROM get_writings_requests as g, responses as r
        WHERE g.team_id=r.team_id AND
            g.team_id=:team_id AND
            g.task_id=r.task_id AND
            g.task_id=:task_id AND
            g.current_post_number=r.current_post_number AND
            r.run_id IN ({', '.join([":run"+str(i) for i in runs_list])})
        ORDER BY r.run_id ASC, r.current_post_number ASC
    """

    runs_dict = {"run" + str(run_id): run_id for run_id in runs_list}
    values = {
        "team_id": team_id,
        "task_id": task_id,
    }
    values = {**values, **runs_dict}

    results = await database.fetch_all(query=query, values=values)
    elapsed_times = [r["elapsed_time"] for r in results]

    # Since the query results were ordered, the results can divided in chunks
    # corresponding to each run.
    elapsed_times_runs = []
    final_pos = 0
    len_each_list = len(elapsed_times) // len(runs_list)
    for _ in runs_list:
        initial_pos = final_pos
        final_pos += len_each_list
        elapsed_times_runs.append(elapsed_times[initial_pos:final_pos])

    # Graph the results.
    fig, ax = plt.subplots(nrows=1, ncols=1)
    x = list(range(1, len(elapsed_times_runs[0]) + 1))
    for idx, values in enumerate(elapsed_times_runs):
        run_id = runs_list[idx] - 1
        ax.plot(x, values, alpha=0.6, label=f"run = {run_id}")
    ax.legend()
    ax.set_xlabel("Number of publication")
    ax.set_ylabel("Time (seconds)")

    fig.suptitle(f"Runs elapsed time from team {team_name}\n{task.value}", fontsize=17)

    # Create a buffer to store image data.
    buf = BytesIO()
    fig.savefig(buf, dpi=300, format="png")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")


@app.get(
    "/graph/{task}/results",
    tags=["graphs"],
    status_code=status.HTTP_200_OK,
    response_description="Table with the results of all finished experiments.",
    response_class=HTMLResponse,
)
async def graph_all_results(task: TaskName):
    """
    Get the table with the results of all finished experiments.

    Example curl commands:
    ```bash
    curl -X GET "localhost:8000/graph/depression/results" --output "results_table.html"
    ```
    """
    # Get the task_id.
    task_id = await get_task_id(task)

    query = """
        SELECT t.name, r.*
        FROM teams as t, results as r
        WHERE t.team_id=r.team_id AND
            task_id=:task_id
        ORDER BY r.team_id ASC, r.run_id ASC
    """
    results = await database.fetch_all(query=query, values={"task_id": task_id})

    # Get the table columns names.
    query = """PRAGMA table_info(results)"""
    table_columns = await database.fetch_all(query=query)
    table_columns = ["name"] + [t[1] for t in table_columns]

    df = pd.DataFrame(results, columns=table_columns)
    # Format the real numbers using two decimals, show a subset of the columns and
    # do not show the (empty) index.
    # The columns omitted are team_id (index 1) and task_id (index 2).
    selected_colums = table_columns[:1] + table_columns[3:]
    html_table = df.to_html(
        columns=selected_colums, float_format=lambda x: f"{x:.2f}", index=False
    )

    html_content = f"""
    <html>
        <head>
            <title>Table with the results of all finished experiments for {task.value}.</title>
        </head>
        <body>
            <h1>Table with the results of all finished experiments for {task.value}.</h1>
            {html_table}
        </body>
    </html>
    """

    return HTMLResponse(content=html_content, status_code=200)
