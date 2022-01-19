"""
Simple client to consume the mock server API.
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


import argparse
import httpx
import asyncio
import sys


GET_TIMEOUT_LIMIT = 240
POST_TIMEOUT_LIMIT = 180
NUMBER_RETRIES = 5


def get_users_nicknames(json_data):
    """Get the users nicknames from the first call to the get writings endpoint."""
    if int(json_data[0]["number"]) != 0:
        raise Exception(
            "ERROR: The function `get_users_nicknames` should have been called the first time you asked "
            "for writings."
        )
    users_nicknames = []
    for user_data in json_data:
        users_nicknames.append(user_data["nick"])
    return users_nicknames


def random_team_response(json_file):
    """Generate a random response for every user."""
    response = []
    for subject_dict in json_file:
        d = {"nick": subject_dict["nick"], "decision": 1, "score": 1.0}
        response.append(d)
    return response


async def create_new_team(base_url, team_data):
    """Create a new team. If it already exists, exit from the script with error."""
    print("Creating a new team.")
    async with httpx.AsyncClient(base_url=base_url) as client:
        response = await client.post("/teams/new", json=team_data)
        if response.status_code == 200:
            print(f"The new team stored information is: {response.json()}.")
        else:
            print(
                f"ERROR: The team ({team_data}) already exists in the database. Either create a new team, or "
                "delete the database."
            )
            sys.exit()


async def get_writings(base_url, server_task, team_token):
    """Get the current users writings."""
    print("Getting the current users writings.")
    request_status_code = 400
    response = None
    number_tries = 0
    while request_status_code != 200 and number_tries < NUMBER_RETRIES:
        try:
            async with httpx.AsyncClient(base_url=base_url) as client:
                response = await client.get(
                    f"/{server_task}/getwritings/{team_token}",
                    timeout=GET_TIMEOUT_LIMIT,
                )
            request_status_code = response.status_code
        except httpx.TimeoutException:
            print(f"WARNING: The request took longer than {GET_TIMEOUT_LIMIT} seconds.")
            request_status_code = 408
        except httpx.ConnectError:
            print(
                "WARNING: Connection Error. It might be that the maximum number of retries with the URL has "
                "been exceeded."
            )
            request_status_code = 429

        if request_status_code != 200:
            print(
                f"WARNING: The request failed, trying again. Current attempt number: {number_tries + 1}."
            )
            number_tries += 1
            await asyncio.sleep(5)
    return response, request_status_code


async def post_team_responses(base_url, server_task, team_token, team_runs, json_data):
    """Post the response for all the team's runs."""
    responses = await asyncio.gather(
        *[
            post_response(base_url, server_task, team_token, i, json_data)
            for i in range(team_runs)
        ]
    )
    # Get the status code of all the POSTs.
    responses_status_code = [r[1] for r in responses]
    return responses_status_code


async def post_response(base_url, server_task, team_token, run_id, json_data):
    """Post the current run response."""
    print("Posting the current run response.")
    request_status_code = 400
    response = None
    number_tries = 0
    while request_status_code != 200 and number_tries < NUMBER_RETRIES:
        try:
            async with httpx.AsyncClient(base_url=base_url) as client:
                response = await client.post(
                    f"/{server_task}/submit/{team_token}/{str(run_id)}",
                    json=json_data,
                    timeout=POST_TIMEOUT_LIMIT,
                )
            request_status_code = response.status_code
        except httpx.TimeoutException:
            print(
                f"WARNING: The request took longer than {POST_TIMEOUT_LIMIT} seconds."
            )
            request_status_code = 408
        except httpx.ConnectError:
            print(
                "WARNING: Connection Error. It might be that the maximum number of retries with the URL has "
                "been exceeded."
            )
            request_status_code = 429

        if request_status_code != 200:
            print(
                f"WARNING: The request failed, trying again. Current attempt number: {number_tries + 1}."
            )
            number_tries += 1
            await asyncio.sleep(5)
    return response, request_status_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to connect to the eRisk mock server to get\
        writings and to send the responses."
    )
    parser.add_argument("-n", "--team_name", help="team name")
    parser.add_argument("-t", "--team_token", help="team token")
    parser.add_argument("-r", "--team_runs", help="team number of runs", type=int)
    parser.add_argument(
        "-s",
        "--server_task",
        help="challenge task to solve",
        choices=["gambling", "depression"],
        default="depression",
    )
    parser.add_argument("-p", "--port", help="mock server port", type=int, default=8000)
    parser.add_argument(
        "-k",
        "--number_posts",
        help="number of post you want to process before stopping the script",
        type=int,
        default=30,
    )
    args = parser.parse_args()

    if args.team_token is None or args.team_name is None or args.team_runs is None:
        print(
            "ERROR: You should specify all the options to run the script. For information on how to call the "
            "script, run `python -m client --help`."
        )
        sys.exit()

    base_url = f"http://localhost:{args.port}"
    print(
        f"Connecting to the mock server for the task {args.server_task} at {base_url}."
    )

    team_data = {
        "name": args.team_name,
        "token": args.team_token,
        "number_runs": args.team_runs,
    }

    # Create a new team
    asyncio.run(create_new_team(base_url, team_data))

    last_json_response = None
    # Get the user writings and post the classification results after every post.
    # When we get the first post of every users, we have to save a list of the users' nicknames.
    get_response, status_code = asyncio.run(
        get_writings(base_url, args.server_task, args.team_token)
    )

    if status_code != 200:
        print("ERROR: GET request failed. Aborting script.")
        sys.exit()

    new_json_response = get_response.json()

    USERS_NICKNAMES = get_users_nicknames(json_data=new_json_response)

    initial_response_number = int(new_json_response[0]["number"])
    current_response_number = initial_response_number

    # Generate a random response for every user.
    model_response = random_team_response(new_json_response)
    while (
        new_json_response != last_json_response
        and (current_response_number - initial_response_number) < args.number_posts
    ):
        print(f">> Post number being processed: {current_response_number + 1}.")
        responses_status_code = asyncio.run(
            post_team_responses(
                base_url,
                args.server_task,
                args.team_token,
                args.team_runs,
                model_response,
            )
        )
        responses_status_are_200 = [r == 200 for r in responses_status_code]
        if not all(responses_status_are_200):
            print("ERROR: At least one of the POST requests failed. Aborting script.")
            sys.exit()

        last_json_response = new_json_response

        get_response, status_code = asyncio.run(
            get_writings(base_url, args.server_task, args.team_token)
        )

        if status_code != 200:
            print("ERROR: GET request failed. Aborting script.")
            sys.exit()

        new_json_response = get_response.json()

        if not new_json_response:
            print("No more posts to process.")
            break

        assert (int(new_json_response[0]["number"]) == current_response_number + 1) or (
            new_json_response == last_json_response
        )
        current_response_number = int(new_json_response[0]["number"])

    if (
        new_json_response
        and new_json_response != last_json_response
        and (current_response_number - initial_response_number) == args.number_posts
    ):
        print(f"Reached the number of posts limit ({args.number_posts} posts).")

    print("End of script.")
