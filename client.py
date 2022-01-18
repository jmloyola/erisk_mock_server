import argparse
import httpx
import asyncio
import sys
import time


GET_TIMEOUT_LIMIT = 240
POST_TIMEOUT_LIMIT = 180
NUMBER_RETRIES = 5


def get_users_nicknames(json_data):
    """Get the users nicknames from the first call to the get writings endpoint."""
    if int(json_data[0]["number"]) != 0:
        raise Exception(
            "The function `get_users_nicknames` should have been called the first time you asked "
            "for writings."
        )
    users_nicknames = []
    for user_data in json_data:
        users_nicknames.append(user_data["nick"])
    return users_nicknames


def random_team_response(json_file):
    response = []
    for subject_dict in json_file:
        d = {"nick": subject_dict["nick"], "decision": 1, "score": 1.0}
        response.append(d)
    return response


def request_retry(url, kind="get", json_data=None):
    request_status_code = 400
    r = None
    while request_status_code != 200:
        try:
            if kind == "get":
                r = httpx.get(url, timeout=GET_TIMEOUT_LIMIT)
            elif kind == "post":
                r = httpx.post(url, json=json_data, timeout=POST_TIMEOUT_LIMIT)
            else:
                print("This functions only accepts GET or POST requests")
                return None
            request_status_code = r.status_code
        except httpx.exceptions.Timeout:
            print(
                f"The request took longer than "
                f'{GET_TIMEOUT_LIMIT if kind == "get" else POST_TIMEOUT_LIMIT} seconds'
            )
            request_status_code = 408
        except httpx.exceptions.ConnectionError:
            print(
                "Connection Error. It might be that the maximum number of retries with the URL has "
                "been exceeded"
            )
            request_status_code = 429

        if request_status_code != 200:
            print("The request failed, trying again...")
            time.sleep(5)
    return r, request_status_code


async def create_new_team(base_url, team_data):
    """Create the team. If it already exists, exit from the script with error."""
    print("Creating a new team.")
    async with httpx.AsyncClient(base_url=base_url) as client:
        response = await client.post("/teams/new", json=team_data)
        if response.status_code == 200:
            print(f"The POST response is: {response.json()}.")
        else:
            print(
                f"The team ({team_data}) already exists in the database. Either create a new team, or "
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
            print(f"The request took longer than {GET_TIMEOUT_LIMIT} seconds")
            request_status_code = 408
        except httpx.ConnectError:
            print(
                "Connection Error. It might be that the maximum number of retries with the URL has "
                "been exceeded"
            )
            request_status_code = 429

        if request_status_code != 200:
            print("The request failed, trying again...")
            number_tries += 1
            time.sleep(5)
    return response, request_status_code


async def post_team_responses(base_url, server_task, team_token, team_runs, json_data):
    """Post the response for all the team's runs."""
    _ = await asyncio.gather(
        *[
            post_response(base_url, server_task, team_token, i, json_data)
            for i in range(team_runs)
        ]
    )


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
            print(f"The request took longer than {POST_TIMEOUT_LIMIT} seconds")
            request_status_code = 408
        except httpx.ConnectError:
            print(
                "Connection Error. It might be that the maximum number of retries with the URL has "
                "been exceeded"
            )
            request_status_code = 429

        if request_status_code != 200:
            print("The request failed, trying again...")
            number_tries += 1
            time.sleep(5)
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
            "You should specify all the options to run the script. For information on how to call the "
            "script, run `python -m client --help`"
        )
        sys.exit()

    base_url = f"http://localhost:{args.port}"
    print(
        f"Connecting to the mock server for the task {args.server_task} at {base_url}"
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
        print("GET request failed. Aborting script...")
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
        print(f">> Post number being processed: {current_response_number + 1}")
        asyncio.run(
            post_team_responses(
                base_url,
                args.server_task,
                args.team_token,
                args.team_runs,
                model_response,
            )
        )
        """
        results = asyncio.gather(
            *[
                post_response(
                    base_url, args.server_task, args.team_token, i, model_response
                )
                for i in range(args.team_runs)
            ]
        )
        print(f"results: {results}")
        """
        """
        for i in range(args.team_runs):
            _, status_code = asyncio.run(
                post_response(base_url, args.server_task, args.team_token, i, model_response)
            )

            if status_code != 200:
                print("POST request failed. Aborting script...")
                sys.exit()
        """

        last_json_response = new_json_response

        get_response, status_code = asyncio.run(
            get_writings(base_url, args.server_task, args.team_token)
        )

        if status_code != 200:
            print("GET request failed. Aborting script...")
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
        print(f"Reached the number of posts limit ({args.number_posts} posts)")

    print("End of script")