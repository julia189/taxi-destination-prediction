import time

import pandas as pd

def athena_query(client, query_string, database_name, output_path, max_execution_sec=30) -> pd.DataFrame:
    """Run a query against an Athena database and return the results.

    Start the (asynchronous) execution of the provided query and check every second if it succeeded.
    If so, load the results into a data frame and return it, otherwise return an empty data frame.
    Try for a maximum of max_execution_sec seconds.

    Args:
        client: boto3 athena client
        query_string: SQL query to run
        database_name: database to query
        output_path: s3 path to bucket into which results are written
        max_execution_sec: maximum query execution time in seconds

    Returns:
        pandas DataFrame containing the query results
    """
    response = client.start_query_execution(QueryString=query_string,
                                            QueryExecutionContext={'Database': database_name},
                                            ResultConfiguration={'OutputLocation': output_path})
    execution_id = response['QueryExecutionId']

    for _ in range(max_execution_sec):
        response = client.get_query_execution(QueryExecutionId=execution_id)
        state = response.get('QueryExecution', {}).get('Status', {}).get('State')
        if state == 'SUCCEEDED':
            return pd.read_csv(response['QueryExecution']['ResultConfiguration']['OutputLocation'])
        elif state == 'FAILED':
            print(f'Query execution failed. Athena Response:\n{response}')
            break
        time.sleep(1)
    client.stop_query_execution(QueryExecutionId=execution_id)
    print(f'''Query execution stopped after {max_execution_sec} sec. Either increase max_execution_sec or run
              a faster query. Tip: add a WHERE statement on a partitioned column (column names starting with p_)''')
    return pd.DataFrame()