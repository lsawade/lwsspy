import os
import asyncio
from typing import List, Optional


async def sync_data(
        data_database: str, new_database: str,
        eventlist: Optional[List[str]] = None):

    # If no list is provided, the entire database will be synchronized
    if eventlist is None:
        eventlist = os.listdir(data_database)
        # Initialize list of process
    processes = []

    # Rsync command to synchronize the databases in terms of events
    rsyncstr = 'rsync -av --include="*/" ' \
        '--include="*.mseed" ' \
        '--include="*.xml" ' \
        '--exclude="*"'

    # define processes
    for event in eventlist:

        # Full RSYNC command
        command = f"{rsyncstr} {data_database}/{event}/ {new_database}/{event}"

        # Create task for asyncio
        process = await asyncio.create_subprocess_shell(command)
        processes.append(process)

    # Run two asyncio processes at the same time with asyncio
    await asyncio.gather(*(_proc.communicate() for _proc in processes))


def bin():

    import argparse

    # Get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        dest='data_database', type=str, required=True,
        help='Database that contains the downloaded data.')
    parser.add_argument(
        dest='new_database', help='Database for inversion',
        required=True, type=str)
    parser.add_argument(
        '-e', '--event-list', dest='event_list', nargs='+',
        help='List of events to sync', default=None,
        required=False)
    args = parser.parse_args()

    asyncio.run(sync_data(args.data_database,
                          args.new_database, args.event_list))
