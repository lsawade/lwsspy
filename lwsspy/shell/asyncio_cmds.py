from typing import Iterable, Any
import asyncio


async def asyncio_commands(cmd_list: list, cwdlist: Iterable[Any] = None):

    # Get directories
    if isinstance(cwdlist, list) is False:
        cwdlist = list(len(cmd_list) * [None])

    # Initialize list of process
    processes = []

    # define processes
    for _cmd, _cwd in zip(cmd_list, cwdlist):
        process = await asyncio.create_subprocess_shell(_cmd, cwd=_cwd)
        processes.append(process)

    # Run two asyncio processes at the same time with asyncio
    await asyncio.gather(*(_proc.communicate() for _proc in processes))
