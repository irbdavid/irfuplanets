import logging
import logging.handlers
import multiprocessing as mp
import os
import sys
import time as py_time

import matplotlib
import numpy as np

import irfuplanets
import irfuplanets.mex.ais as ais
from irfuplanets.mex.ais.aisreview import main
from irfuplanets.time import now

matplotlib.use("Agg")
print("MPL Backend: ", matplotlib.get_backend())

np.seterr(all="ignore")
save_every = 1

queue = False


def determine_last_processed_orbit():
    max_orbit = -1
    p = irfuplanets.config["mex"]["data_directory"] + "marsis/ais/"
    for dname in os.listdir(p):
        if dname[:3] != "RDR":
            continue

        orbit = int(dname[3:-1]) * 10
        if orbit > max_orbit:
            max_orbit = orbit
            for fname in os.listdir(p + dname):
                orbit = int(fname.split("_")[-1][:-4])
                if orbit > max_orbit:
                    max_orbit = orbit

    return max_orbit


def determine_old_orbits(max_age=86400.0 * 7):
    orbits = []
    now = py_time.time()
    p = irfuplanets.config["mex"]["data_directory"] + "ais_plots/v0.9/"
    for root, dirs, files in os.walk(p):
        for f in files:
            s = os.stat(root + "/" + f)
            if ".pdf" in f:
                if (now - s.st_mtime) > max_age:
                    orbits.append(int(f.split(".")[0]))
                    print(orbits[-1])
    return sorted(orbits)


def worker_configurer(queue):
    h = logging.handlers.QueueHandler(queue)  # Just the one handler needed
    root = logging.getLogger()
    root.handlers = []
    root.addHandler(h)
    # send all messages, for demo; no other level or filter logic applied.
    root.setLevel(logging.DEBUG)


def junk(o, queue, configurer):
    configurer(queue)
    # logger.log(logging.DEBUG, str(o))
    logging.info(str(o))
    print(str(o))
    return str(o)


def async_worker_computer(o, queue, configurer, debug=False, verbose=False):
    configurer(queue)
    logging.info("Starting %d" % o)

    try:
        ais.compute_all_digitizations(o)
        logging.info("Completed %d" % o)
    except Exception as e:
        if debug:
            raise
        logging.info("Error %d: %s" % (o, str(e)))
        return False

    return True


def async_worker_review(o, queue, configurer, debug=True, verbose=True):
    configurer(queue)
    logging.info("Starting %d" % o)

    try:
        ais.compute_all_digitizations(o)
        d = ais.DigitizationDB(o)
        if len(d) == 0:
            ais.compute_all_digitizations(o)
        main(o, show=False, save=True)
        logging.info("Completed %d" % o)
    except Exception as e:
        if debug:
            raise
        logging.info("Error %d: %s" % (o, str(e)))
        return False

    return True


def listener_configurer():
    root = logging.getLogger()
    h = logging.FileHandler(
        irfuplanets.config["mex"]["data_directory"] + "ais_workflow_output.txt"
    )
    f = logging.Formatter(
        "%(asctime)s %(processName)-10s %(name)s %(levelname)-8s %(message)s"
    )
    h.setFormatter(f)
    root.addHandler(h)
    print("Configured")


# This is the listener process top-level loop: wait for logging events
# (LogRecords)on the queue and handle them, quit when you get a None for a
# LogRecord.
def listener_process(queue, configurer):
    configurer()
    while True:
        try:
            record = queue.get()
            if (
                record is None
            ):  # We send this as a sentinel to tell the listener to quit.
                break
            logger = logging.getLogger(record.name)
            logger.handle(
                record
            )  # No level or filter logic applied - just do it!
        except Exception:
            import sys
            import traceback

            print("Whoops! Problem:", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)


class DeltaTimer(object):
    """docstring for DeltaTimer"""

    def __init__(self):
        super(DeltaTimer, self).__init__()
        self._t = now()

    def __call__(self, s):
        n = now()
        print(s + ": %0.3fs" % (n - self._t))
        self._t = n


if __name__ == "__main__":
    from irfuplanets.mex import orbits
    from irfuplanets.mex.ais.ais_code import (
        _generate_ais_coverage,
        _generate_ais_index,
    )

    verbose = False
    save_every = 1
    np.seterr(all="ignore")
    t0 = now()
    exception_list = {}
    repeat = True

    start = determine_last_processed_orbit() - 50
    # start = 1840
    start = 17000
    finish = orbits[now()].number - 10

    if len(sys.argv) > 1:
        start = int(sys.argv[1])
    if len(sys.argv) > 2:
        finish = int(sys.argv[2])

    orbits = list(range(start, finish))

    orbits = determine_old_orbits(86400.0 * 5.0)

    # orbits = range(5)

    completed_orbits = []
    duration_list = []

    print("-- Starting writer")
    queue = mp.Manager().Queue(-1)
    writer = mp.Process(
        target=listener_process, args=(queue, listener_configurer)
    )
    writer.start()

    runner = async_worker_review
    # runner = async_worker_computer
    # runner = junk

    # print('--- test --- ')
    # print(runner(4264, debug=True, verbose=True))
    # print('--- /test --- ')

    processes = 1
    if "dja" in os.getenv("USER"):  # dunno why HOSTNAME doesn't work
        processes = 8

    print("-- Creating pool of %d processes" % processes)
    pool = mp.Pool(processes)

    print("-- Starting work...")
    print("-- orbits = ", orbits)
    r = [
        pool.apply_async(runner, args=(o, queue, worker_configurer))
        for o in orbits
    ]
    print("-- Jobs allocated")

    pool.close()
    pool.join()

    queue.put_nowait(None)
    # queue.close()
    # queue.join()
    # print('-- Queue closed, joining writer')
    writer.terminate()
    writer.join()
    # print r[0].get()`

    print("\n" * 5)
    print("Total Duration: %f h" % ((now() - t0) / 3600.0))
    # print "Exceptions encountered:"
    # for k, v in exception_list.iter_items():
    #     print '%s: %s' % (utcstr(k), str(v))

    if not runner == junk:
        _generate_ais_coverage()
        _generate_ais_index(recompute=False, update=True)
