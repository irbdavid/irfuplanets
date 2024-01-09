# irfuplanets

[![codecov](https://codecov.io/gh/irbdavid/irfuplanets/branch/main/graph/badge.svg?token=irfuplanets_token_here)](https://codecov.io/gh/irbdavid/irfuplanets)
[![CI](https://github.com/irbdavid/irfuplanets/actions/workflows/main.yml/badge.svg)](https://github.com/irbdavid/irfuplanets/actions/workflows/main.yml)

Some half-finished MEX and MAVEN routines, and associated gubbins

## Install

For now, only directly from github:

```bash
pip install git+https://github.com/irbdavid/irfuplanets.git

```

## First Run

At first run, to create directories, sync spice kernels etc.:

```py
import irfuplanets

```

This will create a file `irfuplanets.cfg` in your home directory that will store some important paths.
Default settings should be adjusted if you want data to get pushed elsewhere.

Then,

```py
irfuplanets.first_run()

```
will create those directories if needed and sync some spice data for MEX and MAVEN.


## Usage

```py
    import matplotlib.pyplot as plt

    import irfuplanets.maven.lpw as lpw
    import irfuplanets.plot
    import irfuplanets.time

    start = irfuplanets.time.spiceet("2015-001T00:00")
    finish = start + 3600.*5
    data = lpw.lpw_l2_load(kind="lpnt", start=start, finish=finish)


    plt.plot(data['time'], data['ne'])
    irfuplanets.time.setup_time_axis(calendar=True)

    orbit = irfuplanets.maven.orbit(start)
    print(orbit.number)

```

See also various notebooks in the [docs/examples](examples) folder.

## Development

<!-- Read the [CONTRIBUTING.md](CONTRIBUTING.md) file. -->

Run the test suite using pytest if you make changes.

```bash
pytest .
```