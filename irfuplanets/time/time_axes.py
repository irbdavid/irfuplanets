"""Helper classes and functions for making time plots with matplotlib,
using the spice ephemeris time as a base."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import Formatter, Locator
from matplotlib.transforms import nonsingular

from irfuplanets.time.time import CelsiusTime, spiceet, utcstr

__author__ = "David Andrews"
__copyright__ = "Copyright 2023, David Andrews"
__license__ = "MIT"
__email__ = "david.andrews@irfu.se"

MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


def setup_time_axis(
    ax=None,
    xaxis=True,
    verbose=False,
    calendar=True,
    locator_kwargs=None,
    formatter_kwargs=None,
    **kwargs,
):
    """Function that applies SPICEET based axis locator and labeller to the
    current plot axis (default xaxis)

    Parameters
    ----------
    ax : mpl Axes, optional
        default to `plt.gca()`
    xaxis : bool, optional
        xaxis if True, otherwise yaxis, by default True
    verbose : bool, optional
        Shouty or not, by default False
    calendar : bool, optional
        Use Calendar dates, by default True
        If False, will format using DOY.
    locator_kwargs : dict, optional
        passed to locator, by default None
    formatter_kwargs : dict, optional
        passed to formatter, by default None
    **kwargs: passed to locator & formatter (duplicated)

    """

    if locator_kwargs is None:
        locator_kwargs = {}

    if formatter_kwargs is None:
        formatter_kwargs = {}

    if calendar:
        locator_kwargs["calendar"] = True
        # formatter_kwargs['calendar'] = True

    for k in kwargs:
        formatter_kwargs[k] = kwargs[k]
        locator_kwargs[k] = kwargs[k]

    if ax is None:
        ax = plt.gca()
    if xaxis:
        a = ax.xaxis
    else:
        a = ax.yaxis

    loc = SpiceetLocator(**locator_kwargs)
    form = SpiceetFormatter(locator=loc, **formatter_kwargs)
    a.set_major_locator(loc)
    a.set_major_formatter(form)
    if verbose:
        print(a, ax, loc)


class SpiceetLocator(Locator):
    """Locator class that determines sensible time tick locations using
    SPICE."""

    def __init__(
        self,
        nmax=7,
        nmin=3,
        prune=None,
        verbose=False,
        calendar=False,
        spacing=None,
        minor=False,
    ):
        """Create a SpiceetLocator instance.

        Args:
            nmax, nmin (int): max and minimum number of ticks.  If no
                satisfying 'nice' tick intervals are found that conform to this
                range, the range is doubled.  If this doesn't work then a) you
                are probably beyond the use-case of this class, and b) Fixed
                tick spacing is returned.
            prune (str): 'lower'/'upper'/'both' to remove relevant ticks.
            verbose (bool): print some info
            calendar (bool): Use numbered months where appropriate (ISOC),
                instead of DOY (ISOD).
            spacing (str or tupple): Force the use of a specific interval,
                irrespective of the axis limits.  If string, intpreted as one
                of the pre-determined 'allowed intervals', e.g. '10_days'.
                Otherwise, :spacing: should be a tuple of the same format used
                for 'allowed_intervals',
                  ('name including base', spacing_seconds, multiples_of_base),
                e.g. ('3days', 86400.*3., 3).
            minor (bool): Set options for minor tick locating
                    (redefines nmin and nmax)
        """
        self._nmax = nmax
        self._nmin = nmin
        self._prune = prune
        self._spacing = spacing
        self.calendar = calendar

        self._minor = False
        if minor:
            self._minor = True
            self._nmin = self._nmax
            self._nmax = self._nmin**2

        self.verbose = verbose

        self.name = ""
        self.sep = None
        self.multiple = None

        # name, spacing_seconds, integer
        self.allowed_intervals = [
            ("1000_years", 1000 * 365.25 * 86400.0, 1000),
            ("100_years", 100 * 365.25 * 86400.0, 100),
            ("50_years", 50 * 365.25 * 86400.0, 50),
            ("20_years", 20 * 365.25 * 86400.0, 20),
            ("10_years", 10 * 365.25 * 86400.0, 10),
            ("5_years", 5 * 365.25 * 86400.0, 5),
            ("2_years", 2 * 365.25 * 86400.0, 2),
            ("1_year", 365.25 * 86400.0, 1),
            (
                "6_months",
                6 * 28.0 * 86400.0,
                6,
            ),
            ("3_months", 3 * 28.0 * 86400.0, 3),
            ("2_months", 2 * 28.0 * 86400.0, 2),
            ("1_month", 1 * 28.0 * 86400.0, 1),
            ("10_days", 10.0 * 86400.0, 10),
            ("5_days", 5.0 * 86400.0, 5),
            ("2_days", 2.0 * 86400.0, 2),
            ("1_day", 86400.0, 1),
            ("6_hours", 6 * 3600.0, 6),
            ("4_hours", 4 * 3600.0, 4),
            ("2_hours", 2 * 3600.0, 2),
            ("1_hour", 1 * 3600.0, 1),
            ("30_minutes", 30 * 60.0, 30),
            ("20_minutes", 20 * 60.0, 20),
            ("10_minutes", 10.0 * 60.0, 10),
            ("5_minutes", 5 * 60.0, 5),
            ("2_minutes", 2 * 60.0, 2),
            ("1_minute", 60.0, 1),
            ("30_seconds", 30.0, 30),
            ("20_seconds", 20.0, 20),
            ("10_seconds", 10.0, 10),
            ("5_seconds", 5.0, 5),
            ("1_second", 1.0, 1),
            ("0.5_second", 0.5, 0.5),
            ("0.1_second", 0.1, 0.1),
            ("0.05_second", 0.05, 0.05),
            ("0.01_second", 0.01, 0.01),
        ]

        # remove 'month'-based intervals for non-calendar use
        if not self.calendar:
            self.allowed_intervals = [
                p for p in self.allowed_intervals if "month" not in p[0]
            ]
            # self.allowed_intervals.insert(0, ('20_days', 86400.*20., 20))
            self.allowed_intervals.insert(0, ("50_days", 86400.0 * 50.0, 50))
            self.allowed_intervals.insert(
                0, ("100_days", 86400.0 * 100.0, 100)
            )

    def bin_boundaries(self, start, finish):
        if finish < start:
            raise ValueError("Finish before start, doofus")

        duration = finish - start

        if self._spacing is None:
            best = None
            second_best = None
            for p in self.allowed_intervals:
                n = int(duration / p[1]) + 1
                if (n > self._nmin) and (n < self._nmax):
                    best = p
                    break
                if (n > self._nmin) and (n < (2 * self._nmax)):
                    second_best = p

            if best is None:
                if second_best is None:
                    print("Tick Locator had no sensible result")
                    best = "", None, None
                else:
                    best = second_best

        else:
            if isinstance(self._spacing, str):
                for a in self.allowed_intervals:
                    if self._spacing == a[0]:
                        best = a
                        break
                else:
                    raise ValueError(
                        "Spacing name '%s' not recognized" % self._spacing
                    )
            else:
                best = self._spacing

        self.name = best[0]
        self.sep = best[1]
        self.multiple = best[2]

        if self.verbose:
            print(self)
            print("\tDuration ", duration)
            print("\tStart ", utcstr(start))
            print("\tFinish ", utcstr(finish))
            print("\tBest: ", self.name)

        first_tick = start  # + sep/2.
        first = CelsiusTime(first_tick)

        ticks = []
        if "year" in self.name:
            year = self.multiple * int(first.year / self.multiple)
            t = spiceet("%04d-001T00:00:00" % year)
            while t <= finish:
                ticks.append(t)
                year += self.multiple
                t = spiceet("%04d-001T00:00:00" % year)

        elif "month" in self.name:
            month = 1
            year = first.year
            t = spiceet("%04d-%02d-01T00:00:00" % (year, month))
            while t <= finish:
                ticks.append(t)
                month += self.multiple
                if month > 11:
                    month -= 12
                    year += 1
                t = spiceet("%04d-%02d-01T00:00:00" % (year, month))

        elif "day" in self.name:
            day = 1
            year = first.year
            month = first.month
            year = first.year
            if self.calendar:
                t = spiceet("%04d-%02d-%02dT00:00:00" % (year, month, day))
            else:
                t = spiceet("%04d-001T00:00:00" % year)
            day = 0
            while t <= finish:
                ticks.append(t)
                day += self.multiple
                if self.calendar:
                    test = CelsiusTime(
                        "%04d-%02d-%02dT00:00:00" % (year, month, day)
                    )
                    if test.month != month:
                        month = test.month
                        day = 1
                        t = spiceet(
                            "%04d-%02d-%02dT00:00:00"
                            % (test.year, test.month, day)
                        )
                        day = 0  # to get back in even sync
                    else:
                        t = test.spiceet
                else:
                    test = CelsiusTime("%04d-%03dT00:00:00" % (year, day))
                    if test.year != year:
                        year += 1
                        day = 1
                        t = spiceet("%04d-001T00:00:00" % (test.year))
                        day = 0
                    else:
                        t = test.spiceet

        elif "hour" in self.name:
            hour = 0
            day = first.day
            month = first.month
            year = first.year
            t = spiceet("%04d-%02d-%02dT%02d:00:00" % (year, month, day, hour))
            while t <= finish:
                ticks.append(t)
                hour += self.multiple
                if hour > 24:
                    hour -= 24
                    day += 1
                test = CelsiusTime(
                    "%04d-%02d-%02dT%02d:00:00" % (year, month, day, hour)
                )
                if self.calendar and (test.month != month):
                    month = test.month
                    day = 1
                    hour = 0
                    t = spiceet(
                        "%04d-%02d-%02dT%02d:00:00" % (year, month, day, hour)
                    )
                else:
                    t = test.spiceet

        elif "minute" in self.name:
            minute = 0
            hour = 0
            day = first.day
            month = first.month
            year = first.year
            t = spiceet(
                "%04d-%02d-%02dT%02d:%02d:00"
                % (year, month, day, hour, minute)
            )

            while t <= finish:
                ticks.append(t)
                minute += self.multiple
                if minute > 60:
                    minute = 0
                    hour += 1
                if hour > 24:
                    hour -= 24
                    day += 1
                test = CelsiusTime(
                    "%04d-%02d-%02dT%02d:%02d:00"
                    % (year, month, day, hour, minute)
                )
                if test.month != month:
                    month = test.month
                    day = 1
                    hour = 0
                    t = spiceet(
                        "%04d-%02d-%02dT%02d:%02d:00"
                        % (year, month, day, hour, minute)
                    )
                else:
                    t = test.spiceet

        elif "second" in self.name:
            # Finally, in seconds, we don't need to care about
            # leap seconds messing up our nice ordering!

            # Lets remove the '.81608'
            t = spiceet(
                "%04d-%02d-%02dT%02d:%02d:%02d"
                % (
                    first.year,
                    first.month,
                    first.day,
                    first.hour,
                    first.minute,
                    first.second,
                )
            )
            ticks = np.arange(t, finish + self.multiple, self.multiple)

        else:
            # Make the best of a bad situation
            ticks = np.linspace(start, finish, (self._nmax + self._nmin) / 2)

        ticks = [t for t in ticks if (t > start) and (t < finish)]
        # if self.verbose:
        #     print('-'*5)
        #     print(len(ticks))
        #     print(self.name, self.sep, self.multiple)
        #     for t in ticks:
        #         print(t, utcstr(t), 'ISOC')
        #     print('-'*5)

        self.locs = ticks

        return ticks

    def __call__(self):
        vmin, vmax = self.axis.get_view_interval()
        vmin, vmax = nonsingular(vmin, vmax, expander=1.0)
        locs = self.bin_boundaries(vmin, vmax)

        prune = self._prune
        if prune == "lower":
            locs = locs[1:]
        elif prune == "upper":
            locs = locs[:-1]
        elif prune == "both":
            locs = locs[1:-1]

        if self.verbose:
            print(
                self,
                "__call__: ",
                utcstr(vmin),
                utcstr(vmax),
                ": ",
                self.name,
                len(locs),
            )

        return locs  # , self.raise_if_exceeds(locs)


class SpiceetFormatter(Formatter):
    """Subclass of mpl.ticker.Formatter that deals nicely with
    Spice ET values"""

    def __init__(self, locator=None, full=False, verbose=False, label=True):
        """Create an instance of SpiceetFormatter.

        Args:
            locator (SpiceetLocator): Corresponding instance required.
            full (bool): Always print the full time, not just at the start of
                each interval.
            verbose (bool): Print some extra info when called.
            label (bool): Set the axis label as well.

        Returns:
            instance.
        """

        if not isinstance(locator, SpiceetLocator):
            raise ValueError(
                "This will not work without knowledge of the accomapanying "
                "SpiceetLocator() instance"
            )

        self.locator = locator
        self.full = full
        self.verbose = verbose
        self.label = label

        # Will be used to decide e.g. whether the month has changed between
        # adjacent day ticks:
        self.last_large = None

    def __call__(self, x, pos=None):
        if pos == 0:
            self.last_large = None

        # do_label = False
        # if self.label & (pos == 0):
        #     do_label = True

        if self.full:
            self.last_large = None

        t = CelsiusTime(x)

        if "year" in self.locator.name:
            val = "%04d" % t.year
            lbl = "YYYY"

        elif "month" in self.locator.name:
            large = "%04d" % t.year
            if large != self.last_large:
                self.last_large = large
                val = "%04d/%02d" % (t.year, t.month)
            else:
                val = "%02d" % t.month
            lbl = "YYYY/MM"

        elif "day" in self.locator.name:
            if self.locator.calendar:
                large = "%04d-%02d" % (t.year, t.month)
                if large != self.last_large:
                    self.last_large = large
                    val = "%02d\n%s" % (t.day, large)
                else:
                    val = "%02d" % t.day
                lbl = "DD\nYYYY-MM"
            else:
                large = "%04d" % t.year
                if large != self.last_large:
                    self.last_large = large
                    val = "%03d\n%s" % (t.doy, large)
                else:
                    val = "%03d" % t.doy
                lbl = "DOY\nYYYY"

        elif "hour" in self.locator.name:
            if self.locator.calendar:
                large = "%04d-%02d-%02d" % (t.year, t.month, t.day)
                lbl = "YYYY-MM-DD"
            else:
                large = "%04d-%03d" % (t.year, t.doy)
                lbl = "YYYY-DOY"
            if large != self.last_large:
                self.last_large = large
                val = "%02d:%02d\n%s" % (t.hour, t.minute, large)
            else:
                val = "%02d:%02d" % (t.hour, t.minute)
            lbl = "HH:MM\n" + lbl

        elif "minute" in self.locator.name:
            if self.locator.calendar:
                large = "%04d-%02d-%02d" % (t.year, t.month, t.day)
                lbl = "YYYY-MM-DD"
            else:
                large = "%04d-%03d" % (t.year, t.doy)
                lbl = "YYYY-DOY"
            if large != self.last_large:
                self.last_large = large
                val = "%02d:%02d\n%s" % (t.hour, t.minute, large)
            else:
                val = "%02d:%02d" % (t.hour, t.minute)
            lbl = "HH:MM\n" + lbl

        elif ("second" in self.locator.name) or not (self.locator.name):
            if self.locator.calendar:
                large = "%04d-%02d-%02dT%02d:%02d" % (
                    t.year,
                    t.month,
                    t.day,
                    t.hour,
                    t.minute,
                )
            else:
                large = "%04d-%03dT%02d:%02d" % (
                    t.year,
                    t.doy,
                    t.hour,
                    t.minute,
                )
            if large != self.last_large:
                self.last_large = large
                val = "%g\n%s" % (t.second, large)
            else:
                val = "%g" % (t.second)
            lbl = "SS"

        if pos == 0:
            if self.label:
                self.axis.set_label_text("UTC " + lbl)

        return val

    def format_data_short(self, x):
        if self.locator.calendar:
            return utcstr(x, "ISOC")
        return utcstr(x, "ISOD")


if __name__ == "__main__":
    plt.close("all")

    day = 86400.0
    extents = (
        60.0,
        600,
        3600.0,
        day / 2.0,
        day,
        2 * day,
        10.0 * day,
        20.0 * day,
        100.0 * day,
        365 * day,
        365 * 5 * day,
        365 * day * 20.0,
    )

    fig, axs = plt.subplots(len(extents), 1, figsize=(8, 13))

    # print(spiceet_to_strdict(0.0))

    for i, e in enumerate(extents):
        print("---------")
        print(i, e)
        ax = axs[i]
        plt.sca(ax)
        plt.plot((0.0, e), (0.0, 1.0), "k-")
        plt.plot((0.0, e), (1.0, 0.0), "k-")
        plt.xlim(0.0, e)

        loc = SpiceetLocator(verbose=True)
        form = SpiceetFormatter()
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(form)

        #
        # ax2 = plt.twiny()
        # plt.xlim(0., e)
        #
        # l2 = mangled_mpl_dates.AutoDateLocator()
        # f2 = mangled_mpl_dates.AutoDateFormatter(l)
        # ax2.xaxis.set_major_locator(l2)
        # ax2.xaxis.set_major_formatter(f2)

        # l = SpiceetCalendarLocator()
        #
        # if e > 86400. * 100.:
        #     l = SpiceetYearLocator()
        # f = SpiceetFormatter(calendar=True)
        # ax2.xaxis.set_major_locator(l)
        # ax2.xaxis.set_major_formatter(f)

    plt.subplots_adjust(hspace=5.0)

    plt.show()
