import matplotlib.pylab as plt
import numpy as np

import irfuplanets.mex as mex
from irfuplanets.data import deg_unwrap
from irfuplanets.planets.mars import constants
from irfuplanets.planets.mars.field_models import plot_lat_lon_field
from irfuplanets.plot import CircularLocator, make_colorbar_cax, map_along_line

__author__ = "David Andrews"
__copyright__ = "Copyright 2023, David Andrews"
__license__ = "MIT"
__email__ = "david.andrews@irfu.se"


def plot_planet(
    radius=1.0,
    orientation="dawn",
    ax=None,
    origin=(0.0, 0.0),
    scale=0.96,
    edgecolor="black",
    facecolor="white",
    resolution=256,
    zorder=None,
    **kwargs,
):
    """`orientation' specifies view direction, one of 'noon', 'midnight',
    anything else being used as the terminator.  `scale' is used to adjust
    the 'white' polygon size in relation to the radius for aesthetics"""
    if zorder is None:
        zorder = -9999

    if ax is None:
        ax = plt.gca()

    o = orientation.lower()

    if o == "noon":
        ax.add_patch(
            plt.Circle(
                origin,
                radius,
                fill=True,
                zorder=zorder,
                facecolor=facecolor,
                edgecolor=edgecolor,
                **kwargs,
            )
        )
        return

    elif o == "midnight":
        ax.add_patch(
            plt.Circle(
                origin,
                radius,
                fill=True,
                color=edgecolor,
                zorder=zorder,
                **kwargs,
            )
        )
        return

    theta = np.linspace(0.0, np.pi, resolution) - np.pi / 2.0
    xy = np.empty((resolution, 2))
    xy[:, 0] = scale * radius * np.cos(theta)
    xy[:, 1] = scale * radius * np.sin(theta)
    if o == "dusk":
        xy[:, 0] = -1.0 * xy[:, 0]

    semi = plt.Polygon(xy, closed=True, color=facecolor, fill=True)
    ax.add_patch(
        plt.Circle(
            origin,
            radius,
            fill=True,
            color=edgecolor,
            zorder=zorder + 1,
            **kwargs,
        )
    )
    ax.add_patch(semi)


def plot_mpb(resolution=256, model="vignes00", ax=None, zorder=None, **kwargs):
    if zorder is None:
        zorder = -9999

    if ax is None:
        ax = plt.gca()

    m = model.lower()
    if m == "vignes00":
        phi = np.linspace(-np.pi, np.pi, resolution)
        x = 0.78 + 0.96 * np.cos(phi) / (1 + 0.9 * np.cos(phi))
        y = 0.96 * np.sin(phi) / (1 + 0.9 * np.cos(phi))
    else:
        raise ValueError("Model %s not recognized" % model)

    plt.plot(x, y, zorder=zorder, **kwargs)


def plot_bs(resolution=256, model="vignes00", ax=None, zorder=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    if zorder is None:
        zorder = -9999

    m = model.lower()
    if m == "vignes00":
        phi = np.linspace(-np.pi, np.pi, resolution)
        x = 0.64 + 2.04 * np.cos(phi) / (1 + 1.03 * np.cos(phi))
        y = 2.04 * np.sin(phi) / (1 + 1.03 * np.cos(phi))
    else:
        raise ValueError("Model %s not recognized" % model)

    plt.plot(x, y, zorder=zorder, **kwargs)


def orbit_plots(
    orbit_list=[8020, 8021, 8022, 8023, 8024, 8025], resolution=30.0, ax=None
):
    if ax is None:
        ax = plt.gca()
    plt.sca(ax)

    orbits = {}
    props = dict(marker="None", mec=None, linestyle="-", markeredgewidth=0.0)
    # props = dict(marker='o', hold=True,mec='None',
    #       line style='None',markeredgewidth=0.0)

    for orbit in orbit_list:
        orbit_t = mex.orbits[orbit]
        print(orbit, orbit_t.start, orbit_t.finish)
        t = np.arange(orbit_t.start, orbit_t.finish, resolution)
        pos = mex.iau_pgr_alt_lat_lon_position(t)
        # orbits[orbit] = dict(lat = np.rad2deg(pos[2]),
        #           lon = np.rad2deg(np.unwrap(pos[1])),
        #           alt=pos[0] - constants.mars_mean_radius_km, t=t)
        orbits[orbit] = dict(
            lat=pos[1],
            lon=np.rad2deg(np.unwrap(np.deg2rad(pos[2]))),
            alt=pos[0],
            t=t,
        )

        ll = plt.plot(
            orbits[orbit]["lon"],
            orbits[orbit]["lat"],
            label=str(orbit),
            **props,
        )

        plt.plot(
            orbits[orbit]["lon"] + 360.0,
            orbits[orbit]["lat"],
            label="_nolegend_",
            color=ll[0].get_color(),
            **props,
        )
        plt.plot(
            orbits[orbit]["lon"] - 360.0,
            orbits[orbit]["lat"],
            label="_nolegend_",
            color=ll[0].get_color(),
            **props,
        )

    plt.xlim(-180, 180)
    plt.ylim(-90, 90)
    plt.xlabel("Longitude / deg")
    plt.ylabel("Latitude / deg")
    plt.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        numpoints=1,
        title="Orbit",
    )
    ax.xaxis.set_major_locator(CircularLocator())
    ax.yaxis.set_major_locator(CircularLocator())


def mso_orbit(orbit_number, res=60):
    o = mex.orbits[orbit_number]
    start = o.start
    finish = o.finish
    et = np.arange(start, finish, res)

    i = np.sum(et < o.periapsis)
    inbound = np.arange(0, i, 60)
    outbound = np.arange(i + 60, et.shape[0], 60)

    pos = mex.mso_position(et) / constants.mars_mean_radius_km
    lims = (-3, 3)
    plt.rcParams["font.size"] = 8

    phi = np.linspace(-np.pi, np.pi, 100)
    x = 0.78 + 0.96 * np.cos(phi) / (1 + 0.9 * np.cos(phi))
    y = 0.96 * np.sin(phi) / (1 + 0.9 * np.cos(phi))

    plt.figure(figsize=(8, 4))

    def panel(no, xn, yn, bs=True):
        plt.subplot(no, aspect="equal")
        if bs:
            plt.plot(x, y, "k--")
        plt.xlabel(xn + r"$ / R_M$")
        plt.ylabel(yn + r"$ / R_M$")
        plt.xlim(*lims)
        plt.ylim(*lims)
        plt.gca().add_patch(plt.Circle((0.0, 0.0), 1.0, fill=False))

    panel(141, r"$X$", r"$\rho$")
    plt.title("MEX ORBIT %d" % o.number)
    rho = np.sqrt(pos[1] ** 2.0 + pos[2] ** 2.0)
    plt.plot(pos[0], rho, "k-")
    plt.gca().add_patch(
        plt.Circle(
            (0.0, 0.0),
            1.0 + 1200.0 / constants.mars_mean_radius_km,
            fill=False,
            linestyle="dotted",
        )
    )
    plt.gca().add_patch(
        plt.Circle(
            (0.0, 0.0),
            1.0 + 1500.0 / constants.mars_mean_radius_km,
            fill=False,
            linestyle="dotted",
        )
    )
    plt.gca().add_patch(
        plt.Circle(
            (0.0, 0.0),
            1.0 + 2000.0 / constants.mars_mean_radius_km,
            fill=False,
            linestyle="dotted",
        )
    )

    plt.plot(pos[0, inbound], rho[inbound], "ko")
    plt.plot(pos[0, outbound], rho[outbound], "ko", mfc="white")

    panel(142, r"$X$", r"$Z$")
    plt.plot(pos[0], pos[2], "k-")
    plt.plot(pos[0, inbound], pos[2, inbound], "ko")
    plt.plot(pos[0, outbound], pos[2, outbound], "ko", mfc="white")

    panel(143, r"$X$", r"$Y$")
    plt.plot(pos[0], pos[1], "k-")
    plt.plot(pos[0, inbound], pos[1, inbound], "ko")
    plt.plot(pos[0, outbound], pos[1, outbound], "ko", mfc="white")

    panel(144, r"$Y$", r"$Z$", bs=False)
    plt.plot(pos[1], pos[2], "k-")
    plt.plot(pos[1, inbound], pos[2, inbound], "ko")
    plt.plot(pos[1, outbound], pos[2, outbound], "ko", mfc="white")

    plt.subplots_adjust(wspace=0.3, right=0.96, top=0.94, bottom=0.07)

    # return dict(pos=pos, rho=rho, t)


def setup_lat_lon_ax(ax=None, label=True, tickspacing=30.0):
    if ax is None:
        ax = plt.gca()
    else:
        plt.sca(ax)

    plt.xlim(-180, 180)
    plt.xticks(np.arange(-360.0 - 180, 360.0 + 180 + 1.0, tickspacing))
    plt.ylim(-90, 90)
    plt.yticks(np.arange(-90, 90 + 1.0, tickspacing))

    if label:
        plt.xlabel("Lon. / deg")
        plt.ylabel("Lat. / deg")

    ax.set_aspect("equal")


def plot_surface_map(
    orbits, ax=None, param="time", cmap=None, norm=None, **kwargs
):
    summer = plt.colormaps["summer"]

    from irfuplanets.mex.ais import get_ais_index

    if ax is None:
        ax = plt.gca()
    else:
        plt.sca(ax)

    if cmap is None:
        cmap = summer

    setup_lat_lon_ax(ax=ax)

    try:
        plot_lat_lon_field(
            value="|B|", cax=make_colorbar_cax(half=True, upper=True)
        )
    except NotImplementedError:
        pass

    if "ais_index" not in mex.datastore:
        mex.datastore["ais_index"] = get_ais_index()
    g = mex.datastore["ais_index"]
    g = [v for k, v in g.items() if k in orbits]

    if not g:
        raise ValueError("No data?")

    for gg in g:
        gg["iau_pos"][2, :] = deg_unwrap(gg["iau_pos"][2, :])

    if param is None:
        for gg in g:
            for i in (-1, 0, 1):
                plt.plot(
                    gg["iau_pos"][2, :] + i * 360.0,
                    gg["iau_pos"][1, :],
                    **kwargs,
                )

    if param == "sza":
        if norm is None:
            norm = plt.Normalize(vmin=0.0, vmax=180.0)
        title = "SZA / deg"

    elif param == "time":
        if norm is None:
            norm = plt.Normalize(
                vmin=mex.orbits[min(orbits)].start,
                vmax=mex.orbits[max(orbits)].finish,
            )
        title = "Time / ET"

    elif param == "alt":
        if norm is None:
            norm = plt.Normalize(vmin=0.0, vmax=constants.mars_mean_radius_km)
        title = "Alt. / km"

    else:
        raise ValueError("Unrecognised parameter % s" % param)

    cmap = summer
    lc = None

    lc_kw = dict(cmap=cmap, norm=norm, linewidths=4, capstyle="round")

    for gg in g:
        for i in (-1.0, 0.0, 1.0):
            if param == "sza":
                lc = map_along_line(
                    gg["iau_pos"][2, :] + i * 360.0,
                    gg["iau_pos"][1, :],
                    gg["sza"],
                    # time=gg["time"],
                    **lc_kw,
                )
            if param == "time":
                lc = map_along_line(
                    gg["iau_pos"][2, :] + i * 360.0,
                    gg["iau_pos"][1, :],
                    gg["time"],
                    # time=gg["time"],
                    **lc_kw,
                )
            if param == "alt":
                lc = map_along_line(
                    gg["iau_pos"][2, :] + i * 360.0,
                    gg["iau_pos"][1, :],
                    gg["iau_pos"][0, :],
                    # time=gg["time"],
                    **lc_kw,
                )

            # plt.scatter(gg['iau_pos[2,:], gg['iau_pos[1,:], c=gg['sza,
            #             s=10., vmin=30, vmax=150, edgecolor='none')
    if lc:
        c = plt.colorbar(lc, cax=make_colorbar_cax(half=True, upper=False))
        c.set_label(title)
    else:
        raise ValueError("Nothing mapped :(")

    plt.sca(ax)
    return lc


if __name__ == "__main__":
    plt.close("all")
    plt.figure()
    lc = plot_surface_map([2489, 10267], param="alt")
    plt.show()
    # max_orbit = max(mex.orbits.keys())
    # for o in range(1850, max_orbit, 50):
    #     print o
    #     plt.close('all')
    #     mso_orbit(o)
    #     plt.savefig(mex.data_directory() + "orbit_plots/%d.png" % o)
    #     plt.close('all')
