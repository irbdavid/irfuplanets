=============================================================
Python library for accessing and plotting MAVEN Level 2 data.
=============================================================

Introduction
------------
This module provides an interface to the MAVEN Science Data Center (SDC), hosted at LASP.
    `https://lasp.colorado.edu/maven/sdc/public`
Routines are provided that can query the SDC for the latest MAVEN data, download to a local mirror, delete obsolete data, and read and plot said data using matplotlib.  MAVEN science team members with access to the private area of the SDC can also use these routines by supplying their credentials in a shell variable.

The set of instruments and data products supported for reading and plotting is evolving, and contributions to this are very welcome.

Requirements
------------

1. SpiceyPy library required for NAIF spice interface `https://github.com/AndrewAnnex/SpiceyPy`

Installation
------------

1. Install `irfuplanets`
3. If you have team-level SDC access, a shell variable needs to be set
containing your username and password:
    `export MAVENPFP_USER_PASS=username:password`
4. At a minimum, set the local directory that will be used for data storage:
    `export SC_DATA_DIR="~/data"`
4. To provide comapatability with a parallel use of the Berkeley IDL code base, `data_directory` and `kernel_directory` can be used to specify the locations of existing local data, using the `~/irfuplanets.cfg` file.


Examples of use
---------------
```python
import irfuplanets.maven.lpw
import matplotlib.pyplot as plt
import numpy as np

start = spiceet("2015-05-01")
finish = start + 86400. -1. # just to avoid loading two files instead of one.

lpw_data = irfuplanets.maven.lpw.lpw_l2_load(kind='lpnt', start=start, finish=finish)

# Plot some LPW data
fig, axs = plt.subplots(3,1,sharex=True)

plt.sca(axs[0])
plt.plot(lpw_data['time'], lpw_data['ne'], 'k.')

plt.sca(axs[1])
swea_data =  irfuplanets.maven.swea.load_swea_l2_summary(start, finish)
irfuplanets.maven.swea.plot_swea_l2_summary(swea_data)

time = np.linspace(start, finish, 128)
pos_mso =  irfuplanets.maven.mso_position(time)

plt.sca(axs[2])
plt.plot(time, pos_mso[0], 'r-')
plt.plot(time, pos_mso[1], 'g-')
plt.plot(time, pos_mso[2], 'b-')

setup_time_axis()

plt.show()

# Plot some KP densities:
data = irfuplanets.maven.kp.load_kp_data(start, finish)

fig, axs = plt.subplots(3,1,sharex=True)

plt.sca(axs[0])
plt.plot(data.time, data.swia.hplus_density, 'k-')
plt.plot(data.time, data.static.oplus_density, 'b-')
plt.plot(data.time, data.static.o2plus_density, 'b--')
plt.plot(data.time, data.ngims.ion_density_amu_16plus, 'm-')
plt.plot(data.time, data.ngims.ion_density_amu_32plus, 'm--')
plt.plot(data.time, data.lpw.electron_density, 'g-')

plt.yscale('log')
plt.ylabel('Density / cc')

plt.sca(axs[1])
plt.plot(data.time,data.spacecraft.sc_alt_w_r_t_aeroid)

plt.sca(axs[2])
plt.plot(data.time,data.mag.mag_field_mso_x, 'r')
plt.plot(data.time,data.mag.mag_field_mso_y, 'g')
plt.plot(data.time,data.mag.mag_field_mso_z, 'b')

setup_time_axis()
plt.show()




```
