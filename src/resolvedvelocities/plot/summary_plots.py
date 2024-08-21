# summary_plots.py

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import datetime as dt
import os

def plot_components(utime, ybins, vector, covariance, titles=None, ylabel=None, yticklabels=None, clim=None, cmap=None, filename=None, scale_factors=None):

    # pad time gaps
    utime, [vector, covariance] = timegaps(utime, [vector, covariance])

    # get x-axis (time) tick locations and labels
    time, xticks, xlims = get_time_ticks(utime)

    xedge = np.append(utime[:,0],utime[-1,1])
    yedge = calculate_yedge(ybins)

    # loop over every day of experiment
    for xlim in xlims:

        fig = plt.figure(figsize=(15,10))
        gs = gridspec.GridSpec(2,3)
        gs.update(wspace=0.05,hspace=0.2,left=0.05,right=0.9,bottom=0.05,top=0.9)

        for i, A in enumerate([vector,np.sqrt(np.diagonal(covariance,axis1=-1,axis2=-2))]):
            for j in range(3):
                # for along B component, multiply by 10
                if scale_factors:
                    fac = scale_factors[j]
                else:
                    fac=1.

                ax = plt.subplot(gs[i,j])
                f = ax.pcolormesh(xedge, yedge, A[:,:,j].T*fac, vmin=clim[i][0], vmax=clim[i][1], cmap=plt.get_cmap(cmap[i]))
                ax.set_xticks(xticks)
                ax.set_xticklabels(time)
                ax.set_xlim(xlim)
                ax.set_xlabel('Universal Time')

                if j == 0:
                    ax.set_ylabel(ylabel)
                else:
                    ax.set_yticklabels([])

                title = titles[j]
                if fac != 1:
                    title = title + ' x {}'.format(fac)
                if i == 1:
                    title = title + ' Error'

                ax.set_title(title)
                ax.tick_params(labelsize=8)

            pos = ax.get_position()
            cax = fig.add_axes([0.91, pos.y0, 0.015, pos.y1-pos.y0])
            cbar = fig.colorbar(f, cax=cax)

        datestr = '{:%Y-%m-%d %H:%M:%S} - {:%Y-%m-%d %H:%M:%S}'.format(dt.datetime.utcfromtimestamp(xlim[0]),dt.datetime.utcfromtimestamp(xlim[1]))
        # fig.text(0.1, 0.95, datestr, fontsize=12)
        fig.suptitle(datestr, fontsize=12)

        fig.savefig(filename,bbox_inches='tight')
        plt.close(fig)


def plot_magnitude(utime, ybins, vmag, dvmag, vdir, dvdir, chi2, err_thres=None, mag_thres=None, titles=None, ylabel=None, yticklabels=None, clim=None, cmap=None, filename=None):

    # pad time gaps
    utime, [vmag, dvmag, vdir, dvdir, chi2] = timegaps(utime, [vmag, dvmag, vdir, dvdir, chi2])

    # get x-axis (time) tick locations and labels
    time, xticks, xlims = get_time_ticks(utime)

    xedge = np.append(utime[:,0],utime[-1,1])
    yedge = calculate_yedge(ybins)

    # loop over every day of experiment
    for xlim in xlims:

        fig = plt.figure(figsize=(10,10))
        gs = gridspec.GridSpec(5,1)
        gs.update(hspace=0.075,left=0.15,right=0.9,bottom=0.05,top=0.94)

        for i, A in enumerate([vmag, dvmag, vdir, dvdir]):
            ax = plt.subplot(gs[i])
            f = ax.pcolormesh(xedge, yedge, A.T, vmin=clim[i][0], vmax=clim[i][1], cmap=plt.get_cmap(cmap[i]))
            ax.set_xticks(xticks)
            ax.set_xticklabels([])
            ax.set_xlim(xlim)
            ax.set_ylabel(ylabel)
            pos = ax.get_position()
            cax = fig.add_axes([0.91, pos.y0, 0.015, pos.y1-pos.y0])
            cbar = fig.colorbar(f, cax=cax)
            cbar.set_label(titles[i])

        cond1 = (~np.isfinite(chi2)) | (chi2 > 10) | (chi2 < 0.1)  # chi-squared filter
        cond2 = ((dvmag > err_thres) & (vmag > mag_thres))  # absolute error bar filter
        cond3 = (dvmag / vmag > 1) # relative error bar filter
        inds = np.where(cond1 | cond2 | cond3)
        vmag[inds] = np.nan

        # plot quivers
        ax = plt.subplot(gs[4])
        f = ax.quiver(xedge[:-1], yedge[:-1], vmag.T*np.sin(vdir.T*np.pi/180.), vmag.T*np.cos(vdir.T*np.pi/180.), np.sin(vdir.T*np.pi/180.), cmap=plt.get_cmap('coolwarm'), norm=plt.Normalize(vmin=-1,vmax=1))
        ax.set_xticks(xticks)
        ax.set_xticklabels(time)
        ax.set_xlim(xlim)
        ax.set_xlabel('Universal Time')
        ax.set_ylabel(ylabel)
        pos = ax.get_position()
        cax = fig.add_axes([0.91, pos.y0, 0.015, pos.y1-pos.y0])
        cbar = fig.colorbar(f, cax=cax, ticks=[-0.9,0,0.9])
        cbar.ax.set_yticklabels(['West','','East'])

        # for the quiver key, let's make a key representative of the
        # vectors we just plotted. We'll round to the nearest power of
        # 10 but never plot a vector bigger than rounding to nearest 100
        # and never smaller than 10
        max_vmag_plotted = np.nanmax(vmag)
        if np.isnan(max_vmag_plotted):
            max_vmag_plotted = 10
        max_power = int(np.round(np.log10(max_vmag_plotted),decimals=0))
        if max_power > 2:
            max_power = 2
        max_vmag_plotted = np.round(max_vmag_plotted,decimals=-max_power)
        if max_vmag_plotted < 10:
            max_vmag_plotted = 10

        # cbar.ax.quiverkey(f, 1.075, 0.5, clim[0][1], str(clim[0][1]))
        cbar.ax.quiverkey(f, 1.075, 0.55, max_vmag_plotted, '%0.0f' % max_vmag_plotted)

        datestr = '{:%Y-%m-%d %H:%M:%S} - {:%Y-%m-%d %H:%M:%S}'.format(dt.datetime.utcfromtimestamp(xlim[0]),dt.datetime.utcfromtimestamp(xlim[1]))
        # fig.text(0.1, 0.95, datestr, fontsize=12)
        fig.suptitle(datestr, fontsize=12)

        fig.savefig(filename,bbox_inches='tight')
        plt.close(fig)


def get_time_ticks(utime):
    # get x-axis (time) tick locations and labels

    # calculate experiment length
    exp_start = dt.datetime.utcfromtimestamp(utime[0,0])
    exp_end = dt.datetime.utcfromtimestamp(utime[-1,1])
    exp_len = (exp_end-exp_start).total_seconds()
    # set spaceing between each time tick based on lenght of experiment
    if exp_len < 1.*60.*60.:
        dtick = 10.
    elif exp_len < 3.*60.*60.:
        dtick = 15.
    elif exp_len < 6.*60.*60.:
        dtick = 30.
    elif exp_len < 12.*60.*60.:
        dtick = 60.
    else:
        dtick = 3*60.

    # get first time tick
    h0 = exp_start.replace(minute=0, second=0)
    while h0<exp_start:
        h0 = h0+dt.timedelta(minutes=dtick)

    # get list of all time ticks
    time_ticks = []
    while h0 < exp_end:
        time_ticks.append(h0)
        h0 = h0+dt.timedelta(minutes=dtick)

    # calculate position of each time tick
    xticks = [(tt-dt.datetime.utcfromtimestamp(0)).total_seconds() for tt in time_ticks]
    # form list of strings for actual labels
    time = [t.strftime('%H:%M') for t in time_ticks]

    xlims = []
    if exp_len > 24.*60.*60:
    # if experiment is longer than 1 day, break up plot limits and create a series of plots
        st = exp_start
        et = exp_start.replace(hour=0, minute=0, second=0) + dt.timedelta(days=1)
        while et < exp_end:
            xlims.append([st,et])
            st = et
            et = et + dt.timedelta(days=1)
        xlims.append([st,exp_end])
    else:
    # if experiment shorter than one day, just create one plot
        xlims.append([exp_start, exp_end])

    # convert xlims to unix time
    xlims = [[(x[0]-dt.datetime.utcfromtimestamp(0)).total_seconds(), (x[1]-dt.datetime.utcfromtimestamp(0)).total_seconds()] for x in xlims]

    return time, xticks, xlims


def timegaps(time, data_arrays):
# pad time gaps with NaNs for better pcolormesh plotting

    # find the time between sucessive records and the cadance of the entire time series
    time_diff = np.diff(np.mean(time,axis=1))
    dt = np.median(time_diff)
    # find gaps in the time series
    gaps = np.argwhere(time_diff > 2*dt).flatten()+1

    # if no gaps, return original arrays
    if not gaps.size:
        return time, data_arrays

    # create array of times to fill each gap
    time_insert = np.array([time[g-1]+dt for g in gaps])
    # insert fill times into each gap
    time2 = np.insert(time,gaps,time_insert, axis=0)

    data_arrays2 = []
    for data in data_arrays:
        # create array of nans to fill each gap in data array
        data_insert = np.full(data[0].shape,np.nan)
        # insert fill nans into data array
        data2 = np.insert(data,gaps,data_insert, axis=0)
        data_arrays2.append(data2)

    return time2, data_arrays2

def calculate_yedge(ybins):
    yedge = (ybins[:-1]+ybins[1:])/2.
    yedge = np.insert(yedge, 0, ybins[0]-(yedge[0]-ybins[0]))
    yedge = np.append(yedge, ybins[-1]+(ybins[-1]-yedge[-1]))
    return yedge
