import matplotlib.pyplot as plt
import numpy as np
import scienceplots

plt.style.use('ieee')


def plot_figure(means, stdevs, n_steps, dt, tstep=100, fbmod=None, savefig=True):
    #Plot bladder volume and bladder pressure
    if fbmod is not None:
        fig1, ax1_1 = plt.subplots(figsize=(9,3))

        color = 'tab:red'
        ax1_1.set_xlabel('Time (t) [ms]')
        ax1_1.set_ylabel('Bladder Volume (V) [ml]', color=color)
        ax1_1.plot(fbmod.times, fbmod.b_vols, color=color,lw=0.5)
        ax1_1.tick_params(axis='y', labelcolor=color)

        ax2_1 = ax1_1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2_1.set_ylabel('Bladder Pressure (P) [cm H2O]', color=color)  # we already handled the x-label with ax1
        ax2_1.set_ylim(bottom=5,top=60)
        ax2_1.plot(fbmod.times, fbmod.b_pres, color=color,lw=0.5)
        ax2_1.tick_params(axis='y', labelcolor=color)

        fig1.tight_layout()  # otherwise the right y-label is slightly clipped

    # tstep (ms)
    tstop = (n_steps-1)*dt
    t = np.arange(0.0,tstop,tstep)
    ind = np.floor(t/dt).astype(np.int)

    fig2 = plt.figure(figsize=(9,3))
    plt.plot(t, means['Bladaff'][ind], color='k', label='Bladder Afferent',lw=0.5)
    plt.plot(t, means['PGN'][ind], color='g', label='PGN',lw=0.5)
    #plt.plot(t, means['PAGaff'][ind], color='r', marker='D', mfc='r', mec='r', label='PAG')
    # plt.plot(t, means['FB'][ind], color='k', marker='D', mfc='k', mec='k', label='FB')
    plt.plot(t, means['IMG'][ind], color='r', label='IMG',lw=0.5)
    plt.plot(t, means['IND'][ind], color='b', label='IND',lw=0.5)

    plt.xlabel('Time (t) [ms]')
    plt.ylabel('Neuron Firing Rate (FR) [Hz]')
    plt.legend()
    fig2.tight_layout()


    fig3 = plt.figure(figsize=(9,3))
    plt.plot(t, means['INmminus'][ind], color='r', label='INm-',lw=0.5)
    plt.plot(t, means['EUSaff'][ind], color='k', label='EUS Afferent',lw=0.5)
    # plt.plot(t, means['IND'][ind], color='r', marker='^', mfc='r', mec='r', label='IND')
    plt.plot(t, means['INmplus'][ind], color='g', label='INm+',lw=0.5)

    plt.xlabel('Time (t) [ms]')
    plt.ylabel('Neuron Firing Rate (FR) [Hz]')
    plt.legend()
    fig3.tight_layout()


    if savefig:
        if fbmod is not None:
            fig1.savefig('./graphs/Pressure_vol.png',transparent=True)
            fig1.savefig('./graphs/Pressure_vol.svg',dpi=500, transparent=True)
        fig2.savefig('./graphs/NFR_PGN.png',transparent=True)
        fig3.savefig('./graphs/NFR_INm.png',transparent=True)
        fig2.savefig('./graphs/NFR_PGN.svg',dpi=500,transparent=True)
        fig3.savefig('./graphs/NFR_INm.svg',dpi=500,transparent=True)


    plt.show()

def plotting_calculator(spike_trains, n_steps, dt, window, index, num, pop):
    # window (ms)
    ind = index[pop]
    n = num[pop]
    fr_conv = np.zeros((n,n_steps))
    
    def moving_avg(x):
        window_size = np.ceil(window/dt).astype(np.int)
        x_cum = np.insert(np.cumsum(x),0,np.zeros(window_size))
        y = (x_cum[window_size:]-x_cum[:-window_size])/(window_size*dt/1000)
        return y

    for gid in range(ind,ind+n):
        spikes = np.zeros(n_steps)
        spiketimes = spike_trains.get_times(gid)
        if len(spiketimes) > 0:
            spikes[(spiketimes/dt).astype(np.int)] = 1
        fr_conv[gid-ind] = moving_avg(spikes)

    means = np.mean(fr_conv,axis=0)
    stdevs = np.std(fr_conv,axis=0)
    
    return means, stdevs