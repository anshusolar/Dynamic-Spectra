#!/usr/bin/env python
# coding: utf-8

# In[11]:


from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time, TimeDelta
from matplotlib.ticker import FixedLocator, FuncFormatter
from matplotlib.ticker import LogLocator, ScalarFormatter, NullFormatter
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter, MinuteLocator


# In[3]:


ls


# In[4]:


fits_file = 'int_orf20241110_120000_0.1.fts'


# In[5]:


try:
    with fits.open(fits_file) as hdul:
        # Frequencies from hdul[1]
        freq_hdu = hdul[1].data
        freqs = np.concatenate([
            freq_hdu['FREQ_B1'][0],  # 431
            freq_hdu['FREQ_B2'][0],  # 215
            freq_hdu['FREQ_B3'][0],  # 164
            freq_hdu['FREQ_B4'][0],  # 86
            freq_hdu['FREQ_B5'][0]   # 102
        ])  # Total: 998 frequencies
        print(f"Combined frequencies: {len(freqs)} channels, {freqs[0]:.1f}–{freqs[-1]:.1f} MHz")
        print("First 5 frequencies (MHz):", freqs[:5])
        print("Last 5 frequencies (MHz):", freqs[-5:])

        # Spectra from hdul[2]
        spec_hdu = hdul[2].data
        # Stokes I data
        data_i = np.hstack([
            spec_hdu['STOKESI_B1'],  # 72000 × 431
            spec_hdu['STOKESI_B2'],  # 72000 × 215
            spec_hdu['STOKESI_B3'],  # 72000 × 164
            spec_hdu['STOKESI_B4'],  # 72000 × 86
            spec_hdu['STOKESI_B5']   # 72000 × 102
        ])  # Shape: (72000, 998)
        print(f"Stokes I data shape: {data_i.shape}")

        # Stokes V data
        data_v = np.hstack([
            spec_hdu['STOKESV_B1'],  # 72000 × 431
            spec_hdu['STOKESV_B2'],  # 72000 × 215
            spec_hdu['STOKESV_B3'],  # 72000 × 164
            spec_hdu['STOKESV_B4'],  # 72000 × 86
            spec_hdu['STOKESV_B5']   # 72000 × 102
        ])  # Shape: (72000, 998)
        print(f"Stokes V data shape: {data_v.shape}")

        # Fallback time axis (in seconds)
        start_time = Time('2024-11-10T12:00:00', format='isot', scale='utc')
        time_deltas = TimeDelta(np.arange(data_i.shape[0]) * 0.1, format='sec')
        times = start_time + time_deltas
        print(f"Time range: {times[0].iso} to {times[-1].iso}")
        print("First 5 times:", [t.iso for t in times[:5]])
        print("Last 5 times:", [t.iso for t in times[-5:]])

        # Sort frequencies
        freq_indices = np.argsort(freqs)
        freqs = freqs[freq_indices]
        data_i_normalized = data_i - np.mean(data_i[:100, :], axis=0)
        data_i_normalized = data_i_normalized[:, freq_indices]
        data_v_normalized = data_v - np.mean(data_v[:100, :], axis=0)
        data_v_normalized = data_v_normalized[:, freq_indices]

        # Create frequency edges for pcolormesh (998 frequencies -> 999 edges)
        freq_edges = np.linspace(freqs[0], freqs[-1], len(freqs) + 1)
        time_edges = np.array([t.plot_date for t in times])

        # Define custom y-axis ticks
        major_ticks = [150, 200, 300, 400, 500, 600, 800, 1000]  # Clean frequency values
        minor_ticks = np.concatenate([
            np.arange(150, 200, 10),  # 150, 160, 170, ...
            np.arange(200, 300, 20),  # 200, 220, 240, ...
            np.arange(300, 400, 25),  # 300, 325, 350, ...
            np.arange(400, 600, 50),  # 400, 450, 500, ...
            np.arange(600, 1000, 100), # 600, 700, 800, 900
            np.array([1000])  # Ensure 1000 is included
        ])

                # Plot Stokes I
        plt.figure(figsize=(10, 6))
        data_i_normalized = data_i - np.mean(data_i[:100, :], axis=0)
        data_i_normalized = data_i_normalized[:, freq_indices]
        extent = [times[0].plot_date, times[-1].plot_date, freqs[0], freqs[-1]]
        plt.imshow(data_i_normalized.T, origin='lower', aspect='auto', extent=extent,
                   cmap='viridis', vmin=np.percentile(data_i_normalized, 5),
                   vmax=np.percentile(data_i_normalized, 95))
        plt.colorbar(label='Intensity (SFU/pixel)')
        plt.xlabel('Time (UTC)')
        plt.ylabel('Frequency (MHz)')
        plt.title('ORFEES Dynamic Spectrum (Stokes I, All Bands, 2025-01-31)')
        
        # Format time axis
        ax = plt.gca()
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
        ax.xaxis.set_major_locator(MinuteLocator(interval=15))
        ax.xaxis.set_minor_locator(MinuteLocator(interval=5))
        
        # Set log y-axis with good labels
        ax.set_yscale('log')
        ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0, 2.0, 5.0], numticks=15))
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
        
        plt.grid(True, which='both', alpha=0.3)
        plt.gcf().autofmt_xdate()
        plt.tight_layout()
        plt.show()
        
        # Plot Stokes V
        plt.figure(figsize=(10, 6))
        data_v_normalized = data_v - np.mean(data_v[:100, :], axis=0)
        data_v_normalized = data_v_normalized[:, freq_indices]
        plt.imshow(data_v_normalized.T, origin='lower', aspect='auto', extent=extent,
                   cmap='viridis', vmin=np.percentile(data_v_normalized, 5),
                   vmax=np.percentile(data_v_normalized, 95))
        plt.colorbar(label='Intensity (SFU/pixel)')
        plt.xlabel('Time (UTC)')
        plt.ylabel('Frequency (MHz)')
        plt.title('ORFEES Dynamic Spectrum (Stokes V, All Bands, 2025-01-31)')
        
        # Format time axis again
        ax = plt.gca()
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
        ax.xaxis.set_major_locator(MinuteLocator(interval=15))
        ax.xaxis.set_minor_locator(MinuteLocator(interval=5))
        
        # Log scale + readable ticks
        ax.set_yscale('log')
        ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0, 2.0, 5.0], numticks=15))
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
        
        plt.grid(True, which='both', alpha=0.3)
        plt.gcf().autofmt_xdate()
        plt.tight_layout()
        plt.show()

except Exception as e:
    print(f"Error processing FITS file: {str(e)}")


# In[6]:


def preprocess_log_median(data, baseline_len=100):
    """
    Convert data to log10 scale and subtract median over initial baseline_len time steps.
    """
    # Clip to avoid log(0)
    data_clipped = np.clip(data, 1e-3, None)  # Adjust floor if needed
    log_data = np.log10(data_clipped)
    median_baseline = np.median(log_data[:baseline_len, :], axis=0)
    normalized_data = log_data - median_baseline
    return normalized_data

def plot_dynamic_spectrum(times, freqs, data, title, cmap='viridis',
                          vlabel='Intensity (log, relative)', vmin=None, vmax=None):
    """
    Plot a dynamic spectrum (time vs. log-frequency) with optional vmin/vmax scaling.
    """
    extent = [times[0].plot_date, times[-1].plot_date, freqs[0], freqs[-1]]

    if vmin is None:
        vmin = np.percentile(data, 5)
    if vmax is None:
        vmax = np.percentile(data, 95)

    plt.figure(figsize=(10, 6))
    plt.imshow(data.T, origin='lower', aspect='auto', extent=extent,
               cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(label=vlabel)
    plt.xlabel('Time (UTC)')
    plt.ylabel('Frequency (MHz)')
    plt.title(title)

    ax = plt.gca()
    ax.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
    ax.xaxis.set_major_locator(MinuteLocator(interval=15))
    ax.xaxis.set_minor_locator(MinuteLocator(interval=5))
    
    ax.set_yscale('log')
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0, 2.0, 5.0], numticks=15))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())

    plt.grid(True, which='both', alpha=0.3)
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.show()


# In[7]:


try:
    with fits.open(fits_file) as hdul:
        # Load frequency data
        freq_hdu = hdul[1].data
        freqs = np.concatenate([
            freq_hdu['FREQ_B1'][0],
            freq_hdu['FREQ_B2'][0],
            freq_hdu['FREQ_B3'][0],
            freq_hdu['FREQ_B4'][0],
            freq_hdu['FREQ_B5'][0]
        ])
        print(f"Combined frequencies: {len(freqs)} channels")

        # Load spectral data
        spec_hdu = hdul[2].data
        data_i = np.hstack([
            spec_hdu['STOKESI_B1'],
            spec_hdu['STOKESI_B2'],
            spec_hdu['STOKESI_B3'],
            spec_hdu['STOKESI_B4'],
            spec_hdu['STOKESI_B5']
        ])
        data_v = np.hstack([
            spec_hdu['STOKESV_B1'],
            spec_hdu['STOKESV_B2'],
            spec_hdu['STOKESV_B3'],
            spec_hdu['STOKESV_B4'],
            spec_hdu['STOKESV_B5']
        ])
        print(f"Stokes I shape: {data_i.shape}, Stokes V shape: {data_v.shape}")

        # Time axis
        start_time = Time('2024-11-10T12:00:00', format='isot', scale='utc')
        time_deltas = TimeDelta(np.arange(data_i.shape[0]) * 0.1, format='sec')
        times = start_time + time_deltas

        # Sort frequencies
        freq_indices = np.argsort(freqs)
        freqs_sorted = freqs[freq_indices]

        # Process data
        data_i_processed = preprocess_log_median(data_i)
        data_v_processed = preprocess_log_median(np.abs(data_v)) * np.sign(data_v)

        # Apply frequency sorting
        data_i_processed = data_i_processed[:, freq_indices]
        data_v_processed = data_v_processed[:, freq_indices]

        # Plot
        plot_dynamic_spectrum(times, freqs_sorted, data_i_processed,
                              title='ORFEES Dynamic Spectrum (Stokes I, 2025-01-31)',
                              cmap='viridis', vlabel='Relative Log Intensity (I)')
        plot_dynamic_spectrum(times, freqs_sorted, data_v_processed,
                              title='ORFEES Dynamic Spectrum (Stokes V, 2025-01-31)',
                              cmap='RdBu', vlabel='Relative Log Intensity (V)')

except Exception as e:
    print(f"Error processing FITS file: {e}")


# In[12]:


# Define desired range
t_start = Time('2024-11-10T12:02:00', format='isot')
t_end = Time('2024-11-10T12:15:00', format='isot')
f_min = 150  # MHz
f_max = 700  # MHz

# Apply masks
time_mask = (times >= t_start) & (times <= t_end)
freq_mask = (freqs_sorted >= f_min) & (freqs_sorted <= f_max)

# Subset
times_zoom = times[time_mask]
freqs_zoom = freqs_sorted[freq_mask]
data_i_zoom = data_i_processed[time_mask][:, freq_mask]
data_v_zoom = data_v_processed[time_mask][:, freq_mask]

plot_dynamic_spectrum(times_zoom, freqs_zoom, data_i_zoom, title='Zoomed Stokes I', vmin=0, vmax=10)
plot_dynamic_spectrum(times_zoom, freqs_zoom, data_v_zoom, title='Zoomed Stokes V', cmap='RdBu', vmin=-0.25, vmax=0.05)


# In[ ]:




