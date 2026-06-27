import numpy as	np
from  math import pi, cos, sin

def RaisedCosinePulse(t, Freq, Amplitude):
	"""
	Raised-Cosine Pulse
	
	""" 
	N  = np.size(t,0)
	P  = np.zeros((N,),dtype=np.float32)
	for m in range(0,N):
		if t[m] <= 2.0/Freq:
			P[m] = Amplitude *(1-cos(pi*Freq*t[m]))*cos(2*pi*Freq*t[m])

	return P


def PulseUTsin(t,fc,NT, Amp):
	"""
	Gaussian Pulse

	Code inspired by
	see http://www.k-wave.org/
	"""
	def gaussian_func(x,mean,variance):
		return np.exp(-((x - mean)**2)/(2*variance))


	Ts			 = t[1]-t[0]
	Fs			 = 1.0/Ts
	tone_length	 = ( NT/(fc*1.0) )
	tone_t		 = np.arange(0,tone_length,Ts)
	tone_burst_t = np.sin(2*pi*fc*tone_t)

	x_lim = 3
	window_x = np.arange(-x_lim,x_lim, 2.0*x_lim/(np.size(tone_burst_t)-1) )
	window	 = gaussian_func(window_x,0, 1)

	ind = np.min([ np.size(window_x), np.size(tone_burst_t)])
	tone_burst=np.zeros((ind,))
	tone_burst[0:ind] = window[0:ind]*tone_burst_t[0:ind] 

	y	= np.zeros((np.size(t),))


	try:
		y[0:np.size(tone_burst)] = Amp*tone_burst[:]
		return y
	except:
		raise Exception("Signal does not fit in the Configured Time, Increase the Simulation Time to solve this issue")



def _cosine_taper(n, frac=0.05):
    """
    Symmetric cosine taper (Tukey-like) applied only near the ends.
    frac=0.05 means 5% of samples at each end are tapered.
    """
    n = int(n)
    if n <= 1:
        return np.ones((n,), dtype=np.float64)

    frac = float(frac)
    frac = max(0.0, min(frac, 0.5))

    w = np.ones(n, dtype=np.float64)
    m = int(np.floor(frac * n))
    if m <= 0:
        return w

    # rising edge
    t = np.linspace(0.0, 1.0, m, endpoint=False)
    w[:m] = 0.5 * (1.0 - np.cos(np.pi * t))
    # falling edge
    t = np.linspace(1.0, 0.0, m, endpoint=False)
    w[-m:] = 0.5 * (1.0 - np.cos(np.pi * t))
    return w


def _zero_mean(x):
    x = np.asarray(x, dtype=np.float64)
    return x - np.mean(x)


# ---------------------------
# Morlet helpers (NEW)
# ---------------------------
def _tukey_window(n, alpha=0.25):
    """
    Tukey window with guaranteed zeros at both ends for alpha>0.
    alpha in [0,1]; alpha=1 -> Hann, alpha=0 -> rectangular.
    """
    n = int(n)
    if n <= 1:
        return np.ones(n, dtype=np.float64)

    alpha = float(alpha)
    alpha = max(0.0, min(alpha, 1.0))

    if alpha == 0.0:
        return np.ones(n, dtype=np.float64)
    if alpha >= 1.0:
        w = np.hanning(n).astype(np.float64)
        w[0] = 0.0
        w[-1] = 0.0
        return w

    w = np.ones(n, dtype=np.float64)
    edge = int(np.floor(alpha * (n - 1) / 2.0))

    if edge < 1:
        w[0] = 0.0
        w[-1] = 0.0
        return w

    # Rising edge: i = 0..edge
    i = np.arange(0, edge + 1)
    w[i] = 0.5 * (1.0 + np.cos(np.pi * (2 * i / (alpha * (n - 1)) - 1)))

    # Falling edge: i = n-edge-1..n-1
    i = np.arange(n - edge - 1, n)
    w[i] = 0.5 * (1.0 + np.cos(np.pi * (2 * i / (alpha * (n - 1)) - 2 / alpha + 1)))

    # Force exact zeros
    w[0] = 0.0
    w[-1] = 0.0
    return w


def MorletPulse(
    t,
    fc,
    n_cycles=5,
    Amp=1.0,
    omega0=6.0,          # kept for compatibility
    t0=None,             # kept for compatibility (not used)
    sigma=None,          # kept for compatibility (not used directly)
    force_zero_mean=True,
    tukey_alpha=0.25,
    burst_factor=2.0,
    start_at_sigma=3.0
):
    """
    Morlet (Option B): build a finite, tapered burst and embed it into the full record.
    - Exact zeros before/after the burst (no baseline offset tail)
    - Smooth start/end (no sharp edges)
    """

    t = np.asarray(t, dtype=np.float64)
    Ts = t[1] - t[0]
    N = len(t)

    # keep your sigma rule (derived from cycles)
    sigma_eff = (n_cycles / float(fc)) / 6.0

    # burst length long enough that Gaussian ~0 at edges
    burst_T = (n_cycles / float(fc)) * float(burst_factor)
    nb = int(np.ceil(burst_T / Ts))
    nb = max(nb, 16)
    if nb % 2 == 1:
        nb += 1

    # local time centered at 0
    tl = (np.arange(nb) - nb / 2.0) * Ts

    env = np.exp(-0.5 * (tl / sigma_eff) ** 2)
    carrier = np.cos(2.0 * np.pi * fc * tl)
    yb = Amp * env * carrier

    # taper to zero at ends (guaranteed)
    yb *= _tukey_window(nb, alpha=tukey_alpha)

    # enforce zero-mean on the burst (prevents DC injection)
    if force_zero_mean:
        yb -= np.mean(yb)

    # force exact zeros (numerical safety)
    yb[0] = 0.0
    yb[-1] = 0.0

    # embed without cropping (avoid cutting nonzero tail)
    i0 = int(np.round((float(start_at_sigma) * sigma_eff) / Ts))
    i0 = max(0, min(i0, N - nb))
    i1 = i0 + nb

    y = np.zeros(N, dtype=np.float64)
    y[i0:i1] = yb

    return y.astype(np.float32)



class Signals():
	
	def __init__(self, Name="RaisedCosine", Amplitud=1, Frequency=1, N_Cycles=1):
		
		self.Name      = Name
		self.Amplitude = Amplitud
		self.Frequency = Frequency
		self.N_Cycles  = N_Cycles
	
	
	def __str__(self):
		return "Signal: "  + str(self.Name)

	def __repr__(self):
		return "Signal: " + str(self.Name)
		

	def generate(self, t):
		if self.Name == "RaisedCosine":
			return RaisedCosinePulse(t, self.Frequency, self.Amplitude)
		elif self.Name == "GaussianSine":
			return PulseUTsin(t,self.Frequency,self.N_Cycles,self.Amplitude)

		elif self.Name == "Morlet":
			return MorletPulse(
				t,
				self.Frequency,
				n_cycles=self.N_Cycles,
				Amp=self.Amplitude,
				force_zero_mean=True
			)

		
	def save(self,t):
		self.time_signal  = self.generate(t)
