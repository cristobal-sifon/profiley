from matplotlib import pyplot as plt, ticker
import numpy as np

import profiley
from profiley.nfw import NFW

try:
    from plottery.plotutils import update_rcParams

    update_rcParams()
except ImportError:
    pass

print(profiley.__version__)

nfw = NFW(1e14, 5, 0.1, overdensity=200, background="c")
print(nfw)
r200 = nfw.rdelta(200, "c")
print(f"r200c = {r200} Mpc")

r = np.linspace(0, 5 * r200, 100)

fig, axes = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
ax = axes[0]
ax.plot(r / r200, nfw.potential(r))
ax.set(ylabel="$\\phi(r)$ (Mpc$^2$ s$^{-2}$)")
ax = axes[1]
ax.plot(r / r200, nfw.vescape(r))
ax.set(ylabel="$v_\\mathrm{esc}$ (km s$^{-1}$)")
ax.yaxis.set_major_locator(ticker.MultipleLocator(500))
for ax in axes:
    ax.set(xlabel="$r/r_\\mathrm{200c}$")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
plt.savefig("escape_velocity.png")
