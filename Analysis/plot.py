import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors

chirp = [-3,-2.5,-2,-1.5,-1,0,1,1.5,2,2.5,3]

ion = [74.17687570282736, 78.1569962897474, 82.48265147085077, 86.97996552962626, 91.19421837812764, 92.32065911737577, 23.2229852220734, 23.748530230945654, 36.5076740210728, 58.18340652161582, 69.65210517763482]


plt.plot(chirp, ion)
plt.savefig("ion.png")