import lecroyparser as lp
path = "/home/makmak/Projects/cv2/src/split-hopkinson_bar/Bill Krish SPH Data Share/BillTest/C1--Bars1_9-al_con-secontrial--00000.trc"
import numpy as np
import matplotlib.pyplot as plt

data = lp.ScopeData(path, parseAll = True)
i = 0
'''
for dat in data.y:
    plt.plot(data.x[::100], dat[::100], 'x', label = f"channel {i}")
    i += 1
'''
plt.plot(data.x, data.y[2], 'x')
#plt.legend()
plt.show()
