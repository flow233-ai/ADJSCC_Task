# %% 瑞利衰落因子生成
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

x = np.array([0,5,10,15,20])
y = np.array([22.64,26.45,28.25,31.01,31.26])
y2 = np.array([0.87,2.68,3.05,21.63,25.64])

plt.plot(x,y,'v--')
plt.plot(x,y2,'o--')
plt.title('AWGN Channel(R=1/6)')
plt.xlabel("SNR(dB)")
plt.ylabel("PSNR(dB)")
plt.ylim([0.00,35.00])
plt.legend(['ADJSCC','JPEG'])
plt.grid(alpha=0.4)
plt.show()

