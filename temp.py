import matplotlib.pyplot as plt
import numpy as np

logsamples = np.logspace(-2,4,100,base=2)
linsamples = np.linspace(1/4,16,100)

hm12 = np.sum(logsamples[logsamples > 1] <2)
hm816 = np.sum(logsamples[logsamples > 8] <16)

lm12 = np.sum(linsamples[linsamples > 1] <2)
lm816 = np.sum(linsamples[linsamples > 8] <16)


print("Amount of log samples 1-2 : ", hm12)
print("Amount of log samples 8-16 : ", hm816)

print("Amount of lin samples 1-2 : ", lm12)
print("Amount of lin samples 8-16 : ", lm816)

plt.scatter(linsamples,linsamples,label="LinSamples",s=1)
plt.scatter(linsamples,logsamples,label="LogSamples",s=1)
print(logsamples[-10:])
plt.legend()
plt.show()
