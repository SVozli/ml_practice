import numpy as np

start = 6
end  = 20
feature = np.arange(start,end +1)
print(feature)

label = 3*feature +4
noise = np.random.random(np.size(label))*4 -2
print(noise)
label = label + noise
print(label)