from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#Initial data
f = pd.read_csv('us_oil.csv')
print(f)

plt.plot(f['volume'])
plt.show()

#Accumulation function
F = []
F.append(f['volume'][0])
for i in range(len(f) - 1):
	F.append(F[i] + f['volume'][i + 1])

plt.plot(range(len(F)), F, linewidth = 1)
plt.show()

#Linear regression
x = np.array(F[-90:]).reshape(-1, 1)
y = f['volume'][-90:] / F[-90:]
model = LinearRegression().fit(x, y)

plt.plot(x, model.predict(x))
plt.plot(x, y, 'rx')
plt.show()

#Inflection point
x_inf = -model.intercept_ / model.coef_[0] / 2
print(x_inf)
