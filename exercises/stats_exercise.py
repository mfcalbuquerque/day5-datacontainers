import numpy as np
from scipy import stats as sts
import matplotlib.pyplot as plt

#a
print("\na)")
fig,(ax1,ax2,ax3) = plt.subplots(1,3, figsize=(10,6))

mu = 0.5
mean, ver, skew, kurt = sts.poisson.stats(mu, moments='mvsk')
print(mean,ver,skew,kurt)

#PMF
x = np.arange(sts.poisson.ppf(0.01, mu), sts.poisson.ppf(0.99, mu))
ax1.plot(x, sts.poisson.pmf(x,mu), 'bo', label="Poisson PMF")
ax1.vlines(x,0, sts.poisson.pmf(x,mu), colors='k', lw=5, alpha=0.5)
ax1.legend(frameon=False)
ax1.set_title("PMF")

#CDF
prob = sts.poisson.cdf(x,mu)

ax2.plot(x, prob, 'bo', label="Poisson CDF")
ax2.vlines(x, 0, prob, colors='k', lw=5, alpha=0.5)
ax2.legend(frameon=False)
ax2.set_title("CDF")

#Random realizations
r = sts.poisson.rvs(mu, size=1000)

ax3.hist(r, label="random realizations")
ax3.legend(frameon=False)
ax3.set_title("Random realizations of Poisson distribution")

plt.tight_layout()
fig.savefig("poisson.png")




#b
print("\nb)")
