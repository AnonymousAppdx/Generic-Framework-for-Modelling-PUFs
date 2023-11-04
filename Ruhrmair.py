### This code is from https://pypuf.readthedocs.io/en/latest/

import pypuf.simulation, pypuf.io
# puf = pypuf.simulation.FeedForwardArbiterPUF(n=64, ff=[(32,50)],seed=1)
puf = pypuf.simulation.XORArbiterPUF(n=64,k=4,seed=3)
crps = pypuf.io.ChallengeResponseSet.from_simulation(puf, N=10000, seed=2)
import pypuf.attack
attack = pypuf.attack.LRAttack2021(crps, seed=3, k=4, bs=500, lr=.001, epochs=200,stop_validation_accuracy=.99)
attack.fit()
