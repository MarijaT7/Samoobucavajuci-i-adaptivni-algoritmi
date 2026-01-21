from time import sleep
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import random

#nasumicna pocetna politika pi0
#ili eps greedy ili samo greedy
def eps_greedy(q, eps: float):
    if random.random() > eps:
        return np.argmax(q)
    else:
        return random.randint(0,3)

def greedy(q):
    return np.argmax(q)


env = gym.make('FrozenLake-v1',
               desc=None,
               map_name="8x8",
               is_slippery=True,
               reward_schedule=(1, 0, 0),
               success_rate=1.5/2.0,
               render_mode='ansi')

#krecemo od jedan tj totalno random
EPSILON = 1
GAMMA = 0.9
EPISODES = 30000
LEARING_RATE = 0.01
eps_decay =  0.0001

#q tabela nam ima 64 stanja jer ima u 8x8 mapi imamo 64 polja, svako stanje je za svako polje
#pocinjemo iz stanja 0, u svakom stanju imamo 4 akcije koje mozemo preduzeti {0, 1, 2, 3}
#ne secam se koji broj oznacava koju akciju
q_table = np.zeros((64, 4))
rewards = []
returns_sums = np.zeros((64, 4))
returns_count = np.zeros((64, 4))
returns = []
eps = []
#ovo p ovde je verovatnoca kada budem ukljucio slippery
for i in range(EPISODES):
    eps.append(i)
    state, p = env.reset()
    #da li je agent upao u rupu
    term = False
    #da li je agent iskoristio 200(za 4x4 grid je 100) akcija
    trunc = False
    #ovo je samp jedna epizoda
    what_he_saw = []
    reward = 0
    while(not term and not trunc):
        action = eps_greedy(q_table[state, :], EPSILON)

        #print(env.step(action))
        new_state,reward,term,trunc,p = env.step(action)
        #print(new_state, reward, term, trunc, p)
        what_he_saw.append((state, action, reward))
        state = new_state
    
    #nagrada nakon svake epizode
    rewards.append(reward)

    g = 0
    gain = 0
    #azuriranje q tabele
    for i in range(len(what_he_saw)):
        s, a, r = what_he_saw[len(what_he_saw) - 1 - i]
        g = r + GAMMA * g
        gain += r
        #returns_sums[s, a] += g
        #returns_count[s, a] += 1

        #naivni
        #q_table[s, a] = returns_sums[s, a] / returns_count[s, a]

        #inkrementalni
        #q_table[s, a] = (1 - LEARNING_RATE) * q_table[s, a] + LEARNING_RATE * (returns_sums[s, a] / returns_count[s, a])
        #q_table[s, a] += 1/returns_count[s, a]*(g - q_table[s, a])
        q_table[s, a] = (1 - LEARING_RATE) * q_table[s, a] + LEARING_RATE * g

    #dobit nakon svake epizode
    returns.append(gain)
    EPSILON -= eps_decay if EPSILON > 0.1 else 0

#test
state, p = env.reset()
print(env.render())
term = False
trunc = False
while(not term and not trunc):
    action = greedy(q_table[state, :])

    #print(env.step(action))
    new_state,reward,term,trunc,p = env.step(action)
    #print(new_state, reward, term, trunc, p)
    state = new_state
    print(env.render())
    sleep(1)

sum_rewards = np.zeros(EPISODES)
for t in range(EPISODES):
    sum_rewards[t] = np.sum(rewards[max(0, t-100):(t+1)])
plt.plot(sum_rewards)
plt.show()
print(q_table)
