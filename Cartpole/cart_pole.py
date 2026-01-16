import numpy as np
import matplotlib.pyplot as plt
import random

class Enviroment():
    #m - masa stapa
    #M - masa kolica(u klasi cuvamo ukupnu masu kolica + stapa)
    #l - duzina stapa
    #TIME_STEP - za diskretizaciju vremena
    def __init__(self, m: float, M: float, l: float, TIME_STEP: float):
        self.m = m
        #ukupna masa
        self.M = M + m
        self.l = l
        #gravitaciona konstanta
        self.g = 9.81
        self.x = 0
        #diskretizacija vremena
        self.TIME_STEP = TIME_STEP
        #konstanta vezana za moment inercije stapa
        #zelela sam da ignorišem inerciju jer ona nije fokus
        self.k = 0

        #sipka se nalazi u ravnotezi na pocetku
        #ugao u radijanima
        self.theta = 0
        self.x = 0
        self.theta_th = np.deg2rad(15)
        self.x_th = 2.5
        #ugaona brzina(z4)
        self.theta_dot = 0
        #horizontalna brzina(z2)
        self.x_dot = 0

        #ako predjemo 500 koraka vraca se true za truncated (ogranicavamo duzinu epizode)
        self.steps = 0
        self.max_steps = 500
        self.trunc = False
        #da li je sipka pala
        self.term = False

        #sila koju mozemo primeniti na kolica
        #takodje skup akcija
        self.actions = [-10, 10]

    #F - sila koju primenjujemo
    def f_theta(self, x, x_dot, theta, theta_dot, F):
        return (self.M * self.g * np.sin(theta) - \
                np.cos(theta) * (F + self.m * self.l * theta_dot**2 * np.sin(theta))) \
                / ((1+self.k) * self.M * self.l  - self.m*self.l*np.cos(theta)**2)

    #F - sila koju primenjujemo
    def f_x(self, x, x_dot, theta, theta_dot, F):
        return (self.m * self.g * np.sin(theta) * np.cos(theta) - (1 + self.k) * (F + self.m * self.l * theta_dot**2 * np.sin(theta))) \
                / (self.m * np.cos(theta)**2 - (1 + self.k) * self.M)

    #funkcija nagrade
    def reward(self):
        reward = -abs(self.theta)
        #sipka pala
        if self.term:
            reward = -1000
        return reward
        
    #F - sila koju primenjujemo
    #action - akcija koju uzimamo
    #uzeo sam da je action_space={0, 1} gde 0 predstavlja silu od -10N a 1 silu od +10N
    def take_action(self, action: int):
        self.steps += 1
        if self.steps > self.max_steps:
            self.trunc = True
            self.term = False
            
            return [self.x, self.x_dot, self.theta, self.theta_dot], self.reward(), self.trunc, self.term

        if abs(self.theta) > self.theta_th or abs(self.x) > self.x_th:
            self.term = True

        if self.term:
            return [self.x, self.x_dot, self.theta, self.theta_dot], self.reward(), self.trunc, self.term
        F = self.actions[action]
        self.theta_dot += self.TIME_STEP * self.f_theta(self.x, self.x_dot, self.theta, self.theta_dot, F)
        self.x_dot += self.TIME_STEP * self.f_x(self.x, self.x_dot, self.theta, self.theta_dot, F)
        self.theta += self.TIME_STEP * self.theta_dot
        self.x += self.TIME_STEP * self.x_dot
        return [self.x, self.x_dot, self.theta, self.theta_dot], self.reward(), self.trunc, self.term

    def reset(self):
        self.x = 0
        self.x_dot = 0
        self.theta = 0
        self.theta_dot = 0
        self.steps = 0
        self.trunc = False
        self.term = False
        return [self.x, self.x_dot, self.theta, self.theta_dot], self.reward(), self.trunc, self.term

# Q-learning parametri

ALPHA = 0.1      # learning rate - koliko brzo agent uci iz novih iskustava
                 # ALPHA = 0 - agent u potpunosti ignorise novo iskustvo
                 # ALPHA = 1 - agent u potpunosti ignorise(zaboravlja) staro iskustvo, gleda samo novo
                 # ALPHA = 0.1 - agent polako inkorporira novo iskustvo u staro znanje

GAMMA = 0.99     # discount factor - govori koliko agent ceni buduce nagrade
                 # GAMMA = 0 - agent gleda samo trenutne nagrade
                 # GAMMA = 1 - agent podjednako ceni trenutne i buduce nagrade


#q-learning zahteva diskretna stanja a mi trenutno imamo kontinualna
#tako da pravim bins za svaku dimenziju

#broj binova za svaku dimenziju stanja
N_X_BINS = 10
N_X_DOT_BINS = 10
N_THETA_BINS = 20           # vise binova jer je theta kritičan
N_THETA_DOT_BINS = 10

#granice za diskretizaciju

#granice za poziciju
x_bins = np.linspace(-2.5, 2.5, N_X_BINS)
#granice brzina - namesteno da se sistem ponasa "realno" po claudovom razmisljanju
x_dot_bins = np.linspace(-3, 3, N_X_DOT_BINS)
#granice za ugao (+- ~15stepeni) pri cemu je radijanska vrednost koriscena
theta_bins = np.linspace(-0.26, 0.26, N_THETA_BINS)  
#granice za ugaonu brzinu moraju biti male kako bi sistem bio stabilan
theta_dot_bins = np.linspace(-2, 2, N_THETA_DOT_BINS)

#zatim pravim q tabelu
Q = np.zeros((N_X_BINS, N_X_DOT_BINS, N_THETA_BINS, N_THETA_DOT_BINS, 2))   # 2 jer imamo dve akcije +10N i -10N

#funkcija za konverziju kontinualnog stanja u diskretni indeks

def discretize_state(state):
    x, x_dot, theta, theta_dot = state
    x_idx = np.digitize(x, x_bins) - 1
    x_dot_idx = np.digitize(x_dot, x_dot_bins) - 1
    theta_idx = np.digitize(theta, theta_bins) - 1
    theta_dot_idx = np.digitize(theta_dot, theta_dot_bins) - 1
    
    x_idx = max(0, min(x_idx, N_X_BINS-1))
    x_dot_idx = max(0, min(x_dot_idx, N_X_DOT_BINS-1))
    theta_idx = max(0, min(theta_idx, N_THETA_BINS-1))
    theta_dot_idx = max(0, min(theta_dot_idx, N_THETA_DOT_BINS-1))
    
    return (x_idx, x_dot_idx, theta_idx, theta_dot_idx)
    

env = Enviroment(100, 1000, 0.5, 0.01)

#epsilon sa decay stopom kako bismo u pocetku istrazivali, a kasnije uzimali greedy akcije
EPSILON = 1
eps_decay_rate = 0.0001 #trebace mi nekih 10000 epizoda da bi EPSILON smanjio na 0
EPISODES = 15000

#politika koju primenjujemo
#sa predavanja 
#zelimo da biramo najvece vrednosti ali opet ponekad istrazujemo
#youuuu geettt theeee bessstttt ooof bothhhh worlddddddsssssssss
def eps_greedy(q, eps: float):
    if random.random() > eps:
        return np.argmax(q)
    else:
        return random.randint(0, 1)

#za pracenje napretka
rewards_per_episode = []
steps_per_episode = []

for episode in range(EPISODES):
    EPSILON -= eps_decay_rate if EPSILON > 0.01 else 0

    #resetujem okruzenje
    state, _, _, _ = env.reset()
    state_idx = discretize_state(state)
    total_reward = 0
    done = False

    #epizoda
    while not done:
        #biram akciju pomocu funk eps_greedy
        action = eps_greedy(Q[state_idx], EPSILON)
        #primenjujem akciju 
        next_state, reward, truncated, terminated = env.take_action(action)
        next_state_idx = discretize_state(next_state)

        #q-learning update formula izgleda ovako
        # ~~ Q(s,a) = Q(s,a) + α(Rapaja kaze beta) * [r + γ * max(Q(s',a')) - Q(s,a)] ~~
        Q[state_idx][action] += ALPHA*(reward + GAMMA*np.max(Q[next_state_idx]) - Q[state_idx][action] ) 

        #predji na sledece stanje
        state_idx = next_state_idx
        total_reward += reward

        #provera da li je epizoda zavrsena
        done = terminated or truncated
    
    #hocu da sacuvam rezultate epizode
    rewards_per_episode.append(total_reward)
    steps_per_episode.append(env.steps)

    #na svakih 1000 epizoda zelim da ispisem napredak
    if(episode + 1) % 1000 == 0:
        avg_reward = np.mean(rewards_per_episode[-1000 :])
        avg_steps = np.mean(steps_per_episode[-1000 :])
        print(f"Epizoda {episode+1}/{EPISODES}, Prosecna nagrada>> {avg_reward:.2f}, Prosecan broj koraka>> {avg_steps:.2f}, Epsilon>>{EPSILON:.2f}")


#vizuelizacija rezultata iako msm da nije potrebna
plt.figure(figsize=(12,5))
plt.subplot(2, 2, 1)
plt.plot(rewards_per_episode)
plt.xlabel('Epizoda')
plt.ylabel('Ukupna nagrada')
plt.title('Napredak tokom treniranja')

plt.subplot(2, 2, 2)
plt.plot(steps_per_episode)
plt.xlabel('Epizoda')
plt.ylabel('Broj koraka')
plt.title('Broj koraka po epizodi')

#moving average prikazuje konvergenciju
#dobro jer se ukloni sum
window = 100
moving_avg = np.convolve(
    rewards_per_episode,
    np.ones(window)/window,
    mode='valid'
)
plt.subplot(2, 2, 3)
plt.plot(moving_avg)
plt.title('Pokretni prosek nagrade (100 epizoda)')


plt.tight_layout()
plt.show()