from scipy.optimize import root_scalar
import numpy as np
from bandits import GaussianBandit
from math import log, sqrt

# Instance initialization:

# mu_0 = [1.5, 1.45, 1.43, 1.4]
# mu_1 = [2.6, 2.7, 2.58, 2.54]
# K = 4

# tau = 2.5

# B = GaussianBandit(K, tau, mu_0, mu_1, [1, 1])

# Finding Optiaml Weights:

def constants(B):
    """
    Returns constants corresponding to bandit instance B
    """
    opt_arm = B.optimal_arm
    C1 ,C2, C3 = [], [], []

    for a in range(B.K):
        C1.append((B.mu_0[opt_arm] - B.mu_0[a])**2/(B.sigma[0]**2))
        C2.append((B.mu_1[a] - B.tau)**2/(B.sigma[1]**2))
        C3.append((B.mu_1[opt_arm] - B.mu_1[a])**2/(B.sigma[1]**2))

    return opt_arm, C1, C2, C3

def x_a(B, opt_arm, C1, C2, C3, a, y):
    """
    Returns inverse of g_a for a given value y
    """
    if B.mu_1[opt_arm] < B.tau:
        if B.mu_1[a] < B.tau:
            return 2*y/(C1[a] - 2*y)
        elif B.mu_0[a] < B.mu_0[opt_arm]:
            return (2*y - C1[a] - C2[a] + ((2*y - C1[a] - C2[a])**2 + 8*y*C2[a])**0.5) \
                     /(2*C2[a])
        else:
            return 2*y/C2[a]
    else:
        return 2*y/(C3[a] - 2*y)
    
def der_x_a(B, opt_arm, C1, C2, C3, a, y):
    """
    Returns the derivative of inverse of g_a for a given value y
    """
    if B.mu_1[opt_arm] < B.tau:
        if B.mu_1[a] < B.tau:
            return 2*C1[a]/(C1[a] - 2*y)**2
        elif B.mu_0[a] < B.mu_0[opt_arm]:
            return 1/C2[a] + (2*y - C1[a] + C2[a])/(C2[a]*((2*y - C1[a] - C2[a])**2 + 8*y*C2[a])**0.5)
        else:
            return 2/C2[a]
    else:
        return 2*C3[a]/(C3[a] - 2*y)**2

def y_max(B, opt_arm, C1, C2, C3, eps=0.00001):
    """
    Find the range of min[g_a(x_a)]
    """
    y_max = C2[opt_arm]/2

    if B.mu_1[opt_arm] < B.tau:
        for a in range(K):
            if a != opt_arm and B.mu_1[a] < B.tau:
                y_max = min(C1[a]/2 - eps, y_max)
    else:
        for a in range(K):
            if a != opt_arm:
                y_max = min(C3[a]/2 - eps, y_max)

    return y_max

# def f(y):
#     """
#     Find root of f to obtain y_star/y_tilda
#     """
#     num = 1
#     den = 0
#     for a in range(K):
#         if a != opt_arm:
#             num += x_a(B, opt_arm, C1, C2, C3, a, y)
#             den += der_x_a(B, opt_arm, C1, C2, C3, a, y)

#     return y - (num/den)

def y_star(f, y_max):
    """
    Solve for y_star
    """
    try:
        sol = root_scalar(f, bracket=[0, y_max], method='brentq')
        y_star = sol.root
    except ValueError:
        y_star = y_max
        
    return y_star

def alpha(B, opt_arm, C1, C2, C3, y):
    den = 1
    for a in range(K):
        if a != opt_arm:
            den += x_a(B, opt_arm, C1, C2, C3, a, y)
    alpha = []
    for a in range(K):
        if a != opt_arm:
            alpha.append(x_a(B, opt_arm, C1, C2, C3, a, y)/den)
        else:
            alpha.append(1/den)
    
    return alpha

# opt_arm, C1, C2, C3 = constants(B)

# y_max_B = y_max(B, opt_arm, C1, C2, C3)
# y_star = y_star(f, y_max_B)
# alpha_star = alpha(B, opt_arm, C1, C2, C3, y_star)
# print(alpha_star)

# D-Tracking:

def d_tracking(K, tau, emp_mu_0, emp_mu_1, sigma, t, N_t, U_t):
    """
    A single step of the D-tracking algorithm
    """
    B_t = GaussianBandit(K, tau, emp_mu_0, emp_mu_1, sigma)
    
    opt, C1, C2, C3 = constants(B_t)

    y_max_t = y_max(B_t, opt, C1, C2, C3)

    def f_t(y):
        """
        Find root of f to obtain y_star/y_tilda
        """
        num = 1
        den = 0
        for a in range(K):
            if a != opt:
                num += x_a(B_t, a, C1, C2, C3, a, y)
                den += der_x_a(B_t, a, C1, C2, C3, a, y)

        return y - (num/den)
    
    y_star_t = y_star(f_t, y_max_t)

    alpha_star_t = alpha(B_t, opt, C1, C2, C3, y_star_t)

    if U_t == []:
        return np.argmax(t*np.array(alpha_star_t) - np.array(N_t)), alpha_star_t
    else:
        return np.argmin(np.array(N_t)), alpha_star_t

# _, alpha = d_tracking(2, 2.5, [2, 1.5], [1, 2.6], [2, 1], 1, [0, 0], [])
# print(alpha)

# Stopping Rule:

def Z_ab_t(tau, emp_mu_a, emp_mu_b, sigma, N_a, N_b):
    """
    switching the answer, assuming arm a is the better among the two
    """
    emp_mu_ab_0 = (N_a*emp_mu_a[0] + N_b*emp_mu_b[0])/(N_a + N_b)
    emp_mu_ab_1 = (N_a*emp_mu_a[1] + N_b*emp_mu_b[1])/(N_a + N_b)

    if emp_mu_a[1] < tau:
        if  emp_mu_b[1] < tau:
            return (N_a/(2*sigma[0]**2))*(emp_mu_a[0] - emp_mu_ab_0)**2 + (N_b/(2*sigma[0]**2))*(emp_mu_b[0] - emp_mu_ab_0)**2
        elif emp_mu_b[1] > tau and emp_mu_b[0] < emp_mu_a[0]:
            return (N_a/(2*sigma[0]**2))*(emp_mu_a[0] - emp_mu_ab_0)**2 + (N_b/(2*sigma[0]**2))*(emp_mu_b[0] - emp_mu_ab_0)**2 \
                + (N_b/(2*sigma[1]**2))*(emp_mu_b[1] - tau)**2 
        else:
            return (N_b/(2*sigma[1]**2))*(emp_mu_b[1] - tau)**2
    else:
        return (N_a/(2*sigma[1]**2))*(emp_mu_a[1] - emp_mu_ab_1)**2 + (N_b/(2*sigma[1]**2))*(emp_mu_b[1] - emp_mu_ab_1)**2

def Z_gamma_t(tau, emp_mu_a, sigma, N_a):
    """
    Changing the feasibility criterion of the 2-armed instance 
    """
    return (N_a/(2*sigma[1]**2))*(emp_mu_a[1] - tau)**2

def Z_t(K, tau, emp_mu_0, emp_mu_1, sigma, N_t):
    """
    Z(t) calculation
    """
    B_t = GaussianBandit(K, tau, emp_mu_0, emp_mu_1, sigma)
    
    a = B_t.optimal_arm
    emp_mu_a = [emp_mu_0[a], emp_mu_1[a]]

    Z_abg_t = []

    for b in range(K):
        if b != a:
            emp_mu_b = [emp_mu_0[b], emp_mu_1[b]]
            Z_abg_t.append(min(Z_ab_t(tau, emp_mu_a, emp_mu_b, sigma, N_t[a], N_t[b]), Z_gamma_t(tau, emp_mu_a, sigma, N_t[a])))   

    return min(Z_abg_t)

# Track-and-stop algorithm:

def track_and_stop(B, delta = 0.1):
    """
    Performs track-and-stop algorithm on a given bandit intance 
    for a fixed confidence delta
    """
    emp_mu_0 = []
    emp_mu_1 = []
    sigma = []
    N_t = []

    mu_0 = 0
    mu_1 = 0
    for a in range(K):
        X, Y = B.pull(a)
        mu_0 = X
        mu_1 = Y
        emp_mu_0.append(mu_0)
        emp_mu_1.append(mu_1)
        N_t.append(1)

    sigma = B.sigma

    t = sum(N_t)
    while Z_t(K, tau, emp_mu_0, emp_mu_1, sigma, N_t) < log((log(t) + 1)/(delta)):

        U_t = []

        for a in range(K):
            if N_t[a] < sqrt(t) - K/2:
                U_t.append(a)
            
        a_t, _ = d_tracking(K, tau, emp_mu_0, emp_mu_1, sigma, t, N_t, U_t)
        
        temp_0 = N_t[a_t]*emp_mu_0[a_t]
        temp_1 = N_t[a_t]*emp_mu_1[a_t]
        N_t[a_t] += 1
        t += 1
        X, Y = B.pull(a_t)
        emp_mu_0[a_t] = (temp_0 + X)/N_t[a_t]
        emp_mu_1[a_t] = (temp_1 + Y)/N_t[a_t]

    B_predict = GaussianBandit(K, tau, emp_mu_0, emp_mu_1, sigma)

    a_predict = B_predict.optimal_arm

    if emp_mu_1[a_predict] < tau:
        return a_predict, 0, t
    else:
        return a_predict, 1, t

# Action Elimination:

# Confidence bound:

def conf_bound(K, c, t, delta, sigma):
    """
    Confidence bounds for the objective and constraint after t rounds
    """
    obj_bound = sigma[0]*sqrt((1/(2*t))*log((c*K*(t**2))/(delta)))
    cnst_bound = sigma[1]*sqrt((1/(2*t))*log((c*K*(t**2))/(delta)))
    return obj_bound, cnst_bound

def U_t(K, emp_mu_0, emp_mu_1, c, N_t, delta, sigma):
    """
    Upper confidence bound after t rounds for and constraint
    """
    obj_bound, cnst_bound = conf_bound(K, c, N_t, delta, sigma)

    U_0t = []
    U_1t = []
    for a in range(K):
        U_0t.append(emp_mu_0[a] + obj_bound)
        U_1t.append(emp_mu_1[a] + cnst_bound)
    
    U_t = [U_0t, U_1t]

    return U_t

def L_t(K, emp_mu_0, emp_mu_1, c, t, delta, sigma):
    """
    Lower confidence bound after t rounds for and constraint
    """
    obj_bound, cnst_bound = conf_bound(K, c, t, delta, sigma)

    L_0t = []
    L_1t = []
    for a in range(K):
        L_0t.append(emp_mu_0[a] - obj_bound)
        L_1t.append(emp_mu_1[a] - cnst_bound)
    
    L_t = [L_0t, L_1t]

    return L_t

# Action Elimination algorithm:

def action_elimination(B, delta = 0.1, c = 5):
    """
    Performs action-elimintion algorithm on a given bandit intance 
    for a fixed confidence delta
    """

    emp_mu_0 = []
    emp_mu_1 = []
    sigma = []
    N_t = []

    mu_0 = 0
    mu_1 = 0
    for a in range(K):
        X, Y = B.pull(a)
        mu_0 = X
        mu_1 = Y
        emp_mu_0.append(mu_0)
        emp_mu_1.append(mu_1)
        N_t.append(1)

    sigma = B.sigma

    active_arms = range(K)
    t = 1
    while len(active_arms) > 1:

        for a in active_arms:
            temp_0 = N_t[a]*emp_mu_0[a]
            temp_1 = N_t[a]*emp_mu_1[a]
            N_t[a] += 1
            X, Y = B.pull(a)
            emp_mu_0[a] = (temp_0 + X)/N_t[a]
            emp_mu_1[a] = (temp_1 + Y)/N_t[a]

        U = U_t(K, emp_mu_0, emp_mu_1, c, t, delta, sigma)
        L = L_t(K, emp_mu_0, emp_mu_1, c, t, delta, sigma)

        likely_feasible_arms  = []
        for a in active_arms:
            if U[1][a] < tau:
                likely_feasible_arms.append(a)
            
        if likely_feasible_arms != []:
            a_0 = likely_feasible_arms[0]
            for a in likely_feasible_arms:
                if emp_mu_0[a] > emp_mu_0[a_0]:
                    a_0 = a
            a_1 = active_arms[0]
            for a in active_arms:
                if emp_mu_1[a] < emp_mu_1[a_1]:
                    a_1 = a
            new_active_arms = []           
            for a in active_arms:
                if (L[0][a_0] < U[0][a]) and (L[1][a] < U[1][a_1] or L[1][a] < tau):
                    new_active_arms.append(a)
        else:
            a_1 = active_arms[0]
            for a in active_arms:
                if emp_mu_1[a] < emp_mu_1[a_1]:
                    a_1 = a
            new_active_arms = []
            for a in active_arms:
                if (L[1][a] < U[1][a_1] or L[1][a] < tau):
                    new_active_arms.append(a)
    
        active_arms = new_active_arms
        t += 1

    a_predict = active_arms[0]

    if emp_mu_1[a_predict] < tau:
        return a_predict, 0, sum(N_t)
    else:
        return a_predict, 1, sum(N_t)
    
# Monte-carlo simulation:

def monte_carlo(B, delta, correct_ans, algo, no_of_sims):
    """
    Performs Monte-carlo simultation using algo on instance B
    """

    no_of_correct = 0
    expected_stopping_time = 0

    for _ in range(no_of_sims):
        a_pred, f, T = algo(B, delta)

        if (a_pred, f) == correct_ans:
            no_of_correct += 1

        expected_stopping_time += T
    
    expected_stopping_time = expected_stopping_time/no_of_sims
    correct_prob = no_of_correct/no_of_sims

    return correct_prob, expected_stopping_time

# Main program:

if __name__ == "__main__":

    # Instance Initialization

    mu_0 = [[1.5, 1.4, 1.3, 1.2], [2.5, 2, 1.5, 1], [1.5, 1.4, 1.3, 1.2], [1.5, 1.42, 1.47, 1.38]]
    mu_1 = [[2.9, 2.7, 2.6, 2.54], [3, 2, 3, 1.5], [2.7, 2.4, 2.2, 2.6], [2.58, 2.41, 2.23, 2.64]]
    correct_answers = [(3, 1), (1, 0), (1, 0), (2, 0)]
    K = 4
    tau = 2.5

    B = []
    for i in range(4):
        B.append(GaussianBandit(K, tau, mu_0[i], mu_1[i], [1, 1]))

    betas = [0.1, 0.05, 0.01]
    # Monte-Carlo Simulation

    no_of_sims = 1000
    for i in range(3, 4):
        for beta in betas:
            correct_prob_ts, T_ts = monte_carlo(B[i], beta, correct_answers[i], track_and_stop, no_of_sims)
            correct_prob_ae, T_ae = monte_carlo(B[i], beta, correct_answers[i], action_elimination, no_of_sims)

            print(f"Bandit Instance: {i}, Beta: {beta}, Track-and-Stop ---> Correct Probability: {correct_prob_ts}, Expected Stopping Time: {T_ts}")
            print(f"Bandit Instance: {i}, Beta: {beta}, Action Elimination ---> Correct Probability: {correct_prob_ae}, Expected Stopping Time: {T_ae}\n")

    

    



