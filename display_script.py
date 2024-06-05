import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np

root = "./results/test/exp1/"
filename_vanilla = [
    "dummy-ackley4_25_qLogNEI_run41.csv",
    "dummy-ackley4_25_qLogNEI_run42.csv",
    "dummy-ackley4_25_qLogNEI_run43.csv",
    "dummy-ackley4_25_qLogNEI_run44.csv",
    "dummy-ackley4_25_qLogNEI_run45.csv"
]
filename_AR = [
    "AR-ackley4_25_qLogNEI_run41.csv",
    "AR-ackley4_25_qLogNEI_run42.csv",
    "AR-ackley4_25_qLogNEI_run43.csv",
    "AR-ackley4_25_qLogNEI_run44.csv",
    "AR-ackley4_25_qLogNEI_run45.csv"
]
filename_AR2 = [
    "AR-2-ackley4_25_qLogNEI_run41.csv",
    "AR-2-ackley4_25_qLogNEI_run42.csv",
    "AR-2-ackley4_25_qLogNEI_run43.csv",
    "AR-2-ackley4_25_qLogNEI_run44.csv",
    "AR-2-ackley4_25_qLogNEI_run45.csv"
]
filename_scheduler = [
    "scheduler-ackley4_25_qLogNEI_run41.csv",
    "scheduler-ackley4_25_qLogNEI_run42.csv",
    "scheduler-ackley4_25_qLogNEI_run43.csv",
    "scheduler-ackley4_25_qLogNEI_run44.csv",
    "scheduler-ackley4_25_qLogNEI_run45.csv"
]
filename_scheduler2 = "scheduler-2-ackley4_25_qLogNEI_run43.csv"
filename_TR = [
    "turbo-ackley4_25_qLogNEI_run41.csv",
    "turbo-ackley4_25_qLogNEI_run42.csv",
    "turbo-ackley4_25_qLogNEI_run43.csv",
    "turbo-ackley4_25_qLogNEI_run44.csv",
    "turbo-ackley4_25_qLogNEI_run45.csv"
]
filename_TR2 = "turbo-2-ackley4_25_qLogNEI_run43.csv"
filename_sigmoid = [
    "sigmoid-ackley4_25_qLogNEI_run41.csv",
    "sigmoid-ackley4_25_qLogNEI_run42.csv",
    "sigmoid-ackley4_25_qLogNEI_run43.csv",
    "sigmoid-ackley4_25_qLogNEI_run44.csv",
    "sigmoid-ackley4_25_qLogNEI_run45.csv"
]

init_num = 20

df_vanilla = [
    pd.read_csv(root + filename_vanilla) for filename_vanilla in filename_vanilla
]
df_AR = [
    pd.read_csv(root + filename_AR) for filename_AR in filename_AR
]
df_AR2 = [
    pd.read_csv(root + filename_AR2) for filename_AR2 in filename_AR2
]
df_scheduler = [
    pd.read_csv(root + filename) for filename in filename_scheduler
]
df_scheduler2 = pd.read_csv(root + filename_scheduler2)
df_TR = [
    pd.read_csv(root + filename_TR) for filename_TR in filename_TR
]
df_TR2 = pd.read_csv(root + filename_TR2)
df_sigmoid = [
    pd.read_csv(root + filename) for filename in filename_sigmoid
]

best_values_vanilla = [
    df['Best Value'].tolist() for df in df_vanilla
]
best_values_AR =[
    df['Best Value'].tolist() for df in df_AR
]
best_values_AR2 = [
    df['Best Value'].tolist() for df in df_AR2
]
best_values_scheduler = [
    df['Best Value'].tolist() for df in df_scheduler
]
best_values_scheduler2 = df_scheduler2['Best Value'].tolist()
best_values_TR = [
    df['Best Value'].tolist() for df in df_TR
]
best_values_TR2 = df_TR2['Best Value'].tolist()
best_values_sigmoid = [
    df['Best Value'].tolist() for df in df_sigmoid
]

# EI = df_scheduler['EI'].tolist()
# EI2 = df_scheduler2['EI'].tolist()

log_best_values_vanilla = [
    [math.log(-v) for v in best_values] for best_values in best_values_vanilla
]
log_best_values_vanilla_mean = [
    sum([lst[i] for lst in log_best_values_vanilla])/5 for i in range(len(log_best_values_vanilla[0]))
]
log_best_values_vanilla_median = [
    sorted([lst[i] for lst in log_best_values_vanilla])[2] for i in range(len(log_best_values_vanilla[0]))
]
log_best_values_vanilla_mad = [
    sorted([abs(lst[i] - log_best_values_vanilla_median[i]) for lst in log_best_values_vanilla])[2] for i in range(len(log_best_values_vanilla[0]))
]
log_best_values_AR = [
    [math.log(-v) for v in best_values] for best_values in best_values_AR
]
log_best_values_AR_mean = [
    sum([lst[i] for lst in log_best_values_AR])/5 for i in range(len(log_best_values_AR[0]))
]
log_best_values_AR_median = [
    sorted([lst[i] for lst in log_best_values_AR])[2] for i in range(len(log_best_values_AR[0]))
]
log_best_values_AR_mad = [
    sorted([abs(lst[i] - log_best_values_AR_median[i]) for lst in log_best_values_AR])[2] for i in range(len(log_best_values_AR[0]))
]
log_best_values_AR2 = [math.log(-v) for v in best_values_AR2]
log_best_values_scheduler = [
    [math.log(-v) for v in best_values] for best_values in best_values_scheduler
]
log_best_values_scheduler_mean = [
    sum([lst[i] for lst in log_best_values_scheduler])/5 for i in range(len(log_best_values_scheduler[0]))
]
log_best_values_scheduler_median = [
    sorted([lst[i] for lst in log_best_values_scheduler])[2] for i in range(len(log_best_values_scheduler[0]))
]
log_best_values_scheduler_mad = [
    sorted([abs(lst[i] - log_best_values_scheduler_median[i]) for lst in log_best_values_scheduler])[2] for i in range(len(log_best_values_scheduler[0]))
]
log_best_values_scheduler2 = [math.log(-v) for v in best_values_scheduler2]
log_best_values_TR = [
    [math.log(-v) for v in best_values] for best_values in best_values_TR
]
log_best_values_TR_mean = [
    sum([lst[i] for lst in log_best_values_TR])/5 for i in range(len(log_best_values_TR[0]))
]
log_best_values_TR_median = [
    sorted([lst[i] for lst in log_best_values_TR])[2] for i in range(len(log_best_values_TR[0]))
]
log_best_values_TR_mad = [
    sorted([abs(lst[i] - log_best_values_TR_median[i]) for lst in log_best_values_TR])[2] for i in range(len(log_best_values_TR[0]))
]
log_best_values_TR2 = [math.log(-v) for v in best_values_TR2]
log_best_values_sigmoid = [
    [math.log(-v) for v in best_values] for best_values in best_values_sigmoid
]
log_best_values_sigmoid_mean = [
    sum([lst[i] for lst in log_best_values_sigmoid])/5 for i in range(len(log_best_values_sigmoid[0]))
]
log_best_values_sigmoid_median = [
    sorted([lst[i] for lst in log_best_values_sigmoid])[2] for i in range(len(log_best_values_sigmoid[0]))
]
log_best_values_sigmoid_mad = [
    sorted([abs(lst[i] - log_best_values_sigmoid_median[i]) for lst in log_best_values_sigmoid])[2] for i in range(len(log_best_values_sigmoid[0]))
]

# log_EI = [math.log(v) for v in EI]
# log_EI2 = [math.log(v) for v in EI2]

# plt.plot(log_best_values_vanilla, label='vanilla BO', color='saddlebrown', marker='o', markevery=20)
# plt.plot(log_best_values_sigmoid, label='soft Winsorization', color='black', marker='*', markevery=20)
plt.plot(log_best_values_vanilla_mean, label='vanilla BO', color='saddlebrown')
plt.fill_between(range(len(log_best_values_vanilla[0])), [mean - mad for mean, mad in zip(log_best_values_vanilla_mean, log_best_values_vanilla_mad)], [mean + std for mean, std in zip(log_best_values_vanilla_mean, log_best_values_vanilla_mad)], color='saddlebrown', alpha=0.3)
plt.plot(log_best_values_sigmoid_mean, label='soft Winsorization', color='black')
plt.fill_between(range(len(log_best_values_sigmoid[0])), [mean - mad for mean, mad in zip(log_best_values_sigmoid_mean, log_best_values_sigmoid_mad)], [mean + std for mean, std in zip(log_best_values_sigmoid_mean, log_best_values_sigmoid_mad)], color='black', alpha=0.3)
plt.plot(log_best_values_AR2, label='AR cool down opt2', color='springgreen', marker='s', markevery=20)
# plt.plot(log_best_values_AR, label='AR cool down opt3', color='green', marker='s', markevery=20)
plt.plot(log_best_values_AR_mean, label='AR cool down opt3', color='green')
plt.fill_between(range(len(log_best_values_AR[0])), [mean - mad for mean, mad in zip(log_best_values_AR_mean, log_best_values_AR_mad)], [mean + std for mean, std in zip(log_best_values_AR_mean, log_best_values_AR_mad)], color='green', alpha=0.3)
plt.plot(log_best_values_scheduler2, label='fixed scheduler opt2', color='lightcoral', marker='^', markevery=20)
# plt.plot(log_best_values_scheduler, label='fixed scheduler opt3', color='red', marker='^', markevery=20)
plt.plot(log_best_values_scheduler_mean, label='fixed scheduler', color='red')
plt.fill_between(range(len(log_best_values_scheduler[0])), [mean - mad for mean, mad in zip(log_best_values_scheduler_mean, log_best_values_scheduler_mad)], [mean + std for mean, std in zip(log_best_values_scheduler_mean, log_best_values_scheduler_mad)], color='red', alpha=0.3)
plt.plot(log_best_values_TR2, label='success/failure counter opt2', color='cornflowerblue', marker='x', markevery=20)
# plt.plot(log_best_values_TR, label='success/failure counter opt3', color='blue', marker='x', markevery=20)
plt.plot(log_best_values_TR_mean, label='success/failure counter opt3', color='blue')
plt.fill_between(range(len(log_best_values_TR[0])), [mean - mad for mean, mad in zip(log_best_values_TR_mean, log_best_values_TR_mad)], [mean + std for mean, std in zip(log_best_values_TR_mean, log_best_values_TR_mad)], color='blue', alpha=0.3)
plt.axvline(x=init_num, color='black', linestyle='--')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Transformed Best Value')
plt.title('Average Performance of Different Strategies')
plt.show()

# plt.plot(EI2, label='fixed scheduler opt2', color='cornflowerblue')
# plt.plot(EI, label='fixed scheduler opt3', color='blue')
# plt.ylim(0, 0.3)  # Add constraint on y-axis
# plt.legend()
# plt.axvline(x=65, color='r', linestyle='--')
# plt.axvline(x=115, color='r', linestyle='--')
# plt.axvline(x=165, color='r', linestyle='--')
# plt.xlabel('Iteration')
# plt.ylabel('Expected Improvement')
# plt.title('EI values for Fixed Scheduler')
# plt.show()

# from torch.quasirandom import SobolEngine

# sobol = SobolEngine(dimension=1)

# def unnest(lst):
#     return [item for sublist in lst for item in sublist]

# raw_sobol_points = sorted(unnest(sobol.draw(n=500).tolist()))
# sobol_points = [20 * (x-0.5) for x in raw_sobol_points]

# f = lambda x : -20 * math.exp(-0.2 * math.sqrt(0.5 * x**2)) - math.exp(0.5 * (1 + math.cos(2 * math.pi* x))) + math.exp(1) + 20
# h = lambda x : math.sin(math.pi * (1+(x-1)/4)) ** 2 + ((1+(x-1)/4)-1)**2 * (1 + 10 * math.sin(math.pi * (1+(x-1)/4) + 1)**2) + ((1+(x-1)/4)-1)**2 * (1 + math.sin(2*math.pi*(1+(x-1)/4))**2)
# k, c = 2, 1.5
# g = lambda y : (y/1.5) / (1 + abs((y/1.5))**k) ** (1/k) # sigmoid
# def sigma_k(y):
#     if y <= 0:
#         return c * (y/c) / (1 + abs((y/c))**k) ** (1/k) 
#     else:
#         return y
# g = sigma_k

# ground_truth_ackley = [f(x) for x in sobol_points]
# ground_truth_levy = [h(x) for x in sobol_points]

# def normalize_list(lst):
#     mean = sum(lst) / len(lst)
#     std = math.sqrt(sum((x - mean) ** 2 for x in lst) / len(lst))
#     return [(x - mean) / std for x in lst]

# # Subtract mean and divide by standard deviation for each sobol_values_*
# ackley_values_normalized = normalize_list(ground_truth_ackley)
# ackley_values_simplified = normalize_list([g(x) for x in ackley_values_normalized])

# levy_values_normalized = normalize_list(ground_truth_levy)
# levy_values_simplified = normalize_list([g(x) for x in levy_values_normalized])

# plt.plot(sobol_points, levy_values_normalized, label='standardized Levy', color='mediumslateblue')
# plt.plot(sobol_points, levy_values_simplified, label="simplified Levy", color='purple')
# plt.plot(sobol_points, ackley_values_normalized, label='standardized Ackley', color='orange')
# plt.plot(sobol_points, ackley_values_simplified, label="simplified Ackley", color='red')
# plt.title("Simplification of Objective Function")
# plt.legend()
plt.show()

# x = np.linspace(-2, 2, 500)

# k_values = [1, 4, 10, 50]

# for k in k_values:
#     y = x / (np.power(1 + np.abs(x)**k, 1/k))
#     plt.plot(x, y, label=f'k = {k}')

# plt.axvline(x=1.0, color='b', linestyle='--')
# plt.axvspan(-2.0, -1.0, alpha=0.3, color='gray', label='68% winsorization')
# plt.axvspan(1.0, 2.0, alpha=0.3, color='gray')
# plt.axvline(x=-1.0, color='b', linestyle='--')
# plt.axhline(y=0, color='black', linestyle='-')
# plt.axvline(x=0, color='black', linestyle='-')
# plt.axhline(y=1.0, color='r', linestyle='--')
# plt.axhline(y=-1.0, color='r', linestyle='--')
# plt.axhline(y=1.0, color='r', linestyle='--')
# plt.axhline(y=-1.0, color='r', linestyle='--')
# plt.xlabel('y')
# plt.ylabel('$\sigma_k(y)$')
# plt.title('Plot of $\sigma_k(\cdot)$ for different $k$ values')
# plt.legend()
# plt.show()


