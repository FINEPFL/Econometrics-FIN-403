clearvars; close all; clc

%% helper functions
getPval_t = @(T, Y) 2*(1-tcdf(abs(T),length(Y)-1));
getPval_f = @(T, Y, n, K) (1-fcdf(abs(T), 2, length(Y)-n-K));
getF = @(R2_lsdv, R2_pool, n, T, K) (R2_lsdv-R2_pool)/(n-1)/(1-R2_lsdv)*(n*T-n-K);

%% Data processing
raw_data = xlsread('Data_Exercises_Set_8.xls');

T = 10;

y1 = raw_data(:, 2);
x1 = raw_data(:, 3);

y2 = raw_data(:, 4);
x2 = raw_data(:, 5);

y3 = raw_data(:, 6);
x3 = raw_data(:, 7);

x = [x1; x2; x3];

%% a. pool the data
y = [y1; y2; y3];
X = [ones(length(y), 1), [x1; x2; x3]];

[b, bStd, tStats, SSE, S2, R2, ~, ~, ~] = fitting(y, X) % ols estimator
pvalues_1 = getPval_t(tStats, y) % get p value of t test

%% b. For fixed effect model, utilise the lesat squares dummy variable estimator

X_2 = [x, kron(eye(3), ones(T, 1))];
[b_2, bStd_2, tStats_2, SSE_2, S2_2, R2_2, ~, ~, ~] = fitting(y, X_2)
pvalues_2 = getPval_t(tStats_2, y)

F_2 = getF(R2_2, R2, 3, T, 1)
pval_F_2 = getPval_f(F_2, y, 3, 1)
F_cv = finv(0.95, 2, length(y)-3-1) % F critical value

%% c. use GLS in fixed effect model estimating
Sigma = S2_2 * eye(T) + (S2-S2_2) * ones(T, T); 
% sigma is the summation calculated from S2LSDV and S2POOL

% Obtain GLS Omiga matrix
Omiga = kron(eye(3), Sigma);

b_3 = inv(X' * inv(Omiga) * X) *  X' / Omiga * y %GLS estimator
b_3_std = sqrt(diag(inv(X' * inv(Omiga) * X)))

b_3_tStats = b_3./b_3_std % get t statistic
pvalues_3 = getPval_t(b_3_tStats, y) % p value fo test

T_cv = tinv(0.95, length(y)-2) % T critical value

%% d. Hausman specification test
D = kron(eye(3), ones(T, 1));
Md = eye(length(y)) - D * inv(D' * D) * D';

AsyVar = S2_2 * inv(x' * Md * x); % asymptotic variance of FE estimator
Var_gls = inv(X' * inv(Omiga) * X); % variance of GLS estimator

chi2 = (b_2(1, 1) - b_3(2, 1))^2 / (AsyVar - Var_gls(2, 2))
Chi_cv = chi2inv(0.95, 1)
pvalue = 1 - chi2cdf(chi2,1)
