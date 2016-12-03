clear; close all; clc

raw_data = xlsread('Data_Week05.xls');

% ---------- structure of data -----------%
% Date | Mkt-RF | SMB | HML | RF | Manuf | HiTec|%
%a)
yeM = raw_data(:, 2);
SMB = raw_data(:, 3);
HML = raw_data(:, 4);
RF = raw_data(:, 5);
hitec = raw_data(:, 7);
manuf = raw_data(:, 6) ;
len = size(raw_data, 1);
N = size(raw_data, 2);

figure; plot(1:length(yeM), yeM, '-', 'color', rand(1, 3)); grid on;
xlabel('timesteps'); ylabel('return'); title('y^e_M')
set(gca, 'fontsize', 15)

figure; plot(1:length(hitec), hitec-RF, '-', 'color', rand(1, 3)); grid on;
xlabel('timesteps'); ylabel('return'); title('hitec')
set(gca, 'fontsize', 15)

figure; plot(1:length(manuf), manuf-RF, '-', 'color', rand(1, 3)); grid on;
xlabel('timesteps'); ylabel('return'); title('manuf')
set(gca, 'fontsize', 15)

%% b)

X = [ones(len, 1), yeM, SMB, HML];
manuf_rf = manuf - RF;
hitec_rf = hitec - RF;

beta_m = regress(manuf_rf, X);
beta_h = regress(hitec_rf, X);

err_m = manuf_rf - X * beta_m;
err_h = hitec_rf - X * beta_h;

%% white test

err_m_2 = err_m.^2;
err_h_2 = err_h.^2;

Xwhite = [X, yeM.^2, SMB.^2, HML.^2, yeM.*SMB, yeM.*HML, SMB.*HML];

beta_m_W = inv((Xwhite)'*Xwhite)*Xwhite'*err_m_2;
beta_h_W = inv((Xwhite)'*Xwhite)*Xwhite'*err_h_2;

err_m_w = err_m_2 - Xwhite * beta_m_W;
err_h_w = err_h_2 - Xwhite * beta_h_W;

SSE_w_m = err_m_w' * err_m_w;
SSE_w_h = err_h_w' * err_h_w;

SST_w_m = (err_m_2 - mean(err_m_2))' * (err_m_2 - mean(err_m_2));
SST_w_h = (err_h_2 - mean(err_h_2))' * (err_h_2 - mean(err_h_2));

SSR_w_m = SST_w_m - SSE_w_m;
SSR_w_h = SST_w_h - SSE_w_h;

F_w_m = SSR_w_m * (len-N)/(SSE_w_m * (N - 1))
F_w_h = SSR_w_h * (len-N)/(SSE_w_h * (N - 1))

F90 = finv( 0.9, N-1, len - N)
F95 = finv(0.95, N-1, len - N)
F99 = finv(0.99, N-1, len - N)

%% bp lm test
g_m = err_m_2 / (err_m_2' * err_m_2 / len) - 1;
g_h = err_h_2 / (err_h_2' * err_h_2 / len) - 1;

beta_bp_m = inv(X' * X) * X' * g_m;
beta_bp_h = inv(X' * X) * X' * g_h;

gr_m = X * beta_bp_m;
gr_h = X * beta_bp_h;

LM_m = 0.5 * (gr_m)' * gr_m
LM_h = 0.5 * (gr_h)' * gr_h

V90 = chi2inv( 0.9, 3)
V95 = chi2inv(0.95, 3)
V99 = chi2inv(0.99, 3)

%% c) 
var_b_m_w = 1/len * inv(X' * X / len) * (X' * (err_m) * (err_m') * X / len) * inv(X' * X / len)
var_b_h_w = 1/len * inv(X' * X / len) * (X' * (err_h) * (err_h') * X / len) * inv(X' * X / len)

t_b_m_w = beta_m./sqrt(diag(var_b_m_w))
t_b_h_w = beta_h./sqrt(diag(var_b_h_w))

m_ols_sse = err_m' * err_m / (len - N);
h_ols_sse = err_h' * err_h / (len - N);

var_b_m_ols = m_ols_sse * inv(X' * X)
var_b_h_ols = h_ols_sse * inv(X' * X)

t_ols_m = beta_m./diag(sqrt(m_ols_sse * inv(X' * X)))
t_ols_h = beta_h./diag(sqrt(h_ols_sse * inv(X' * X)))


close all