%% data generation process
clear; clc; close all;

n = 20;
x = 10 * rand(n, 1);
z = 10 * rand(n, 1);
epsl = normrnd(0, sqrt(4), [n, 1]);
y = 0.5 .* ones(n, 1) + 0.8 .* x + 1.3 .* z + epsl;
X = [ones(n, 1), x, z];

%% OLS estimator
b = (X' * X) \ X' * y

%% c) check if b'X'y == y'Xb
if (b' * X' * y - y' * X * b < 1e-5)
    disp('The two terms are equal!')
end

%% d) get M and P, check symmetric and idempotent
getP = @(T)  T * ((T' * T)\T');
getM = @(T) eye(n) - T * ((T' * T)\T');

M = getM(X);
MT = M';
if(sum(sum(abs(MT - M))) < 1e-5)
    disp('Transpose of M is equal to M')
end

P = getP(X);
PT = P';
if(sum(sum(abs(PT - P))) < 1e-5)
    disp('Transpose of P is equal to P')
end

if(sum(sum(abs(M * M - M))) < 1e-5)  %idempotent
    disp('M is idempotent!')
end

if(sum(sum(abs(P * P - P))) < 1e-5) %idempotent
    disp('P is idempotent!')
end

%% FW Theorem verification
X1 = X(:, 1:end-1);
X2 = X(:, end);

P1 = getP(X1);
M1 = getM(X1);

b2 = (X2' * M1' * X2) \ (X2' * M1 * y) 
b

%% f)
e_comp = M * y;
e_partial = M1 * y;

SSE_comp = e_comp' * e_comp
SSE_partial = e_partial' * e_partial

%% g)
M0 = eye(n) - ones(n, 1) * ones(1, n)./n;

SST = y' * M0 * y;

R2_comp = 1 - SSE_comp/SST
R2_adj_comp = 1 - (SSE_comp/(n - 3))/(SST/(n - 1))

R2_partial = 1 - SSE_partial/SST
R2_adj_partial = 1 - (SSE_partial/(n - 3))/(SST/(n - 1))

