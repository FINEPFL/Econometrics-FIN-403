%% data generation process
clear; clc; close all;

n = 100;
x = 10 * rand(n, 1);
z = 10 * rand(n, 1);
epsl = normrnd(0, 4, [n, 1]);
y = 0.5 .* ones(n, 1) + 0.8 .* x + 1.3 .* z + epsl;
X = [ones(n, 1), x, z];

%% OLS estimator
b = (X' * X) \ X' * y;
y_hat = X * b;

%% c) check if b'X'y == y'Xb
if (b' * X' * y ~= y' * X * b)
    disp('Equal');
else
    disp('Not Equal');
end
%% d) get M and P, check symmetric and idempotent
getP = @(T)  T * ((T' * T)\T');
getM = @(T) eye(n) - T * ((T' * T)\T');

M = getM(X);
sum(sum(M * M - M)) < 1e-5  %idepotent

P = getP(X);
sum(sum(P * P - P)) < 1e-5  %idepotent

%% FW Theorem verification
X1 = X(:, 1:end-1);
X2 = X(:, end);

P1 = getP(X1);
M1 = getM(X1);

b2 = (X2' * M1' * X2) \ (X2' * M1 * y); % b2 is actually equals to b[3]
%% f)
y_partial = 0.5 .* ones(n, 1) + 0.8 .* x + epsl;
b_partial = (X1' * X1) \ X1' * y_partial;
y_partial_hat = X1 * b_partial;

SSE_partial = sum((y_partial_hat - y_partial).^2)
SSE_comp = sum((y - y_hat).^2)

%% g)
SSR_partial = sum(y_partial_hat.^2);
SST_partial = sum(y_partial.^2);

SSR_comp = sum(y_hat.^2);
SST_comp = sum(y.^2);

R2_partial = SSR_partial/SST_partial
R2_comp = SSR_comp/SST_comp
