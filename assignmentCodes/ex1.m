%*************    Econometrics FIN-403    ***************
% Computational exercise of problem set 1
% by
% Wenfei Chu, Shan Jiang, Jiangnan Zhang and Mengjie Zhao
%********************************************************
clear; clc; close all;

%% a) data generation process
n = 20;                 % number of observations
x = 10 * rand(n, 1);    % entries of x and z iid U(0, 1) distributed
z = 10 * rand(n, 1);
epsl = normrnd(0, sqrt(4), [n, 1]); % entries of epsl is iid N(0, 4) distributed
y = 0.5 .* ones(n, 1) + 0.8 .* x + 1.3 .* z + epsl;
X = [ones(n, 1), x, z];

%% b) OLS estimator
b = (X' * X) \ X' * y

%% c) check if b'X'y == y'Xb
if (b' * X' * y - y' * X * b < 1e-5) % if the difference between the two terms are smaller 
                                     % than 1e-5, we consider they are
                                     % equal
    disp('The two terms are equal!')
end

%% d) get M and P, check symmetric and idempotent
getP = @(T)  T * ((T' * T)\T');             % functions to get
getM = @(T) eye(n) - T * ((T' * T)\T');     % M (residual maker) and 
                                            % P (projector)
    
M = getM(X);    % get M
P = getP(X);    % get P
MT = M';        % transpose of M
PT = P';        % transpose of P

% following we consider that if sum of abosolute differences between all entries
% in two matrices are smaller than 1e-5, then the two matrices are considered same
if(sum(sum(abs(MT - M))) < 1e-5)  % check symmetric of M
    disp('Transpose of M is equal to M')
end

if(sum(sum(abs(PT - P))) < 1e-5)  % check symmetric of p
    disp('Transpose of P is equal to P')
end

if(sum(sum(abs(M * M - M))) < 1e-5)  % check idempotent of M
    disp('M is idempotent!')
end

if(sum(sum(abs(P * P - P))) < 1e-5) % check idempotent of P
    disp('P is idempotent!')
end

%% e) FW Theorem verification
X1 = X(:, 1:end-1); % X1 contains the first 2 columns of X
X2 = X(:, end); % X2 contains the last column of X, i.e., z

P1 = getP(X1);  % project from X1 only
M1 = getM(X1);  % residual maker from X1 only

b2 = (X2' * M1' * X2) \ (X2' * M1 * y) 
b   % compare b with b2[3], they are the same

%% f) compute SSE for both complete model and partial model
e_comp = M * y;     % residual from complete model
e_partial = M1 * y; % residual from partial model

SSE_comp = e_comp' * e_comp % SSE from complete model
SSE_partial = e_partial' * e_partial % SSE from partial model

%% g) compute R2 and R_adj_2 for complete and partial mdoel
M0 = eye(n) - ones(n, 1) * ones(1, n)./n; % firstly get the M0 matrix and calculate SST
SST = y' * M0 * y;
% compute R2 and R_adj_2 for complete and partial mdoel respectively
R2_comp = 1 - SSE_comp/SST
R2_adj_comp = 1 - (SSE_comp/(n - 3))/(SST/(n - 1))

R2_partial = 1 - SSE_partial/SST
R2_adj_partial = 1 - (SSE_partial/(n - 3))/(SST/(n - 1))

