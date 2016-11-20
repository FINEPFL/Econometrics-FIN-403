clear; close all; clc

%% Exercise I
raw_data = xlsread('Data_week05.xls');
raw_data = raw_data(:, 2:end);

% ---------- structure of data -----------%
% Mkt-RF | SMB | HML | RF | Manuf | HiTec|%

% check histogram and distribution of market return
yeM = raw_data(:, 1);

hist(yeM, 60); grid on; title('Histogram of excess market return');
xlabel('excess market return'); ylabel('frequency')
set(gca, 'fontsize', 15)

% get statistics of data,e.g., mean, variance, min, max, skewness, kurtosis
xdsyem = datastats(yeM);
[xdsManuf, xdsHiTec] = datastats(raw_data(:, 5) ...
                    - raw_data(:, 4), raw_data(:, 6) - raw_data(:, 4));

skew = skewness(raw_data);
kurt = kurtosis(raw_data);


% the fitlm() function includes one columns of ones automatically, thus we
% can simply use yeM
regressor = yeM;
dependent = raw_data(:, 5) - raw_data(:, 4);

mdl = fitlm(regressor, dependent);
SSE_manuf_ori = mdl.Residuals.Raw' * mdl.Residuals.Raw

% get s2
s_squared = mdl.Residuals.Raw' * mdl.Residuals.Raw/(mdl.NumObservations - 2);

% get AIC(K) and BIC(K), the information criteria
lnAIC = log(mdl.Residuals.Raw' * mdl.Residuals.Raw/mdl.NumObservations) + ...
                                                2 * 2 / mdl.NumObservations;
lnBIC = log(mdl.Residuals.Raw' * mdl.Residuals.Raw/mdl.NumObservations) + ...
                           2 * log(mdl.NumObservations)/mdl.NumObservations;

% In this case we have to fit the model manually as we need to drop the
% constant column in regressor
new_estimator = (regressor' * regressor) \ regressor' * dependent;
new_residual  = dependent - regressor * new_estimator;
nonzero_avg_residual = mean(new_residual);

% e)
dependent_hi = raw_data(:, 6) - raw_data(:, 4);
mdl2 = fitlm(regressor, dependent_hi);
% get s2
SSE_hi_ori = mdl2.Residuals.Raw' * mdl2.Residuals.Raw
s_squared_hi = mdl2.Residuals.Raw' * mdl2.Residuals.Raw/(mdl2.NumObservations - 2)

lnAIC_hi = log(mdl2.Residuals.Raw' * mdl2.Residuals.Raw/mdl2.NumObservations) + ...
                                                2 * 2 / mdl2.NumObservations;
lnBIC_hi = log(mdl2.Residuals.Raw' * mdl2.Residuals.Raw/mdl2.NumObservations) + ...
                           2 * log(mdl2.NumObservations)/mdl2.NumObservations;

%% Exercise II

% ---------- structure of data -----------%
% Mkt-RF | SMB | HML | RF | Manuf | HiTec|%
raw_data = xlsread('Data_Week05.xls');

date = raw_data(:,1);
MKt  = raw_data(:,2);
SMB  = raw_data(:,3); 
HML  = raw_data(:,4); 
RF   = raw_data(:,5); 
Manuf = raw_data(:,6); 
HiTec = raw_data(:,7); 
N = length(MKt);

% obtain regressor and dependent then start fitting
X_2_1 = [ones(N,1), MKt, SMB, HML];
[b_manuf, bSTD_manuf, tM_manuf, SSE_manuf, s2_manuf, R2_manuf, ...
        adj_R2_manuf, aic_manuf, bic_manuf] = fitting(Manuf - RF, X_2_1);

[b_hi, bSTD_hi, tM_hi, SSE_hi, s2_hi, R2_hi, ...
        adj_R2_hi, aic_hi, bic_hi] = fitting(HiTec - RF, X_2_1);

% num constriants = 2
J = 2;

F_manuf = (SSE_manuf_ori - SSE_manuf) / (SSE_manuf) * (N - size(X_2_1, 2)) / J;
F_hi = (SSE_hi_ori - SSE_hi)/ (SSE_hi) * (N - size(X_2_1, 2))/J;


% find the break point then seperate data
BP = find(date == 19830103);

yM = Manuf - RF;
yH = HiTec - RF;

X_2_1_1 = X_2_1(1:(BP-1), :);
X_2_1_2 = X_2_1(BP:end, :);

yM1 = yM(1:(BP-1));
yM2 = yM(BP:end);

yH1 = yH(1:(BP-1));
yH2 = yH(BP:end);

% start fitting for manuf
[~, ~, ~, SSE_M_1, ~, ~, ~, ~, ~] = fitting(yM1, X_2_1_1); 
[~, ~, ~, SSE_M_2, ~, ~, ~, ~, ~] = fitting(yM2, X_2_1_2); 
 
SSEM = SSE_M_1 + SSE_M_2;
FM =  (SSE_manuf - SSEM)/SSEM * (N - 2 * size(X_2_1, 2)) / size(X_2_1, 2)

% start fitting for hitec
[~, ~, ~, SSE_H_1, ~, ~, ~, ~, ~] = fitting(yH1, X_2_1_1); 
[~, ~, ~, SSE_H_2, ~, ~, ~, ~, ~] = fitting(yH2, X_2_1_2); 
 
SSEH = SSE_H_1 + SSE_H_2;
FH = (SSE_hi - SSEH ) / SSEH  * (N - 2 * size(X_2_1, 2))/size(X_2_1, 2)



