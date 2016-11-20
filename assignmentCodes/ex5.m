clear; close all; clc

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

mdl = fitlm(regressor, dependent)

% get s2
s_squared = mdl.Residuals.Raw' * mdl.Residuals.Raw/(mdl.NumObservations - 2);

% get AIC(K) and BIC(K), the information criteria
lnAIC = log(mdl.Residuals.Raw' * mdl.Residuals.Raw/mdl.NumObservations) + ...
                                                2 * 2 / mdl.NumObservations
lnBIC = log(mdl.Residuals.Raw' * mdl.Residuals.Raw/mdl.NumObservations) + ...
                                                2 * log(mdl.NumObservations)/mdl.NumObservations

% In this case we have to fit the model manually as we need to drop the
% constant column in regressor
new_estimator = (regressor' * regressor) \ regressor' * dependent;
new_residual  = dependent - regressor * new_estimator;
nonzero_avg_residual = mean(new_residual);

