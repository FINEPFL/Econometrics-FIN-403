clear; close all; clc

raw_data = xlsread('Data_week05.xls');
raw_data = raw_data(:, 2:end);

% ---------- structure of data -----------%
% Mkt-RF | SMB | HML | RF | Manuf | HiTec|%

yeM = raw_data(:, 1);

hist(yeM, 60); grid on; title('Histogram of excess market return');
xlabel('excess market return'); ylabel('frequency')
set(gca, 'fontsize', 15)

xdsyem = datastats(yeM);
[xdsManuf, xdsHiTec] = datastats(raw_data(:, 5) ...
                    - raw_data(:, 4), raw_data(:, 6) - raw_data(:, 4));

                skew = skewness(raw_data);
kurt = kurtosis(raw_data);


regressor = yeM;
dependent = raw_data(:, 5) - raw_data(:, 4);

mdl = fitlm(regressor, dependent);

new_estimator = (regressor' * regressor) \ regressor' * dependent;
new_residual  = dependent - regressor * new_estimator;
nonzero = mean(new_residual)