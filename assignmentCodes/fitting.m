function [b, bStd, tStats, SSE, s_squared, R2, adj_R2, lnAIC, lnBIC] = fitting(y, X)

 % number of dimensions and number of observations
 K = size(X, 2);
 n = length(y);

 % fitting b
 b = (X' * X) \ X' * y;
   
 % get residuals
 e = y - X * b; 
 
 % SSE calculation, as e_bar = 0
 SSE = e' * e;
 
 % estimate s^2
 s_squared = SSE / (n - K);

 % get STD
 bStd = sqrt(diag(s_squared * inv(X' * X)));
 
 % t-statistic
 tStats = b./bStd;

 % R^2
 R2 = 1- SSE/((y - mean(y))' * (y - mean(y)));
 adj_R2 = 1 - ((1-R2)*(n-1)/(n-K)); 

 % information criterias
 lnAIC = log(SSE / n) + 2 * K /n ;
 lnBIC = log(SSE / n) + K * log(n) /n;  
end
