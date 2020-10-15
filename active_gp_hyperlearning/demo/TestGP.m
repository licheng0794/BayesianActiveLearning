 meanfunc = @meanConst;                    % empty: don't use a mean function
 covfunc = @covSEard;              % Squared Exponental covariance function
 likfunc = @likGauss;              % Gaussian likelihood
 hyp = struct('mean', 1, 'cov', zeros(1,35), 'lik', -1);
 hyp2 = minimize(hyp, @gp, -500, @infExact, meanfunc, covfunc, likfunc, x_star, y_star);