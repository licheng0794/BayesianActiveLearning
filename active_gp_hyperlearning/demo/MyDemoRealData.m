clear;
% rng('default');
%rng(2)
% Setup prediction GP. We'll use a constant mean and a squared
% exponential covariance. We must use mean/covariance functions that
% support an extended GPML syntax allowing the calculation of second
% partial dervivatives with respect to hyperparameters. The
% gpml_extensions package contains implementations of some common
% choices.

model.mean_function       = {@constant_mean};
% model.mean_function       = {@meanZero};
model.covariance_function = {@ard_sqdexp_covariance};
model.likelihood          = @likGauss;


Runtimes = 5;
Results = cell(Runtimes,3);

for rr = 1:Runtimes
    % Setup hyperparameter prior. We'll use independent normal priors on
    % each hyperparameter.
    
    % generate demo data
    load DSTdata.mat
    num_points = size(DSTdata,1);
    
    x_star = DSTdata(:,2:end);
    x_star = (x_star-repmat(mean(x_star,1), [num_points 1]))./ repmat(std(x_star, 0,1), [num_points 1]);
    y_star = DSTdata(:,1);
    d = size(x_star,2);
    % N(0, 0.5^2) priors on each log covariance parameter
    priorcov = cell(1, d+1);
    for i = 1:(d+1)
        priorcov{i} = get_prior(@gaussian_prior, 0, 5);
    end
    priors.cov = priorcov;
    priors.lik  = {get_prior(@gaussian_prior, 0, 1)};
    
    % N(0, 1) prior on constant mean
    priors.mean = {get_prior(@gaussian_prior, 0, 1)};
    
    model.prior = get_prior(@independent_prior, priors);
    model.inference_method = ...
        add_prior_to_inference_method(@exact_inference, model.prior);
    
    
    % setup problem struct
    problem.num_evaluations  = 500;
    
    
    problem.f                = ...
        @(x) (y_star(find(ismember( x_star, x, 'row'))));
    
    initN = 1;
    rndIdx = randperm(num_points, initN);
    problem.initx = x_star(rndIdx,:);
    y = [];
    for i = 1:length(rndIdx)
        y = [y; problem.f(problem.initx(i,:))];
    end
    problem.inity = y;
    
    problem.candidate_x_star = x_star(setdiff(1:num_points, rndIdx),:);
    
    % actively learn GP hyperparameters
    % results.map_hyperparameters is the mean of hyperparameters. the first 1:d is the
    % lengthscales we need
    % results.secondSigma is the variance of hyperparameters
    
    results = learn_gp_hyperparameters(problem, model); % the results for active learning
    results_BAL = results;
    % uncertainty sampling
    results_unc = learn_gp_hyperparameters_unc(problem, model); % the results for uncertainty sampling
    
    % random sampling
    
    HnlZ = cell(1, problem.num_evaluations); % when you need uncertainty for random sampling, please compute the inverse of HnlZ
    map_hyperparameters_random = cell(1, problem.num_evaluations);
    ind = randperm(num_points, problem.num_evaluations);
    
    clear results
    results.secondSigma = cell(1, problem.num_evaluations);
    results.map_embeddings = repmat(model.prior(), [problem.num_evaluations, 1]);
    iter = [50:50:500];
    ni = 1;
    for i = iter
        fprintf('random sampling: %d\n', i)
        x = x_star(ind(1:i), :);
        y = y_star(ind(1:i));
        
        x = [x; problem.initx];
        y = [y; problem.inity];
        
        if (ni > 1)
            initial_hyperparameters = results.map_hyperparameters(iter(ni - 1));
        else
            initial_hyperparameters = model.prior();
        end
        results.map_hyperparameters(i) = ...
            minimize_minFunc(model, x, y, ...
            'initial_hyperparameters', initial_hyperparameters, ...
            'num_restarts',            0, ...
            'minFunc_options',         struct('Display',     'off', ...
            'MaxFunEvals', 500));
        
        [~, ~, results.map_posteriors(i)] = gp(results.map_hyperparameters(i), ...
            model.inference_method, model.mean_function, ...
            model.covariance_function, model.likelihood, x, y);
        
        
        [~, ~, ~, ~, ...
            ~, ~, ~, ~,  ~, ~,   results.secondSigma{i}  ] = ...
            mgp(results.map_hyperparameters(i), model.inference_method, model.mean_function, ...
            model.covariance_function, model.likelihood, x, results.map_posteriors(i), problem.candidate_x_star);
        
        ni = ni + 1;
    end
    
    results_rnd = results;
    
    Results{rr,1} = results_BAL;
    Results{rr,2} = results_unc;
    Results{rr,3} = results;
    
    save OurfinalResults.mat Results
    
end



