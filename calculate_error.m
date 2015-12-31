function [train_err, test_err] = calculate_error (D, train_features, train_targets, test_features, test_targets, region, Nclasses)

% Calculate error (used by the main calculation functions)

train_err   = zeros(Nclasses+1,1);
test_err    = zeros(Nclasses+1,1);

if ~isempty(train_targets),    
    [classify, err]  = classification_error(D, train_features, train_targets, region);
    for j = 1:Nclasses,
        train_err(j)   = 1 - classify(j,j);    
    end
    train_err(Nclasses+1) = err;     
end

if ~isempty(test_targets),
    [classify, err]  = classification_error(D, test_features, test_targets, region);
    for j = 1:Nclasses,
       test_err(j)   = 1 - classify(j,j);    
    end
    test_err(Nclasses+1) = err;     
end
