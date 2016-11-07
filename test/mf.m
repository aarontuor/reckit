function model = mf(update, R, Dev, itemdevcold, userdevcold, bothdevcold, k, iterations, maxbadcount, alpha, lambda, verbose, bias)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% Basic Matrix Factorization using gradient descent with L2 regularization.
% 
% Optimizes the following objective function using Batch gradient descent:
% min: ||R - U*P'||^2 + lambda*||U||^2 + lambda*||P||^2
% 
%
% Data: 
%    
%   R: training matrix containing user u's rating of item i at R_ui
%
%   Dev: dev matrix with same size and format as R
%
%
% Hyper-parameters:
%     
%     update: optimization method, options are
%       'batch': batch gradient descent
%       'stoch': stochastic gradient descent
%       'mult': multiplicative updates (bias not currently supported)
%
%     k: number of latent factors.
% 
%     iterations: Maximum number of iterations. 
%
%     maxbadcount: Parameter for early stopping
%
%     alpha: the step-size for the gradient update.
% 
%     lambda: weight on L_2 Regularization.
%  
%     verbose: true|false. If set to True prints the value of the objective function 
%         evaluated on the training set (R) and the dev set (Dev) in each iteration.
%
%     bias: true|false. If set to True adds bias to factors
%  
%	
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    addpath('functions/'); 
    
    %calculate constants
    numDevRatings = nnz(Dev);
    numRRatings = nnz(R);
    m = size(R, 1);
    n = size(R, 2);
    %initialize U and P
    [U, P] = initfactors(m, n, k, update);
    
    mu = 0;
    %add bias to matrices
    if bias 
        [R, Dev, U, P, mu] = addbias(R, Dev, U, P);
    end
    
    best_dev = Inf; bad_count = 0;  i = 1;
    if strcmp(update, 'stoch') [row, col , val] = find(R); end
    while i <= iterations %&& bad_count <= maxbadcount
        tic    
%sparse calculation of Devmask.*(R - U*P')==========================
        Devdif = Dev - smult(Dev, U, P);		
%==============================================================			
%sparse calculation of Rmask.*( R - U*P')==========================
        Rdif = R -  smult(R, U, P);		
%==============================================================	
        %Update Gradients
        switch update
            case 'batch'
                U = U + alpha*((Rdif*P - lambda*U));
                    if bias U(:, k + 2) = 1; end
                P = P + alpha*((R' - smult(R', P, U))*U - lambda*P);
                    if bias P(:, k + 1) = 1; end
            case 'stoch'
                for entry = 1:length(val);
                    item = col(entry);
                    user = row(entry);
                    rating = val(entry);
                    error = rating - dot(U(user, :), P(item, :));
                    
                    %Update parameters
                    U(user, :) = U(user, :) + alpha*(error*P(item, :) - lambda*U(user, :));
                        if bias U(user, k + 2) = 1; end%fix user bias multiplier
                    P(item, :) = P(item, :) + alpha*(error*U(user, :) - lambda*P(item, :));
                        if bias P(item, k + 1) = 1; end%fix item bias multiplier
                end
            case 'mult'
                U = U .* ( (R*P)  ./ max(smult(R, U, P)*P + lambda*U, 1e-10));
                P = P .* ( (R'*U) ./ max(smult(R', P, U)*U + lambda*P, 1e-10));
            otherwise
                fprintf('Invalid update method. Accepted methods:\nbatch\nstoch\nmult\n\n');
                return;
        end
            
        %Calculate current (last round's RMSE's)        
        trainRMSE(i) = rms(Rdif, numRRatings);
        devRMSE(i) = rms(Devdif, numDevRatings);        
        itemcoldRMSE(i) = coldpredict(U, P, itemdevcold, 'item', m, n, mu);
        usercoldRMSE(i) = coldpredict(U, P, userdevcold, 'user', m, n, mu);
        bothcoldRMSE(i) = coldpredict(U, P, bothdevcold, 'both', m, n, mu);
        %for early stopping		
        if devRMSE(i) <= best_dev
            bad_count = 0;
            best_dev = devRMSE(i);
        else
            bad_count = bad_count + 1;
            averageEpoch = mean(t);
        end

        %sanity check: objective function should always be decreasing (for small alpha)
        % ||R - U*P'||^2 + lambda*U + lambda*P	
        objHistory(i) = squaredFrob(Rdif) + lambda*squaredFrob(U) + lambda*squaredFrob(P);		
        if i ~= 1 && objHistory(i) > objHistory(i - 1)
            fprintf('The objective function just went up. Check your updates!\n');
            %return;
        end
			
        %print stats
        if verbose        
            printstats(i, trainRMSE(i), devRMSE(i), objHistory(i));
        end
        i = i + 1;
        t(i) = toc;
    end
    averageEpoch = mean(t);
    if verbose
        fprintf('Average secs per epoch: %.5f\n', averageEpoch); 
    end
    
    %make model to return//this won't eat up memory since matlab will copy on write
    model = mfmodel(update, R, U, P, alpha, lambda, bias, best_dev, averageEpoch,  trainRMSE, devRMSE, objHistory, itemcoldRMSE, usercoldRMSE, bothcoldRMSE, mu)
    plotrmse(model);
end