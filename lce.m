function [best_dev, averageEpoch, trainRMSE, devRMSE, objHistory, U, P, W] = lce(Q, DevQ, R, Dev, Devcold, k, iterations, maxbadcount, alpha, gamma, lambda, verbose, bias, fixed_seed)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% Modified implementation of Local Collective Embeddings model (skip locality, non-negativity constraints, add bias).
% 
% Optimizes the following objective function using Batch gradient descent:
% min: ||R - U*P'||^2 + ||Q - W*P'||^2 + lambda*U + lambda*P + lambda*W 
% 
% Hyper-parameters:
% 
%     k: number of latent factors.
% 
%     iterations: Maximum number of iterations. 
%
%     maxbadcount: Parameter for early stopping
%
%     alpha: the step-size for the gradient update.
% 	  
%     gamma: in [0, 1] controls the importance of each factorization.
%         Setting alpha = 0.5 gives equal importance to both factorizations, while 
%         values of alpha >0.5 (or $\alpha < 0.5$) give more importance to the 
%         factorization of R or P^T.    
%     
%	  lambda: weight on L_2 Regularization.
%  
%     verbose: true|false. If set to true prints the value of the objective function 
%         evaluated on the training set (R) and the dev set (Dev) in each iteration.
%
%     bias: true|false. If set to true adds bias to factors according to the scheme outlined in notes
%     
%     center: true|false. If set to true subtracts (1 + mean rating) from training and dev matrices and zeros
%			out unobserved ratings to the new centered mean. 
%	
%	  fixed_seed: true|false. If set to true creates the same initial factor matrices for each experiment where the
%				matrix dimensions match. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    addpath('functions');
    m = size(R,1);
    n = size(R,2);
    w = size(Q,1);
    
    %subtract mean from ratings in training and d- gamma*(P_Tdif)ev matrix.
    if bias
        %center R		
        [i, j, v] = find(R);
        %v = v -1;        
        mu = sum(v)/length(v);
        v = v - mu;
        R = sparse(i,j,v, m, n);
		
        %center Dev		- gamma*(P_Tdif)
        [i2, j2, v2] = find(Dev);
        %v2 = v2 -1;        
        mu = sum(v2)/length(v2);
        v2 = v2 - mu;
        Dev = sparse(i2,j2,v2, m, n);
    end        
    
    %calculate constants
    numDevRatings = nnz(Dev);
    numRRatings = nnz(R);
    
    % fix seed for reproducible experiments
    if fixed_seed    
        rand('seed', 354);
    end

    %initialize U, P, and W
    U = 2*rand(m, k) - 1;
    P = 2*rand(n, k) - 1;
    W = 2*rand(w, k) - 1;
    
    %add bias to matrices
    if bias 
        if bias 
        userBias = 2*rand(m,1) -1;
        userMult = ones(m, 1);
        U = [U userBias userMult];
        
        
        itemBias = 2*rand(n,1) -1;
        itemMult = ones(n, 1);
        P = [P itemMult itemBias];
       
        
        wordBias = 2*rand(w,1) -1;
        wordMult = ones(w, 1);
        W = [W wordBias wordMult];
        
    end
    end
    
    best_dev = Inf;
    bad_count = 0;
    i = 1;
    
    while i <= iterations && bad_count <= maxbadcount
        tic    
        %Pre-calculate matrix products and transposes
		%transposes get updated first       
		
        %Update reusable products and differences  
        %sparse calculation of Rmask.*( R - U*P')==========================
        Rdif = R - smult(R, U, P);	
%==============================================================	
%sparse calculation of Devmask.*(R - U*P')==========================
        Devdif = Dev - smult(Dev, U, P);
%==============================================================			
        
        %Calculate current RMSE's        
        %trainRMSE = sqrt(sum(sum(R1dif.^2))/numRRatings);   
        %devRMSE = sqrt(sum(sum(Devdif.^2))/numDevRatings);
        trainRMSE = rms(Rdif, numRRatings);
        devRMSE = rms(Devdif, numDevRatings);        
        devHistory(i) = devRMSE;
        trainHistory(i) = trainRMSE;
      
        %print rmse before gradient update
        if verbose        
            fprintf('iteration: %d\n', i);
            fprintf('\t train rmse: %.5f\n', full(trainRMSE));
            fprintf('\t dev rmse: %.5f\n', full(devRMSE));
        end		
        
        %sanity check: objective function should always be decreasing
        %if gradient is small enough.
        % Omega.*||R - U*P'||^2 + ||Q - W*P'||^2 + lambda*||U||^2 + lambda*||P||^2 + lambda*||W||^2
        objHistory(i+1) = squaredFrob(Rdif) + squaredFrob(Q - W*P') + lambda*squaredFrob(U) + lambda*squaredFrob(P) + lambda*squaredFrob(W);		
        if objHistory(i +1) > objHistory(i)
                fprintf('The objective function just went up. Check your updates, or adjust your step-size.\n');
        %return;
        end
        
        %Update Gradients
        U = U + alpha*((1-gamma)*(Rdif*P - lambda*U));
            if bias U(:, k + 2) = 1; end
        P = P + alpha*((1-gamma)*(R' - smult(R', P, U))*U + gamma*(Q'- P*W')*W - lambda*P);
            if bias P(:, k + 1) = 1; end
        W = W + alpha*(gamma*(Q - W*P')*P - lambda*W);
            if bias W(:, k + 2) = 1; end
        if devRMSE <= best_dev
            bad_count = 0;
            best_dev = devRMSE;
        else
            bad_count = bad_count + 1;
        end
            
        toc
        t(i) = toc;
        i = i + 1;
    end
    averageEpoch = mean(t(1:length(t) -1));
    fprintf('Average time per epoch: %.5f\n', averageEpoch); 
end

