function model = oocolfac(Q, DevQ, R, Dev, itemdevcold, userdevcold, bothdevcold, k, iterations, maxbadcount, alpha, gamma, lambda, verbose, bias)
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
    
    

	[i, j, v] = find(R);
        %v = v -1;        
        mu = sum(v)/length(v);
	%subtract mean from ratings in training and dev matrix.
    if bias
        %center R		
        
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
    %if fixed_seed    
    %    rand('seed', 354);
    %end

    %initialize U, P, and W
    U = 2*rand(m, k) - 1;
    P = 2*rand(n, k) - 1;
    W = 2*rand(w, k) - 1;
    
    %add bias to matrices
    if bias  
        userBias = 2*rand(m,1) -1;
        userMult = ones(m, 1);
        U = [U userBias userMult];
        
        
        itemBias = 2*rand(n,1) -1;
        itemMult = ones(n, 1);
        P = [P itemMult itemBias];
       
        
        wordMult = zeros(1, k);	       			
		W = [W; wordMult; wordMult];
		wordMult = zeros(w+2, 1);		
		W = [W wordMult wordMult];
 		W(w+2, k+2) = 1;
 		W(w+1, k+1) = 1;
		%this preserves the item bias in the lower factorization
		Q = [Q; ones(1, n); itemBias'];
    end
    
    best_dev = Inf;
    bad_count = 0;
    i = 1;
    
    while i <= iterations %&& bad_count <= maxbadcount
        tic            
%sparse calculation of Rmask.*( R - U*P')==========================
        Rdif = R - smult(R, U, P);	
%==============================================================	
%sparse calculation of Devmask.*(R - U*P')==========================
        Devdif = Dev - smult(Dev, U, P);
%==============================================================			
        
       %Update Gradients
        U = U + alpha*(((1 - gamma)*Rdif*P - lambda*U));
            if bias U(:, k + 2) = 1; end
        P = P + alpha*((1-gamma)*(R' - smult(R', P, U))*U + gamma*(Q'*W - P) - lambda*P);
            if bias P(:, k + 1) = 1; end
        W = W + alpha*(gamma*Q*(P - Q'*W) - lambda*W);
            if bias 
				W(:, [k + 1, k+2]) = 0; 
				W([w + 1, w+2], :) = 0;
				W(w+2, k+2) = 1;
 				W(w+1, k+1) = 1;
			end									
        
		%Calculate current RMSE's             
        devRMSE(i) = rms(Devdif, numDevRatings); 
        trainRMSE(i) = rms(Rdif, numRRatings);
      	itemcoldRMSE(i) = 0;%coldpredict(U, W, Q, itemdevcold, DevQ, 'skip', m, n, mu);
        usercoldRMSE(i) = 0;%coldpredict(U, W, Q, userdevcold, DevQ, 'skip', m, n, mu);
    	bothcoldRMSE(i) = 0;%coldpredict(U, W, Q, bothdevcold, DevQ, 'skip', m, n, mu);
        
		%for early stopping
		if devRMSE(i) <= best_dev
            bad_count = 0;
            best_dev = devRMSE;
        else
            bad_count = bad_count + 1;
        end

		%sanity check: objective function should always be decreasing for small alpha
        % Omega.*||R - U*P'||^2 + ||Q - W*P'||^2 + lambda*||U||^2 + lambda*||P||^2 + lambda*||W||^2
        objHistory(i) = squaredFrob(Rdif) + squaredFrob(Q - W*P') + lambda*squaredFrob(U) + lambda*squaredFrob(P) + lambda*squaredFrob(W);		
        if i ~= 1 && objHistory(i) > objHistory(i - 1)
            fprintf('The objective function just went up. Check your updates!\n');
        end

		%print rmse
        if verbose        
            printstats(i, trainRMSE(i), devRMSE(i), objHistory(i));
        end		

        toc
        t(i) = toc;
        i = i + 1;
    end
    averageEpoch = mean(t);
    fprintf('Average time per epoch: %.5f\n', averageEpoch); 
	%make model to return//this won't eat up memory since matlab will copy on write
    model = lcemodel(R, U, P, alpha, lambda, bias, best_dev, averageEpoch, trainRMSE, devRMSE, objHistory, itemcoldRMSE, usercoldRMSE, bothcoldRMSE, mu, Q, W, gamma)
    plotrmse(model);
end
%===========================================================================================
        function rmse = coldpredict(U, W, Q, test, testQ, type, m, n, mu)
%============================================================================================
%============makes predictions for held out items and users in data set
%============type can be either 'user', 'item', or 'both' corresponding to user cold start
%============item cold start and the unlikely but possible complete cold start respectively. 
%==============================================================================================
			
            if strcmp(type, 'item') %cold item prediction
                rmse = rms(smult(test, U, testQ'*W), nnz(test));
            elseif strcmp(type, 'user') %cold user prediction
                [row, col, val] = find(test);
                rmse = rms(Q(size(Q,1), col)' + mu*ones(length(col), 1) - val, length(val));
            elseif strcmp(type, 'both')
				testP = testQ'*W;
               
                [row, col, val] = find(test);
                rmse = rms(testP(size(testP, 2), col) + mu*ones(length(col), 1) - val, length(val));
            else
                fprintf('Invalid cold prediction type. Accepted types:\nitem\nuser\nboth\n\n');
				rmse = 0;
            end% end if statement
        end% end function
%==========================================================================================================================
%==========================================================================================================================
