function [best_dev, averageEpoch, trainRMSE, devRMSE, objHistory, U, P, W] = colfac1(Q, DevQ, R, Dev, Devcold, k, iterations, maxbadcount, alpha, gamma, lambda, verbose, bias, center, fixed_seed)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% Collective Matrix Factorization L_8 from the notes.
% 
% Optimizes the following objective function using Batch gradient descent:
% min: ||R - U*P'||^2 + ||P' - W'Q||^2 + lambda*U + lambda*P + lambda*W 
% 
% Matrix Inputs:fprintf('iteration: %d\n', i);
%
% 		fullQ: term/doc matrix for all docs
%		Qmask: a mask that restricts Q to docs of items in train set
%		R: matrix of ratings in train set
%		Dev: matrix of ratings in Dev set
%		Devcold: matrix of ratings for held out items in dev set
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
%  
%  Return Values:
%		objHistory, devRMSE, and trainRMSE are row vectors of values from each epoch
%		U and P are latent factor matrices for users and products respectively
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    m = size(R,1);
    n = size(R,2);
    w = size(Q,1);
    averageEpoch = 0;
    
    %subtract mean from ratings in training and dev matrix.
    if center
        %center R		
        [i, j, v] = find(R);
        %v = v -1;        
        mu = sum(v)/length(v);
        v = v - mu;
        R = sparse(i,j,v, m, n);
		    
      	%center Dev		
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
    Pmask = ones(n, k);
    Umask = ones(m, k);
    Wmask = ones(w, k);
    
    %add bias to matrices
    if bias 
        userBias = 2*rand(m,1) -1;
        userMult = ones(m, 1);
        U = [U userBias userMult];
        Umask = [Umask userMult 0*userMult];
        
        itemBias = 2*rand(n,1) -1;
        itemMult = ones(n, 1);
        P = [P itemMult itemBias];
        Pmask = [Pmask 0*itemMult itemMult];
        
		%this preserves the item bias in the lower factorization
        wordMult = zeros(1, k);	       
		Wmask = [W; wordMult; wordMult];				
		W = [W; wordMult; wordMult];
		wordMult = zeros(w+2, 1);
		Wmask = [W wordMult wordMult];		
		W = [W wordMult wordMult];
 		W(w+2, k+2) = 1;
 		W(w+1, k+1) = 1;
		%this preserves the item bias in the lower factorization
		Q = [Q; ones(1, n); itemBias'];
    end
    
    best_dev = Inf;
    bad_count = 0;
    i = 1;
    P_T = P';
    W_T = W';
    Q_T = Q';
    while i <= iterations && bad_count <= maxbadcount
        tic    
    %Pre-calculate matrix products and transposes
    %transposes get updated first       
        W = W_T';
        P = P_T';
        U_T = U';	
		
        %update the item bias hanging on Q in lower factorization
        if bias
            %Q(w + 2, :) = P(: , k +2);
        end
        %Update reusable products and differences        
        %sparse calculation of Rmask.*( R - U*P')==========================
        UP_T = smult(R, U, P);		
        Rdif = R - UP_T;
%==============================================================	
%sparse calculation of Devmask.*(R - U*P')==========================
        UP_T = smult(Dev, U, P);		
        Devdif = Dev - UP_T;
%==============================================================	
        W_TQ = W_T*Q;   
        P_Tdif =  P_T - W_TQ;       
        
        %Calculate current RMSE's        
        %fullpredictions = U*W_T*fullQ;		
        trainRMSE(i) = rms(Rdif, numRRatings);
        devRMSE(i) = rms(Devdif, numDevRatings); 
        
        %sanity check: objective function should always be decreasing
        %if gradient is small enough.
        % ||R - U*P'||^2 + ||P' - W'Q||^2 + lambda*U + lambda*P + lambda*W
        objHistory(i+1) = squaredfrob(Rdif) + squaredfrob(P_Tdif) + lambda*squaredfrob(U) + lambda*squaredfrob(P) + lambda*squaredfrob(W);		
        if objHistory(i +1) > objHistory(i)
                fprintf('The objective function just went up. Check your updates, or adjust your step-size.\n');
        %return;
        end
        
        %Update Gradients
        U = U + alpha*(((1 - gamma)*Rdif*P - lambda*U));
            if bias U(:, k + 2) = 1; end
        P_T = P_T - alpha*(((1-gamma)*U_T*Rdif + gamma*(W_TQ - P_T) - lambda*P_T));
            if bias P(:, k + 1) = 1; end
        W_T = W_T - alpha*((gamma*(P_T - W_TQ)*Q_T - lambda*Wmask'.*W_T));
            if bias W(:, k + 1) = 1; end
        %Update Gradients
       % U = U + alpha*(Umask.*((1 - gamma)*Rdif*P - lambda*U));
        %P_T = P_T + alpha*(Pmask'.*((1-gamma)*U_T*Rdif - gamma*(P_Tdif) - lambda*P_T));
        %W_T = W_T + alpha*(Wmask'.*(P_Tdif*Q_T - lambda*W_T));     
        
        
        

        %print rmse before gradient update
        if verbose        
            fprintf('iteration: %d\n', i);
            fprintf('\t train rmse: %.5f\n', full(trainRMSE(i)));
            fprintf('\t dev rmse: %.5f\n', full(devRMSE(i)));
            fprintf('\t objective: %.5f\n', full(objHistory(i +1)));
        end		

                                                      
        
		%for early stopping		
        if devRMSE(i) <= best_dev
            bad_count = 0;
            best_dev = devRMSE(i);
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

%auxilliary functions
	function x = squaredfrob(A)
		x = sum(sum(A.*A));
	end

	function x = rms(A, nnzA)
		x = sqrt((sum(sum(A.*A))/nnzA));
	end

    function AB_T = smult(mask, A, B)
		[m, n] = size(mask);
		[i,j,v] = find(mask);		
		indices = 1:length(j);	
		Arows = A(i(indices), :);
		Brows = B(j(indices), :);
		AB_T_values = dot(Arows, Brows, 2);		
		AB_T = sparse(i, j, AB_T_values, m, n);
	end

	function AB_T = sminus(mask, A, B)
		[m, n] = size(mask);
		[i,j,v] = find(mask);		
		indices = 1:length(j);	
		Arows = A(i(indices), :);
		Brows = B(j(indices), :);
		AB_T_values = dot(Arows, Brows, 2);		
		AB_T = sparse(i, j, AB_T_values, m, n);
	end
	
