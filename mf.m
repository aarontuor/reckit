function model = mf(update, R, Dev, itemdevcold, userdevcold, bothdevcold, k, iterations, maxbadcount, alpha, lambda, verbose, bias)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% Basic Matrix Factorization using gradient descent with L2 regularization.
% 
% Optimizes the following objective function using Batch gradient descent:
% min: ||R - U*P'||^2 + lambda*||U||^2 + lambda*||P||^2
% 
%
% Data (see dataNotes.pdf: 
%    
%   R: training matrix containing user u's rating of item i at R_ui. called train in our data processing
%
%   Dev: dev matrix with same size and format as R. called dev in our data processing
%   
%   itemdevcold: hold out columns. called item_cold_dev in our data processing
%
%   userdevcold: hold out rows. called user_cold_dev in our data processing
%
%   bothdevcold: hold out rows and colums. called both_cold_dev in our data processing
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
	Rmask = spones(R);
    %initialize U and P
	%if update == 'mult'
		[U, P] = initfactors(m, n, k, 'plus');
	%else
    %	[U, P] = initfactors(m, n, k, 'both');
    %end
    %add bias to matrices
    if bias 
        [R, Dev, U, P, mu] = addbias(R, Dev, U, P, update);
		if strcmp(update , 'mult')		
			UnegB = -1*rand(m, 1);
			UnegB = smult(R, UnegB, ones(n,1));
			PnegB = -1*rand(1, n);
			PnegB = smult(R, ones(m, 1), PnegB');
		end
    end
    
    best_dev = Inf; bad_count = 0;  i = 1;
    if strcmp(update, 'stoch') [row, col , val] = find(R); end
    while i <= iterations %&& bad_count <= maxbadcount
        tic    
	
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
				if bias
			        U = U .* (((R  - UnegB - PnegB) *P) ./ max(smult(R, U, P)*P + lambda*U, 1e-10));
						U(:, k + 2) = 1;
			        P = P .* (((R - UnegB - PnegB)'*U) ./ max(smult(R', P, U)*U + lambda*P, 1e-10));
						P(:, k + 1) = 1;
					
					UnegB = UnegB.*div( smult(R, U, P), (R - UnegB - PnegB -lambda*UnegB));
					PnegB = PnegB.*div(smult(R, U, P), (R - UnegB - PnegB -lambda*PnegB));
				else 
									
					U = U .* ( (R*P)  ./ max(smult(R, U, P)*P + lambda*U, 1e-10));
			        P = P .* ( (R'*U) ./ max(smult(R', P, U)*U + lambda*P, 1e-10));
				end				
            otherwise
                fprintf('Invalid update method. Accepted methods:\nbatch\nstoch\nmult\n\n');
                return;
        end % end switch
            
        %Calculate current rmse's (previous round for train for less operations)      
%sparse calculation of Devmask.*(R - U*P')==========================
 	if strcmp(update,  'mult')
		trainRMSE(i) = rms( R - (smult(R, U, P) + UnegB + PnegB), numRRatings);
		[a b c] = find(UnegB);
		d = unique([a c], 'rows');
		ubiasvec = d(:, 2);
		if m ~= size(ubiasvec, 1)
			s = size(ubiasvec, 1)
			size(UnegB)
			full(UnegB(:,1)) 			
			model = 0; 
			return
		end
		devUnegB = smult(Dev, ubiasvec, ones(n,1));
				
		%ubiasvec = sum(UnegB, 2)./sum(spones(UnegB), 2);		
		%devUnegB = smult(Dev, ubiasvec, ones(n,1));
		%pbiasvec = sum(PnegB, 1)./sum(spones(PnegB), 1);
		%devPnegB = smult(Dev, ones(m, 1), pbiasvec');		
		devRMSE(i) = rms( Dev - (smult(Dev, U, P) + devUnegB + 0), numDevRatings);



	else      
		  Devdif = Dev - smult(Dev, U, P);		
%==============================================================		        
		trainRMSE(i) = rms(Rdif, numRRatings);
        devRMSE(i) = rms(Devdif, numDevRatings);        
	end        
		itemcoldRMSE(i) = coldpredict(U, P, itemdevcold, 'item', m, n, mu);
        usercoldRMSE(i) = coldpredict(U, P, userdevcold, 'user', m, n, mu);
        bothcoldRMSE(i) = coldpredict(U, P, bothdevcold, 'both', m, n, mu);
        
		%for early stopping		
        if devRMSE(i) <= best_dev
            bad_count = 0;
            best_dev = devRMSE(i);
        else
            bad_count = bad_count + 1;
            %averageEpoch = mean(t);
        end

        %sanity check: objective function should always be decreasing (for small alpha)
        % ||R - U*P'||^2 + lambda*U + lambda*P	
        
		if strcmp(update, 'mult') && bias
			objHistory(i) = trainRMSE(i) + lambda*(squaredFrob(U) + squaredFrob(P) + squaredFrob(UnegB) + squaredFrob(PnegB));
		else
			objHistory(i) = squaredFrob(Rdif) + lambda*squaredFrob(U) + lambda*squaredFrob(P);		
		end        
		if i ~= 1 && objHistory(i) > objHistory(i - 1)
            fprintf('The objective function just went up. Check your updates!\n');
        end
			
        %print stats
        if verbose        
            printstats(i, trainRMSE(i), devRMSE(i), objHistory(i));
        end
        i = i + 1;
        t(i) = toc;
    end % end while loop
    averageEpoch = mean(t);
    if verbose
        fprintf('Average secs per epoch: %.5f\n', averageEpoch); 
    end
    
    %make model to return//this won't eat up memory since matlab will copy on write
    model = mfmodel(update, R, U, P, alpha, lambda, bias, best_dev, averageEpoch,  trainRMSE, devRMSE, objHistory, itemcoldRMSE, usercoldRMSE, bothcoldRMSE, mu)
    plotrmse(model);
end %end main function
