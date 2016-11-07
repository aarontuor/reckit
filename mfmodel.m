classdef mfmodel
%===============================================================================================================================
%Author: Aaron Tuor
%Last modified: oct. 10, 2015
%hutch research work
%===================================================================================================================
    properties
%========================================================================================================================
        update % string: the update method. supported: 'batch', 'stoch', 'mult'
        best_dev % scalar: best you did on dev in training
        averageEpoch % scalar: average time in seconds for finishing an epoch
        trainRMSE % vector: record of training rmse  
        devRMSE % vector: record of dev rmse
        objHistory % vector: record of objective function value
        R % mXn matrix: training data, called R for mneomic purposes
        U % mXk matrix: matrix of k-long user feature vectors (k + 2 long with bias)
        P % nXk matrix: matrix of k-long item feature vectors (k + 2 long with bias)
        alpha % scalar: gradient step-size
        lambda % scalar: L2 regularization coefficient
        bias % boolean: true|false 
        mu % scalar: the average rating
        itemcoldRMSE % vector: record of item cold rmse
        usercoldRMSE % vector: record of user cold rmse
        bothcoldRMSE % vector: record of both cold rmse
    end % end properties

%==========================================================================================================================================    
    methods
%=================================================================================================
        function model = mfmodel(update, R, U, P, alpha, lambda, bias, best_dev, averageEpoch, trainRMSE, devRMSE, objHistory, itemcoldRMSE, usercoldRMSE, bothcoldRMSE, mu)
%=======================================================================================================================================
%==============constructor function for model. For efficiency purposes the model should only be extended for post training functionality.
%=======================================================================================================================================
            model.bias = bias;
            model.alpha = alpha;
            model.lambda = lambda
            model.U = U;
            model.P = P;
            model.objHistory = objHistory;
            model.trainRMSE = trainRMSE;
            model.devRMSE = devRMSE;
            model.best_dev = best_dev;
            model.update = update;
            model.mu = mu;
            model.R = R;
            model.itemcoldRMSE = itemcoldRMSE;
            model.usercoldRMSE = usercoldRMSE;
            model.bothcoldRMSE = bothcoldRMSE;
            model.averageEpoch = averageEpoch;
        end %end function

%===========================================================================
%===========begin class specific supporting functions
%===========================================================================

%=============================================================================       
        function plotrmse(model) 
%==============================================================================
%=================This function plots the rmse for training data, dev data, 
%=================dev item cold data, dev user cold data, and dev both data.
%============================================================================== 
            k = size(model.U, 2) - 2;
            m = size(model.U, 1);
            n = size(model.P, 2);
            plot(model.trainRMSE);
            hold on;
            plot(model.devRMSE, 'r');
            plot(model.itemcoldRMSE, 'g');
            plot(model.usercoldRMSE, 'c');
            plot(model.bothcoldRMSE, 'm');
            legend('train', 'dev', 'itemcold', 'usercold', 'bothcold','Location','northoutside','Orientation','horizontal');
            xlabel('epoch');
            ylabel('rmse');
            title({['k = ' num2str(k) ' alpha = ' num2str(model.alpha) ' lambda = ' num2str(model.lambda) ' bias = ' num2str(model.bias)]; ['Average secs per epoch: ' num2str(model.averageEpoch) ' Best dev: ' num2str(model.best_dev)]}); 
            if model.bias 
                biasword = 'bias';
            else
                biasword = '';
            end % end if statement
            hold off;
        end % end function

%====================================================================================        
        function rmse = predict(model, X)
%==============================================================================
%=================This function plots the rmse for training data, dev data, 
%============================================================================== 
            rmse = rms(X - (smult(X, model.U, model.P) + model.mu*spones(X)), nnz(X));
        end % end function
        
%===========================================================================================
        function rmse = coldpredict(model, X, type)
%============================================================================================
%============makes predictions for held out items and users in data set
%============type can be either 'user', 'item', or 'both' corresponding to user cold start
%============item cold start and the unlikely but possible complete cold start respectively. 
%==============================================================================================
            if strcmp(type, 'item') %cold item prediction
                [row, col, val] = find(X);
                user_bias_col = size(model.U, 2) - 1;
                rmse = rms(model.U(row, user_bias_col) + model.mu*ones(length(row), 1) - val, length(val));
            elseif strcmp(type, 'user') %cold user prediction
                item_bias_col = size(model.P, 2);
                [row, col, val] = find(X);
                rmse = rms(model.P(col, item_bias_col) + model.mu*ones(length(col), 1) - val, length(val));
            elseif strcmp(type, 'both')
                [row, col, val] = find(X);
                rmse = rms(model.mu*(ones(length(val), 1)) - val, length(val));
            else
                fprintf('Invalid cold prediction type. Accepted types:\nitem\nuser\nboth\n\n');
            end% end if statement
        end% end function

%===============================================================================================
        function plotcold(model, X, coldX, type, binsize)
%===============================================================================================
%=============making pictures of coldish start bins. The y axis is root mean square for the 
%==============bin of x values up to and including the bar position. the zero bar is for unrated items
%================or nonrating users. 
%
%	Arguments:
%		X: held out test set matrix same dimensions as training set matrix
%       type: 'user', or 'item' for user or item cold start plotting
% 		coldX: held out cold test matrix with same # rows as X for items and same number columns
%				of X for users.
%		binsize: how big you want the bins.
%================================================================================================

				switch type
					case 'user'
						num_rated = sum(spones(X), 2);
						max_count = max(num_rated);
						coldpred = model.coldpredict(coldX, 'user');% zero rated
						i = 1;					
						while i <= (max_count)						
							users = zeros(size(X,1), 1);			
							count = 1;						
							while count <= binsize
								users = users + (num_rated == i);	 
								count = count + 1;
								i = i + 1;
							end
							rmse(i -1) = model.predict(spdiags(users, 0, size(X,1), size(X,1))*X);	
						end	% end while block
						rmse = [coldpred rmse];
					case 'item'
						num_rated = sum(spones(X), 1);
						max_count = max(num_rated);
						coldpred = model.coldpredict(coldX, 'item');% zero rated
						i = 1;					
						while i <= (max_count)						
							items = zeros(1, size(X,2));			
							count = 1;						
							while count <= binsize
								items = items + (num_rated == i);	 
								count = count + 1;
								i = i + 1;
							end
							rmse(i -1) = model.predict(X*spdiags(items', 0, size(X,2), size(X,2)));	
						end	% end while block
						rmse = [coldpred rmse];
				end
				bar(0:(length(rmse)-1),  rmse);
        end % end function

%==========================================================================================================================
%==========================================================================================================================
    end % end methods
end % end class

    
