classdef lcemodel < mfmodel
%===============================================================================================================================
%Author: Aaron Tuor
%Last modified: oct. 10, 2015
%hutch research work
%===================================================================================================================
    properties
%========================================================================================================================
		Q % wXn matrix: term doc matrix
	    W % wXk matrix: word feature matrix        
        gamma % scalar: controls the importance of each factorization
    end % end properties

%==========================================================================================================================================    
    methods
%=================================================================================================
        function model = lcemodel(R, U, P, alpha, lambda, bias, best_dev, averageEpoch, trainRMSE, devRMSE, objHistory, itemcoldRMSE, usercoldRMSE, bothcoldRMSE, mu, Q, W, gamma)
%=======================================================================================================================================
%==============constructor function for model. For efficiency purposes the model should only be extended for post training functionality.
%=======================================================================================================================================
            model = model@mfmodel('batch', R, U, P, alpha, lambda, bias, best_dev, averageEpoch, trainRMSE, devRMSE, objHistory, itemcoldRMSE, usercoldRMSE, bothcoldRMSE, mu);
			model.W = W;
			model.Q = Q;
			model.gamma = gamma;
        end %end function

%===========================================================================
%===========begin class specific supporting functions
%===========================================================================
%=============================================================================       
      
%===========================================================================================
        function rmse = coldpredict(model, X, Q_cold, type)
%============================================================================================
%============makes predictions for held out items and users in data set
%============type can be either 'user', 'item', or 'both' corresponding to user cold start
%============item cold start and the unlikely but possible complete cold start respectively. 
%==============================================================================================
			p_vecs = model.W/Q;
            p_vecs(:, size(p_vecs, 1) - 1) = 1;
            if strcmp(type, 'item') %cold item prediction
                pred = spones(X).*(model.U*p_vecs);
                rmse = rms(X - pred); 
            elseif strcmp(type, 'user') %cold user prediction
                item_bias_col = size(model.P, 2);
                [row, col, val] = find(X);
                rmse = rms(model.P(col, item_bias_col) + model.mu*ones(length(col), 1) - val, length(val));
            elseif strcmp(type, 'both')
                item_bias_col = size(p_vecs, 1);
                [row, col, val] = find(X);
                 rmse = rms(pvecs(col, item_bias_col) + model.mu*ones(length(col), 1) - val, length(val));
            else
                fprintf('Invalid cold prediction type. Accepted types:\nitem\nuser\nboth\n\n');
            end% end if statement
        end% end function
%==========================================================================================================================
%==========================================================================================================================
    end % end methods
end % end class

    
