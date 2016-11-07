classdef mfmodel
    properties
        update
        best_dev
        averageEpoch
        trainRMSE
        devRMSE
        objHistory
        R
        U
        P
        alpha
        lambda
        bias
        mu
        itemcoldRMSE
        usercoldRMSE
        bothcoldRMSE
    end
    
    methods
        function model = mfmodel(update, R, U, P, alpha, lambda, bias, best_dev, averageEpoch, trainRMSE, devRMSE, objHistory, itemcoldRMSE, usercoldRMSE, bothcoldRMSE, mu)
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
        end
        
        
        
        function plotrmse(model) 
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
            end
            hold off;
        end
        
        function rmse = predict(model, X)
            rmse = rms(X - (smult(X, model.U, model.P) + model.mu*spones(X)), nnz(X));
        end
        
        function rmse = coldpredict(model, X, type)
            %makes predictions for held out items and users in data set
            %type can be either 'user', 'item', or 'void' corresponding to user cold start
            %item cold start and the unlikely but possible complete cold start respectively. 
            %(complete cold start is where the user hasn't rated and the item hasn't been rated)
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
            end
        end
        
        function test = plotcold(model, X, coldX, type)
			a = 2;            
			if strcmp(type, 'item')
				rmse(1) = model.coldpredict(coldX, 'item');% zero rated
				plotters(1) = 0;
				                
				num_rated = sum(spones(X), 1);
				for i = 1:(max(num_rated)); 
					if isnan(model.predict(X(:, find(num_rated == i))))
						rmse(i+1) = 0;
					else					
						rmse(i + 1) = model.predict(X(:, find(num_rated == i)));
						plotters(a) = (i);
						a = a + 1;                
					end				
				end				
            end
			if strcmp(type, 'user')
				rmse(1) = model.coldpredict(coldX, 'user');% zero rated
                num_rated = sum(spones(X), 2);
				for i = 1:(max(num_rated)); 
					if isnan(model.predict( X(find(num_rated == i), :)))
						rmse(i+1) = 0;
					else					
						rmse(i + 1) = model.predict( X(find(num_rated == i), :)); 
						plotters(a) = (i);
						a = a + 1;                
					end	

					               
				end			
            end
			bar(plotters, rmse(plotters + 1));	
        end
            
        
        
        
        
        
        
        
        
        
    end
end

    
