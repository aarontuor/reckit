function rmse = coldpredict(U, P, X, type, m, n, mu)
    %makes predictions for held out items and users in data set
    %type can be either 'user', 'item', or 'void' corresponding to user cold start
    %item cold start and the unlikely but possible complete cold start respectively. 
    %(complete cold start is where the user hasn't rated and the item hasn't been rated)
    
    if strcmp(type, 'item')%cold item prediction
        [row, col, val] = find(X);
        user_bias_col = size(U, 2) - 1;
        rmse = rms(U(row, user_bias_col) + mu*ones(length(row), 1) - val, length(val));
    elseif strcmp(type, 'user') %cold user prediction
        item_bias_col = size(P, 2);
        [row, col, val] = find(X);
        rmse = rms(P(col, item_bias_col) + mu*ones(length(col), 1) -val, length(val));
    elseif strcmp(type, 'both') %cold user-item prediction
        [row, col, val] = find(X);
        rmse = rms(mu*(ones(length(val), 1)) - val, length(val));
    else
        fprintf('Invalid cold prediction type. Accepted types:\nitem\nuser\nboth\n\n');
        rmse = -1;
    end
end