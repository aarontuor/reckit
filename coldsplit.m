function result = coldsplitword(fullratings, termdocfilename, filename)
%=======================================================================================
%
% given an mXn matrix X, splits data 70/5/5/5/5/5/5, train/dev/test/item_cold_dev/item_cold_test/user_cold_dev/user_cold_test
% where the rows and columns of X are shuffled, item_cold_test is the rightmost columns (about 5% ratings), 
% item_cold_dev is the next rightmost columns (about 5% ratings), user_cold_test is the 
% bottomost columns(about 5% ratings), user cold_dev is the next bottomost columns(about 5% ratings) 
% and dev and test are randomly drawn w/out replacement from remaining rows and columns of matrix
%
% filename should be of the form 'filepath/datasetSubset.mat'
% for instance if I am making matrices for Amazon Movies and I want my folder in data nested below the current director
% filename would be something like 'data/amazonMovies.mat'
%
% saves data to .mat file containing sparse matrices:
% 	
%
%       fullratings: X
%	item_cold_dev: m X a
%	item_cold_test: m X b
%	user_cold_dev: c X n
%	user_cold_test: d X n
%	train: (m - a - b) X (n - c - d)
%	dev: (m - a - b) X (n - c - d)
%	item_order: the item permutation 
%		(in case additional data (like word counts needs to be processed from this set)
%   user_order: the user permutation 
%========================================================================================
	%return is zero if something goes wrong w/split and one if the split is good
	result = 0;
	
	%load full set of ratings
	m = size(X, 1);
	n = size(X, 2);
	fullratings = X;
	clear X;
    %permute row and columns for random selection of cold start data
	item_order = randperm(size(fullratings, 2));
	user_order = randperm(size(fullratings, 1));
	fullratings = fullratings(:, item_order);
	fullratings = fullratings(user_order, :);
	
	%calculate split size
	num_ratings = nnz(fullratings);
	split_size = num_ratings*0.05;
%---------------------------------------------------------------------------------------------	
	%find test_cold set
	i = n + 1;
	num_cold_test = 0;
	while num_cold_test < split_size
		i = i - 1;
		num_cold_test = num_cold_test + nnz(fullratings(:,i));
	end
	start_item_test_cold = i;
	%item_test_cold = fullratings(:, start_test_cold:n);
	
	%find item_dev_cold set
	num_cold_dev = 0;
	end_item_dev_cold = i -1;
	while num_cold_dev < split_size
		i = i - 1;
		num_cold_dev = num_cold_dev + nnz(fullratings(:, i));
	end
	start_item_dev_cold = i;
	%item_dev_cold = fullratings(:, start_dev_cold:end_dev_cold);
	non_cold_items = 1:(start_item_dev_cold - 1);

%======================================================================
%find user_test_cold set====================================
%========================================

	i = m + 1;
	num_cold_test = 0;
	while num_cold_test < split_size
		i = i - 1;
		num_cold_test = num_cold_test + nnz(fullratings(i,:));
	end
	start_user_test_cold = i;
	%user_test_cold = fullratings(start_test_cold:m, :);
	
	%find user_dev_cold set
	num_cold_dev = 0;
	end_user_dev_cold = i -1;
	while num_cold_dev < split_size
		i = i - 1;
		num_cold_dev = num_cold_dev + nnz(fullratings(i, :));
	end
	start_user_dev_cold = i;
	%user_dev_cold = fullratings(start_dev_cold:end_dev_cold, :);
        non_cold_users = 1:(start_user_dev_cold - 1);
%==============================	
%make 6 cold split matrices=================================================================
%================================        
        item_test_cold = fullratings(non_cold_users, start_item_test_cold:n);
        item_dev_cold = fullratings(non_cold_users, start_item_dev_cold:end_item_dev_cold);
        user_test_cold = fullratings(start_user_test_cold:m, non_cold_items);
        user_dev_cold = fullratings(start_user_dev_cold:end_user_dev_cold, non_cold_items);
	both_cold = fullratings(start_user_dev_cold:m, start_item_dev_cold:n);
	%both_dev_cold = both_cold(:, 1:floor(size(both_cold, 2)/2));
	%both_test_cold = both_cold(:,(floor(size(both_cold, 2)/2) + 1):(size(both_cold, 2)));
	[row, col, val] = find(both_cold);
	num_cold_ratings = length(row);
	split = randperm(1:num_cold_ratings);
	dev_split_size = floor(length(row)/2);
	test_split_size = ceil(length(row)/2);
	both_dev_cold = sparse(row(split(1:dev_split_size)), col(split(1:dev_split_size)), val(split(1:dev_split_size)), size(both_cold, 1), size(both_cold, 2));
	both_test_cold = sparse(row(split(test_split_size:num_cold_ratings)), col(split(test_split_size:num_cold_ratings)), val(split(test_split_size:num_cold_ratings)), size(both_cold, 1), size(both_cold, 2));
	clear both_cold;
%=====================	
%find dev set======================================================================
%=======================	
	
	
	[i, j, s] = find(fullratings(non_cold_users, non_cold_items));
	num_non_cold = length(i);
	split_size = round(split_size);
	
	drawlist = randperm(length(i));
	
	non_cold_users = length(non_cold_users);
	non_cold_items = length(non_cold_items);
	%find test set
	test_start = split_size + 1;
	test_end = 2*split_size;
	
	%make dev and test matrices
	dev = sparse(i(drawlist(1:split_size)), j(drawlist(1:split_size)), s(drawlist(1:split_size)), non_cold_users, non_cold_items);
	test = sparse(i(drawlist(test_start:test_end)), j(drawlist(test_start:test_end)), s(drawlist(test_start:test_end)), non_cold_users, non_cold_items);
	
	%make train matrix
	train_start = test_end + 1;
	train = sparse(i(drawlist(train_start:length(i))), j(drawlist(train_start:length(i))), s(drawlist(train_start:length(i))), non_cold_users, non_cold_items);
	
	%test split
	
	%num_user_dev_cold =  nnz(user_dev_cold(:, 1:size(train,2)));%minus overlap
	%num_user_test_cold = nnz(user_test_cold(:, 1:size(train,2)));
	%num_item_dev_cold = nnz(item_dev_cold(1:size(train,1), :));
	%num_item_test_cold = nnz(item_test_cold(1:size(train,1), :));
	%num_overlap = nnz(fullratings((size(train, 1) + 1):m, (size(train, 2) + 1):n));
	
	%total_ratings = nnz(train) + nnz(test) + nnz(dev) + num_user_dev_cold + num_user_test_cold + num_item_dev_cold + num_item_test_cold + num_overlap;
        total_ratings = nnz(train) + nnz(test) + nnz(dev) + nnz(item_test_cold) + nnz(user_test_cold) + nnz(item_dev_cold) + nnz(user_dev_cold) + nnz(both_test_cold) + nnz(both_dev_cold);
	%fprintf('train: %d\n test: dev: %d\n %d\n num_user_dev_cold: %d\n num_user_test_cold: %d/n num_item_dev_cold: %d\n num_item_test_cold: %d\n' full(nnz(train)), full(nnz(test)), num_user_dev_cold, num_user_test_cold, num_item_dev_cold, num_item_test_cold);
	%fprintf('total_ratings: %d\n', total_ratings);
	%fprintf('nnz(fullratings: %d\n', full(nnz(fullratings)));
	if total_ratings == nnz(fullratings)
		result = 1;
	end
	if dev.*test.*train ~= 0
		result = 0;
	end
	
	clear train_start test_end test_start drawlist split_size num_non_cold start_dev_cold; 
	clear start_item_dev_cold start_item_test_cold start_user_dev_cold start_user_test_cold end_item_dev_cold end_user_dev_cold;
	clear num_cold_dev i j s m n non_cold end_dev_cold num_cold_test num_ratings total_ratings;
	clear num_user_dev_cold num_user_test_cold num_item_dev_cold num_item_test_cold;        
	clear non_cold_users non_cold_items;
	
	%save variables
	save(filename);
end