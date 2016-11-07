function result = coldmatmaker(folder)
	%return is zero if something goes wrong and one if the split is good
	result = 0;
	
	%load full set of ratings
	load([folder '/fullratings.sparse']);
	fullratings = spconvert(fullratings);
        m = size(fullratings, 1);
	n = size(fullratings, 2);
	fullratings = fullratings*0;
	full2 = load([folder '/fullratings.sparse']);
	for i = 1:size(full2, 1)
            fullratings(full2(i,1), full2(i,2)) = full2(i,3);
	end
        clear full2;
        
        %permute columns for random selection of items in dev_cold and test_cold
        item_order = randperm(size(fullratings, 2));
        user_order = randperm(size(fullratings, 1));
        fullratings = fullratings(:, item_order);
        fullratings = fullratings(user_order, :);
        %calculate split size
        num_ratings = nnz(fullratings);
        split_size = num_ratings*0.05;
        
        %find item test_cold set
        i = n + 1;
        num_cold_test = 0;
        while num_cold_test < split_size
            i = i - 1;
            num_cold_test = num_cold_test + nnz(fullratings(:,i));
        end
        item_start_test_cold = i;
        item_test_cold = fullratings(:, start_test_cold:n);
        
        %find item dev_cold set
        num_cold_dev = 0;
        end_dev_cold = i -1;
        while num_cold_dev < split_size
            i = i - 1;
            num_cold_dev = num_cold_dev + nnz(fullratings(i, :));
        end
        start_dev_cold = i;
        item_dev_cold = fullratings(:, start_dev_cold:end_dev_cold);
        
        %find user test_cold set
        i = m + 1;
        num_cold_test = 0;
        while num_cold_test < split_size
            i = i - 1;
            num_cold_test = num_cold_test + nnz(fullratings(:,i));
        end
        start_test_cold = i;
        user_test_cold = fullratings(start_test_cold:m, :);
        
        %find item dev_cold set
        num_cold_dev = 0;
        end_dev_cold = i -1;
        while num_cold_dev < split_size
            i = i - 1;
            num_cold_dev = num_cold_dev + nnz(fullratings(i, :));
        end
        start_dev_cold = i;
        dev_cold = fullratings(:, start_dev_cold:end_dev_cold);
        
        %find dev set
        non_cold = 1:(start_dev_cold -1);
        [i, j, s] = find(fullratings(:, non_cold));
        num_non_cold = length(i);
        split_size = round(split_size);
        drawlist = randperm(length(i));
        
        %find test set
        test_start = split_size + 1;
        test_end = 2*split_size;
        
        %make dev and test matrices
        dev = sparse(i(drawlist(1:split_size)), j(drawlist(1:split_size)), s(drawlist(1:split_size)), m, start_dev_cold -1);
        test = sparse(i(drawlist(test_start:test_end)), j(drawlist(test_start:test_end)), s(drawlist(test_start:test_end)), m, start_dev_cold -1);
        
        %make train matrix
        train_start = test_end + 1;
        train = sparse(i(drawlist(train_start:length(i))), j(drawlist(train_start:length(i))), s(drawlist(train_start:length(i))), m, start_dev_cold -1);
        
        %make term document matrices- the full term doc is partioned into train/dev/test for testing and tuning cold start        
        load([folder '/termDoc.sparse']);
        termDoc = spconvert(termDoc);
        termDoc = termDoc(:, item_order);
        termDoc_tfidf = tfidf(termDoc);
        termDoc_train = termDoc(:, non_cold);
        termDoc_dev = termDoc(:, start_dev_cold:end_dev_cold); 
        termDoc_test = termDoc(:, start_test_cold:n);   
        termDoc_train_tfidf = termDoc(:, non_cold);
        termDoc_dev_tfidf = termDoc_tfidf(:, start_dev_cold:end_dev_cold); 
        termDoc_test_tfidf = termDoc_tfidf(:, start_test_cold:n);   
		
        
        total_ratings = nnz(train) + nnz(test) + nnz(fullratings(:, start_test_cold:n)) + nnz(fullratings(:, start_dev_cold:(start_test_cold - 1))) + nnz(dev);
        if total_ratings == nnz(fullratings)
            result = 1;
        else
            result = 0;
        end
		
        %only want to save: fullratings, train, dev, test, dev_cold, test_cold, fulltermDoc, termDoc_dev, termDoc_train, termDoc_test  
        clear train_start test_end test_start drawlist split_size num_non_cold start_dev_cold num_cold_dev i j s m n non_cold end_dev_cold num_cold_test num_ratings itemorder total_ratings;        
        save([folder '/' folder '.mat']);
end
