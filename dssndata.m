function dssndata(M, Q, matfilename)
    [row, col, val] = find(M);
	num_words = size(Q, 1);    
	num_users = size(M, 1);
    num_ratings = length(row); 
    twohotmatrix = sparse(num_ratings, num_users);
	tfidfmatrix = sparse(num_ratings, num_words); 
    val = val';
    for entry = 1:num_ratings
        onehotmatrix(entry, row(entry)) = 1;
		tfidfmatrix(entry, :) = Q(:, col(entry))';
    end
	save(matfilename);
end
