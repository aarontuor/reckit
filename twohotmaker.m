function [twohotmatrix, val] = twohotmaker(M)
    [row, col, val] = find(M);
    num_users = size(M, 1);
    len = length(row); 
    adder = num_users*ones(len, 1);
    col = col + adder;
    %twohotmatrix = zeros(size(M, 1) + size(M, 2), len);
    m = size(M, 1) + size(M, 2);
    nonzeros = round(m*len/100);
    twohotmatrix = spalloc(m, len, 100);
    ratings_vector = zeros(1, len);
    for entry = 1:len
        twohotmatrix(row(entry), entry) = 1;
        twohotmatrix(len + col(entry), entry) = 1;
    end

end