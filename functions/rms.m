%returns the square root of the mean of the squared values in a matrix
function x = rms(A, nnzA)
        x = sqrt((sum(sum(A.*A))/nnzA));
end