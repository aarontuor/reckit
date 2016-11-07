%returns the squared Frobenius norm of a matrix
function x = squaredfrob(A)
    x = sum(sum(A.*A));
end