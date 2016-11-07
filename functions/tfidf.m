function X_tfidf = tfidf(X)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Calculates the idf on the train set and performs tf-idf normalization of
% both matrices. Also does L2 normalization.
%
% tf-idf = tf * log(|D| / n_occurences)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

idf = log(size(X,1) ./ (sum(X>0) + eps));
IDF = spdiags(idf, 0, size(idf,2), size(idf,2));
X_tfidf = X * IDF;
X_tfidf = L2_norm_row(X_tfidf);

function Xnorm = L2_norm_row(X)
  Xnorm = spdiags(1 ./ (sqrt(sum(X.*X,2)) + eps), 0, size(X,1), size(X,1)) * X;
end

end
