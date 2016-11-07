%=========================================================================
%
% Makes and returns matlab sparse matrix from .sparse files.
% If two arguments are present saves created matrix in .mat format.   
%
% parameters:
%   
%       filename: name of .sparse file
%
%       matfilename: name of .mat file to save created matrix to
%
%==========================================================================
function X = matmake(varargin)
	filename = varargin{1};
	matfilename = varargin{2};
	
	%load .sparse file
	X = load(filename);%this gives a matrix with rows [row column value] of target matrix
	X = spconvert(X);%makes canonical matrix rep. in sparse format
	m = size(X, 1);
	n = size(X, 2);
	
	%this step is to avoid data corruption from multiple entries for 
	%row column pairs
	X = X*0;
	X2 = load(filename);
	for i = 1:size(X2, 1)
            X(X2(i,1), X2(i,2)) = X2(i,3);
	end
        clear X2 i m n varargin filename;
        if nargin == 2
            save(matfilename);
        end
        
end
