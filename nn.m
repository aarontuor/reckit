function result = nnet(trainX trainT trainP k epochs)
	m = size(trainX, 1);
	n = size(trainX, 2);
	num_features = m;
	num_datapoints = n;
	upperbound = 1/sqrt(num_features + 1);
	Wghts1 = 2*upperbound*rand(num_features + 1, k) - upperbound;
	Wghts2 = 2*upperbound*rand(num_features + 1, k) - upperbound;
	num_epochs = 0;
	while num_epochs < epochs
		


	end
