function printstats(epoch, trainRMSE, devRMSE, objective)        
    fprintf('epoch: %d\n', epoch);
    fprintf('\t train rmse: %.5f\n', full(trainRMSE));
    fprintf('\t dev rmse: %.5f\n', full(devRMSE));
    fprintf('\t obj: %.5f\n', full(objective));
end