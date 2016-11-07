function results = expt(exName, R, Dev)
    alphavector = [0.001, 0.003, 0.01, 0.1, 0.3, 1];
    results = [];
    for k = linspace(10, 100, 10) 
        for alpha = alphavector
            for lambda = linspace(0, 1, 10)
                for bias = [true, false]
                    center = bias;
                    [best_dev, averageEpoch, trainRMSE, devRMSE, objHistory, U, P] = mafac(R, Dev, k, 100, 10, alpha, lambda, false, bias, center);
                    plot(trainRMSE);
                    hold on;
                    plot(devRMSE, 'r');
                    xlabel('epoch');
                    ylabel('rmse');
                    title({['k = ' num2str(k) ' alpha = ' num2str(alpha) ' lambda = ' num2str(lambda) ' bias = ' num2str(bias)]; ['Average secs per epoch: ' num2str(averageEpoch)]}); 
                    if bias 
                        biasword = 'bias';
                    else
                        biasword = '';
                    end
                    print(['plots/' exName '_' num2str(k) '_' num2str(alpha) '_' num2str(lambda) '_' biasword '.png'], '-dpng');
                    hold off;
                    new_results = [best_dev averageEpoch k alpha lambda bias];
                    results = [results; new_results];
                    save('amazonWatches/watches.mat', 'results', '-append');
                end
            end
        end
    end
end