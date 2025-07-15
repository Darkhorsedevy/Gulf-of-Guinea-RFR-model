
% ------------------------------------------------------------------------
% PATHS Project: Random Forest Regression Models for Key Performance Parameters
% ------------------------------------------------------------------------
% Author: Oluwaseyi Amaize
% GitHub: Darkhorsedevy
% Date: July, 2025
% Description:
% Trains Random Forest Regression models using Bagged Trees for multiple
% key metrics in the PATHS development project. Outputs include model
% performance metrics and predictor importance.
% ------------------------------------------------------------------------

clc; clear; close all;

%% ---------------------------
% 1. Load Dataset
% ----------------------------
data = readtable('PATHS_data.csv');  % Ensure this file is present in your workspace

%% ---------------------------
% 2. Define Predictors and Target Variables
% ----------------------------
predictorNames = {'InfrastructureInvestment', 'PolicyScore', 'GDPGrowth', ...
                  'GlobalOilDemandIndex', 'YouthPopulation', 'TrainingPrograms'};

responseVars = {...
    'GasFlaring', ...             % bcm/year
    'PipelineVandalism', ...      % cases/year
    'OilSpillage', ...            % cases/year
    'EnergyRevenue', ...          % USD billion
    'ElectricityGeneration', ...  % GW
    'GDPGrowthRate', ...          % %
    'YouthEmployment', ...        % thousands
    'PipelineConstruction'};      % km

X = data{:, predictorNames};  % Convert predictor table to matrix

% Set random seed for reproducibility
rng(1);

% Create output folder for models and plots
outputFolder = 'PATHS_RF_Models';
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

%% ---------------------------
% 3. Loop Through Response Variables
% ----------------------------
for i = 1:length(responseVars)
    responseName = responseVars{i};
    fprintf('\n==============================================\n');
    fprintf('Training Model for: %s\n', responseName);
    fprintf('==============================================\n');

    % Extract response vector
    Y = data.(responseName);

    % Split into training and testing sets (80/20)
    cv = cvpartition(length(Y), 'HoldOut', 0.2);
    XTrain = X(training(cv), :);
    YTrain = Y(training(cv), :);
    XTest  = X(test(cv), :);
    YTest  = Y(test(cv), :);

    % Define tree template
    treeTemplate = templateTree('MaxNumSplits', 20);

    % Train Random Forest (Bagging) model
    Mdl = fitrensemble(XTrain, YTrain, ...
                       'Method', 'Bag', ...
                       'NumLearningCycles', 100, ...
                       'Learners', treeTemplate, ...
                       'OOBPrediction', 'On');

    % -----------------------
    % 4. Evaluate Model
    % -----------------------
    % Out-of-Bag Error
    oobErr = oobLoss(Mdl);
    fprintf('OOB Error: %.4f\n', oobErr);

    % Predict on test set
    YPred = predict(Mdl, XTest);
    rmse = sqrt(mean((YTest - YPred).^2));
    fprintf('Test RMSE: %.4f\n', rmse);

    % 5-Fold Cross-Validation Loss
    cvMdl = crossval(Mdl, 'KFold', 5);
    cvLoss = kfoldLoss(cvMdl);
    fprintf('5-Fold CV Loss: %.4f\n', cvLoss);

    % -----------------------
    % 5. Feature Importance Plot
    % -----------------------
    importance = predictorImportance(Mdl);
    fig = figure('Visible', 'off');
    bar(importance, 'FaceColor', [0.2 0.4 0.6]);
    title(['Feature Importance - ', strrep(responseName, '_', ' ')], 'FontSize', 12);
    ylabel('Importance Score');
    xlabel('Predictors');
    set(gca, 'XTickLabel', predictorNames, 'XTickLabelRotation', 45);
    grid on;
    saveas(fig, fullfile(outputFolder, [responseName '_ImportancePlot.png']));

    % -----------------------
    % 6. Save Compact Model
    % -----------------------
    saveCompactModel(Mdl, fullfile(outputFolder, [responseName '_RFModel']));

    % Optionally save predictions and metrics
    T = table(YTest, YPred, 'VariableNames', {'Actual', 'Predicted'});
    writetable(T, fullfile(outputFolder, [responseName '_Predictions.csv']));
end

fprintf('\n✅ All models trained and saved in "%s" folder.\n', outputFolder);
