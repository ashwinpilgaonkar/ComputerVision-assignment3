%% Load Image Dataset
imds=imageDatastore('./q7_image/office_images','IncludeSubfolders',true,'LabelSource','foldernames');

% inspect the number of images per category, as well as category labels
tbl=countEachLabel(imds)

% visualization
figure
montage(imds.Files(1:16:end))

%% Prepare Training and Validation Image Sets
[trainingSet, validationSet] = splitEachLabel(imds, 0.6, 'randomize');

%% Create a Visual Vocabulary and Train an Image Category Classifier

% Creating Bag-Of-Features.
bag = bagOfFeatures(trainingSet);

% Encoding images using Bag-Of-Features.
img = readimage(imds, 1);
featureVector = encode(bag, img);

% Plot the histogram of visual word occurrences
figure
bar(featureVector)
title('Visual word occurrences')
xlabel('Visual word index')
ylabel('Frequency of occurrence')

%% Training an image category classifier for 5 categories.
categoryClassifier = trainImageCategoryClassifier(trainingSet, bag);

%% Evaluate Classifier Performance

% on training set
confMatrix = evaluate(categoryClassifier, trainingSet);

% on validation set
confMatrix_val = evaluate(categoryClassifier, validationSet);

% Compute average accuracy
avg_acc = mean(diag(confMatrix_val));

%% instance study
% read an image
img = imread(fullfile('./q7_image/office_images','backpack','frame_0001.jpg'));
figure
imshow(img)

% run classification
[labelIdx, scores] = predict(categoryClassifier, img);
labelName = categoryClassifier.Labels(labelIdx);
disp(labelName)
