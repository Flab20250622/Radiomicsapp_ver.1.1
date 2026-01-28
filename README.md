# Radiomicsapp_ver.1.1
レディオミクス解析用アプリケーションVersion1.1。全ファイルをダウンロードして用いる。実行ファイルにより使用性の向上を行った他軽微な修正を加えた。

--- How to use ---

--Get the Path--
The sample creates the following folder tree.
* [---] indicates a folder

[Samples]
  |
  |----[images] (*1)
  |      |
  |      |----[sample_1] (.dcm file) (*3)
  |      |----[sample_2] (.dcm file)
  |      |
  |
  |----[labels] (*2)
         |
         |----[mask(sample_1)] (.nrrd file)
         |----[mask(sample_2)] (.nrrd file)
         |

*1: Image files saved in DICOM format
*2: Mask image files saved in Nrrd format

Select the folder corresponding to [Samples]

--Type of Image File--
Set the dimension to be analyzed.
*3: If '2D' is selected, only the first image is used for analysis even if multiple images exist.

--Filter--
Set the pre-processing filter.
Original: no pre-processing
Wavelet: Wavelet transform
LoG: Lapracian of Gausian transform

--Type of Features--
Set the type of features to be calculated.

--Analysis Method--
>>>Lasso (Least Absolute Shrinkage and Selection Operator) Regression
-Set the number-
Set the number of features to be selected.
If not set, the regularisation parameters are set by GridSearch.
-Lasso parameter-
Set the percentage of cases to be assigned to the test data (0 to 1).
If 0 or 1 is selected, all cases are used as training data for feature calculation and 30% of the cases are used as test data in subsequent model building.

>>>Principal Component Analysis(PCA)
Reduce the features to two dimensions. The coordinates and contribution ratio of each point after reduction and a csv file are output.
>>>Regression Analysis-
Correlation coefficients between each selected feature and the residual distribution between measured and predicted values are output.
Reduces the number of features according to the correlation coefficients between the features, taking multicolinearity into account.
>>>Linear Discrimination Analysis-Reciever Operating Characteristic(LDA-ROC)-
Reduction to one dimension and evaluates the model accuracy by ROC.

--Go(only 1 click!)--
Start the analysis.
A 'Finish analysis' pop-up will appear once the analysis is complete.
All analysis data is provided as a csv file.

**License**
This package is covered by the open source 3-clause BSD License.

**Disclaimer**
The author assumes no responsibility or obligation to correct any defects in this program, nor for any damages that may result from its use. 

This product includes software developed by the PyRadiomics team (https://github.com/AIM-Harvard/pyradiomics) under the BSD 3-Clause License.

@Author; Koshi Hasegawa, 2025
All Rights Reserved.
