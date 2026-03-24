# Remote Sensing Applications in Almond Orchards: A Comprehensive Systematic Review of Current Insights, Research Gaps, and Future Prospects

Authors:
- Guimaraes
- Sousa
- Padua
- Bento
- Couto

Link: https://www.mdpi.com/2076-3417/14/5/1749

## Abstract

Almond cultivation is of great socio-economic importance worldwide.
With the demand for almonds steadily increasing due to their nutritional value and versatility, optimizing the management of almond orchards becomes crucial to promote sustainable agriculture and ensure food security.
The present systematic literature review, conducted according to the PRISMA protocol, is devoted to the applications of remote sensing technologies in almond orchards, a relatively new field of research.
The study includes 82 articles published between 2010 and 2023 and provides insights into the predominant remote sensing applications, geographical distribution, and platforms and sensors used.
The analysis shows that water management has a pivotal focus regarding the remote sensing application of almond crops, with 34 studies dedicated to this subject.
This is followed by image classification, which was covered in 14 studies.
Other applications studied include tree segmentation and parameter extraction, health monitoring and disease detection, and other types of applications.
Geographically, the United States of America (USA), Australia and Spain, the top 3 world almond producers, are also the countries with the most contributions, spanning all the applications covered in the review.
Other studies come from Portugal, Iran, Ecuador, Israel, Turkey, Romania, Greece, and Egypt.
The USA and Spain lead water management studies, accounting for 23% and 13% of the total, respectively.
As far as remote sensing platforms are concerned, satellites are the most widespread, accounting for 46% of the studies analyzed.
Unmanned aerial vehicles follow as the second most used platform with 32% of studies, while manned aerial vehicle platforms are the least common with 22%.
This up-to-date snapshot of remote sensing applications in almond orchards provides valuable insights for researchers and practitioners, identifying knowledge gaps that may guide future studies and contribute to the sustainability and optimization of almond crop management.

## 3.3 Spectral Indices

Considering the most frequently used VIs (Figure 8), NDVI [23] was used most frequently in 46% of the studies.
The Enhanced Vegetation Index (EVI) [24] was also frequently used in 16% of the studies, followed by the Soil Adjusted Vegetation Index (SAVI) [25] in 9%, the CWSI [26], and the Green Normalized Difference Vegetation Index (GNDVI) [27] in 8% each.
When analyzing the RS applications, NDVI and EVI were used in all types of applications, especially in WM.
On the other hand, CWSI was mainly used in WM as it is an efficient parameter to evaluate crop water use and water stress [28].
Green Leaf Index (GLI) [29], Plant Senescence Reflectance Index (PSRI) [30], and Red Edge Normalized Difference Vegetation Index (REN) [31] were only implemented in IC applications, while Canopy Chlorophyll Content Index (CCCI) [32] and Wide Dynamic Range Vegetation Index (WDRVI) [33] were specifically used for TSPE applications.

## 3.4.4 Water Management

Considering studies using the satellite platform, 12 studies associated with estimates of evapotranspiration (ET) were identified.
Gaur et al. [71] applied the Simplified-Surface Energy Balance Index Algorithm (S-SEBI) to estimate ET in almond orchards, demonstrating reliable results with a low average root mean square error (RMSE) of 0.12 mm/h.
He et al. [72] used the Mapping ET at high resolution with the Internal Calibration (METRIC) technique for accurate daily and monthly ET estimates in a Californian almond orchard.
Schauer and Senay [73] studied crop water dynamics in the California Central Valley using Landsat-derived annual actual ET with the SSEBop model, revealing a substantial rise in almond cultivation area and water consumption.
Xue et al. [74] compared three RS ET models (pySEBAL, SEBS, and METRIC) for daily actual ET in almond orchards, showing generally acceptable agreement with in situ measurements.
Sánchez et al. [75] used the simplified Two-Source Energy Balance (STSEB) model to assess crop ET and related coefficients, aiding in predicting water needs based on orchard age and biophysical parameters.
Bellvert et al. [76] estimated actual ET and crop coefficients for almonds, revealing varying water stress coefficients (Ks) through regressions between CWSI and stem water potential (SWP).
Another study by Bellvert et al. [77] developed a RS model for almond orchards, accurately estimating actual ET and water stress using multispectral and thermal imagery.
He et al. [78] employed high-resolution satellite data and the METRIC model for precise almond tree crop ET estimation.
Knipper et al. [79] investigated methods for separating transpiration (T) and evaporation (E) in almond orchards using the ALEXI modeling framework.
Mokhtari et al. [80] assessed Multi-Sensor Data Fusion-Evapotranspiration (MSDF-ET) for estimating ETa from Landsat 8 data, displaying reliable results compared to eddy covariance measurements.
Peddinti and Kisekka [81] used the TSEB model to study land use effects on ET in a California almond orchard, emphasizing the importance of high-resolution thermal imagery for precise estimates.
Wong et al. [82] analyzed agricultural water use in the California Central Valley using Landsat data, providing insights for sustainable groundwater management.

Regarding studies using the satellite platform, six studies related to irrigation monitoring were identified.
In Bretreger et al. [83], Landsat 8 data is employed to monitor paddock-scale irrigation.
Strong relationships (R2 between 0.72 and 0.85) between NDVI/EVI and ground-based crop water measurements show the effectiveness of RS for irrigation monitoring.
On the other hand, González-Gómez et al. [84] studied the impact of soil management (conventional and vegetation cover) and irrigation levels on almond orchards from 2018 to 2020.
They found that combining vegetation cover with optimal irrigation improves orchard performance, leading to increased biomass and yield.
Beverly et al. [85], in turn, sought to improve irrigated agricultural productivity in northern Victoria by using a bio-economic modeling framework.
Their study revealed that optimizing water efficiency, achieved through genetic improvement and precision water management, along with accessing 50% of available groundwater, had the greatest potential to maximize irrigated agricultural gross margins.
Bretreger et al. [86] compared tabulated crop coefficients to RS equivalents for monitoring irrigation water use.
Localized tabulated coefficients, particularly for Australia, outperformed crop-specific RS equivalents, which struggled to match North American relationships.
The study suggests that, overall, using localized tabulated crop coefficients is more effective in monitoring irrigation water use.
Bretreger et al. [87] used RS to quantify irrigation water use in remote areas, employing FAO56-based soil water deficit modeling.
Their results revealed close agreement between metered irrigation time series and modeling, with only minor variations.
Monte Carlo uncertainty analysis on RAW showed substantial improvements, ranging from 3% to 15% monthly and 56% to 68% annually, compared to studies neglecting soil water deficits.
Jofre-Čekalović et al. [88] developed a study on almond crop water use under diverse irrigation treatments and surface energy balance algorithms.
Data from a central California almond orchard was used, showing TSEB2 + S3 provided the most accurate evapotranspiration estimates.
The results show that deficit irrigation strategies could save up to 37% of water without significantly reducing crop yield.

In relation to other types of studies concerning different topics, four studies were conducted using a satellite platform.
Wen et al. [89] employed RS to analyze how water and salt stresses affect diverse crops in real agricultural conditions.
Using the Sentinel-2 satellite system, the study revealed varied crop responses to salt and drought stress, considering factors such as crop type, growing season, and stress timing.
Alam et al. [10] studied the water-energy-food nexus in the California Central Valley, providing insights into regional precipitation and actual ET.
Boken [90] enhanced crop models and evaluated agricultural drought effects, revealing correlations between soil moisture, precipitation, and almond crop yields.
Paul et al. [91] proposed a new methodology for agricultural water management, demonstrating reduced water use and 
increased crop yield compared to traditional approaches.