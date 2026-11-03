---
title: CoolKarachi — An Explainable Machine-Learning Digital Twin for 30 m Urban Heat Mapping and Cooling-Intervention Prioritisation in Karachi
authors: Gotam Dulhani (23I-0545), Mohat Kumar (23K-0894)
affiliation: Department of Data Science, FAST-NUCES, Karachi, Pakistan
---

**Abstract**

Karachi experiences pre-monsoon land surface temperatures above 55 °C, and the 2015 heatwave killed more than 1,200 people. Cooling budgets are limited, so interventions must be placed where they actually reduce heat. We build a six-year (2020–2025), 30 m land-surface-temperature (LST) record for Karachi entirely within Google Earth Engine, using Landsat 8/9 Collection 2 Level-2 thermal data, Sentinel-2 spectral indices, Dynamic World land-cover fractions, SRTM elevation and WorldPop population. From 360,000 built-up sample points we train four machine-learning models — Random Forest, XGBoost, LightGBM and a multilayer perceptron — to predict three definitions of LST: a local anomaly (LST minus the 2 km block-year mean), a within-year anomaly, and raw LST. Hyperparameters are tuned with Optuna, and evaluation uses spatially blocked cross-validation on 2 km tiles so that no test pixel is adjacent to a training pixel. Under the local-anomaly target all four models converge to RMSE ≈ 1.22 °C and R² ≈ 0.22, with near-zero mean bias. Tree ensembles and the neural network are statistically indistinguishable, which indicates that the neighbourhood-scale signal is simple and robust rather than model-limited. Using a separate all-land-cover sample we also find that the classical urban–rural surface heat island of Karachi is *negative* (−1.2 °C), because the rural reference is hotter inland desert; the modelling target is therefore block-relative rather than urban–rural. SHAP attribution and counterfactual simulations show that vegetation cover produces the largest and most reliable cooling, whereas surface albedo behaves counter-intuitively because bright dry surfaces in this climate are also hot. We argue that reporting this negative result is more useful than hiding it, and that explainable, spatially blocked, multi-target evaluation is a minimum standard for urban-heat digital twins intended to guide public spending.

# 1. Introduction

Karachi is one of the fastest-growing coastal megacities in the world, with a metropolitan population estimated above sixteen million. Its climate is hot desert (BWh) moderated by sea breeze, and the pre-monsoon months of May and June combine clear skies, high insolation and low wind with land surface temperatures that routinely exceed 50 °C. South Asia is warming at a rate above the global mean, and humid coastal cities face a compounding of heat and humidity that raises the effective risk beyond what dry-bulb temperature alone suggests [22]. The June 2015 heatwave, which caused more than 1,200 deaths within a few days, remains the clearest demonstration that heat in this city is not a comfort issue but a public-health emergency concentrated in low-income, low-vegetation districts.

The urban heat island (UHI) is usually described as the elevation of urban temperature relative to surrounding rural land. In Karachi the surface version of this contrast turns out to be *negative* — the surrounding desert is hotter than the city — for reasons set out in Section 5, and that result alone is a reason to frame the analysis around neighbourhood-relative rather than urban–rural heat. Two distinct quantities are also conflated in the standard definition: the canopy-layer air temperature, which is what people experience, and the surface temperature measured by thermal remote sensing. The two are related but not interchangeable; surface temperature responds within minutes to insolation and material properties, whereas air temperature is buffered by advection and storage. Satellite thermal data therefore cannot replace meteorological measurement. It can, however, resolve spatial patterns at a resolution that ground networks in Karachi do not approach, and those patterns are what determine where a tree, a shaded corridor or a reflective roof will do the most good.

The dominant operational constraint is resolution. The widely used MODIS products deliver 1 km thermal pixels, which is roughly the scale of a whole neighbourhood block in Karachi; a single pixel can average a park, a road and a dense settlement. Landsat 8/9 provides 100 m thermal data resampled to 30 m, and Sentinel-2 provides 10 m optical data, which together are fine enough to separate a street from the buildings beside it. Until recently the computational cost of processing six years of imagery across a metropolitan region made such an analysis impractical for a student project. Google Earth Engine removes that barrier: the entire archive is available server-side, and the analysis becomes a matter of writing the right reduction rather than downloading terabytes.

Machine learning enters for a second reason. The relationship between land cover and surface temperature is non-linear and strongly interacting: the cooling effect of vegetation depends on moisture availability, the warming effect of built-up surfaces depends on geometry and material, and the whole system is modulated by a coastal-to-inland elevation gradient. A linear regression fitted to spectral indices will therefore underfit, and — more importantly — will not support counterfactual reasoning. A tree ensemble or a neural network fitted to the same data can be interrogated: given a pixel with these properties, what would its temperature be if the vegetation fraction were raised by ten percentage points? That question is the basis of a *digital twin*, a surrogate model of the system that can be perturbed cheaply to test interventions before money is spent.

This paper reports the construction and evaluation of such a surrogate for Karachi. Our contributions are:

- A reproducible, entirely open, Earth Engine implementation that produces a 30 m pre-monsoon LST record for Karachi Division for 2020–2025, including a built-up stratified sample of 60,000 points per year with coordinates suitable for spatial validation.
- A methodological comparison of three prediction targets — local anomaly, within-year anomaly and raw LST — which separates neighbourhood-scale drivers from city-wide climate and geographic gradients. Reporting all three prevents a single R² from hiding the design choice.
- A spatially blocked evaluation protocol on roughly 700 tiles of 2 km, with Optuna hyperparameter tuning performed inside GroupKFold on training tiles only, so that reported accuracy is not inflated by spatial autocorrelation.
- SHAP-based attribution and counterfactual greening and cool-roof scenarios, with an explicit and honest treatment of a confounded albedo effect that contradicts the standard cool-roof narrative.

The remainder of the paper is organised as follows. Section 2 reviews related work. Section 3 describes the study area and data. Section 4 sets out the methodology. Section 5 reports exploratory analysis, Section 6 the modelling results, Section 7 discusses their meaning and Section 8 their limitations. Section 9 concludes.

# 2. Related Work

Thermal remote sensing of urban climate has a long methodological tradition. Oke established the energetic basis of the urban heat island and distinguished boundary-layer, canopy-layer and surface heat islands, a distinction that remains essential when interpreting satellite temperatures [1]. Voogt and Oke reviewed the specific problems of thermal remote sensing over heterogeneous urban surfaces, including emissivity variation, mixed pixels and the anisotropy of directional temperature [2]. Weng extended this into an operational review of methods, applications and trends, and documented the now-standard practice of relating LST to spectral indices such as NDVI and NDBI [3].

The drivers of surface urban heat islands have been studied extensively at continental and global scale. Imhoff et al. used MODIS to show that the magnitude of the surface UHI varies systematically across biomes and is largest in ecosystems with high available moisture and dense vegetation [4]. Zhao et al. demonstrated that background climate strongly conditions the intensity of the urban heat island, so that a mitigation strategy effective in one climate zone may be ineffective or even counterproductive in another [5]. Chakraborty and Lee developed a simplified urban-extent algorithm to characterise surface UHI globally and examined the controlling role of vegetation on its spatiotemporal variability [6]. These studies establish that vegetation is the dominant lever in most cities, but they operate at 1 km or coarser and therefore cannot resolve the intra-neighbourhood variation that intervention design requires.

On the mitigation side, Bowler et al. systematically reviewed the empirical evidence for urban greening as a cooling strategy and found consistent but modest cooling, with substantial variation by configuration and climate [7]. Akbari et al. argued that increasing urban albedo globally could offset a meaningful quantity of radiative forcing and reduce cooling demand [8]. The cool-roof literature is, however, largely drawn from temperate and arid North American cities where the surface-energy balance differs from a humid coastal subtropical setting, which motivates the present test.

Machine learning has become the default tool for LST downscaling and driver analysis. Breiman's random forests [9], Chen and Guestrin's XGBoost [10] and Ke et al.'s LightGBM [11] are the three workhorses in this literature, typically applied to Landsat or MODIS predictors with terrain and socioeconomic covariates. Hyperparameter selection is frequently ad hoc; Akiba et al.'s Optuna framework [12] provides a principled and reproducible alternative based on sequential model-based optimisation. For interpretation, Lundberg and Lee's SHAP framework [13] unifies several attribution methods under Shapley values and has become the standard way to move from black-box accuracy to driver ranking. Crucially, Roberts et al. showed that cross-validation for spatially or temporally structured data must respect that structure, otherwise error estimates are optimistic [14]; spatial blocking is now expected practice in geoscientific machine learning, though it is still frequently omitted in applied urban-heat papers.

Finally, the digital-twin framing has entered climate science. Bauer et al. argued for a digital twin of the Earth system to support the green transition, emphasising that a twin must be both fast enough to be interrogated and transparent enough to be trusted [15]. Most urban implementations of this idea are physically based and computationally heavy. A fitted, explainable machine-learning surrogate is a pragmatic middle ground for a city that cannot run a full urban canopy model, provided its limitations are stated clearly.

The gap this paper addresses is therefore specific: there is, to our knowledge, no open, explainable, 30 m, multi-year surface-heat surrogate for Karachi that (i) validates spatially, (ii) reports multiple target definitions, and (iii) tests cooling interventions counterfactually while acknowledging confounding. The present work is a first, deliberately modest step toward filling it.

# 3. Study Area and Data

## 3.1 Study area

The analysis covers Karachi Division, bounded approximately by 66.65–67.45 °E and 24.70–25.15 °N, extended by a 15 km buffer so that rural reference pixels are available for the surface-UHI contrast. The terrain rises from mangrove flats and reclaimed land at or below sea level along the coast to roughly 300 m in the north-eastern inland fringe. This coastal-to-inland gradient is small in absolute terms but, as Section 6 shows, it is a first-order control on surface temperature and therefore a confounder that any driver analysis must confront.

The season is restricted to May and June. This is the pre-monsoon period, characterised by the clearest skies of the year, minimal cloud contamination in the Landsat archive, peak insolation, and the highest surface temperatures. Restricting the season removes the large seasonal cycle from the target and makes inter-annual comparison meaningful; it also means the results describe heat stress in the period when it is most dangerous, not an annual average.

## 3.2 Earth Engine feature stack

All data are accessed through Google Earth Engine [16]. For each year a single multi-band image is assembled and sampled. Table 1 lists the layers.

**Table 1.** Input layers in the Earth Engine feature stack.

| Layer | Product | Native resolution | Role |
|---|---|---|---|
| LST | Landsat 8/9 Collection 2 Level-2, band ST_B10 | 30 m (100 m thermal) | prediction target |
| NDVI | Sentinel-2 surface reflectance | 10 m | vegetation vigour |
| NDBI | Sentinel-2 surface reflectance | 10 m | built-up intensity |
| NDWI | Sentinel-2 surface reflectance | 10 m | surface moisture |
| Albedo | Sentinel-2 surface reflectance | 10 m | reflectivity |
| Built fraction | Dynamic World | 10 m | impervious cover |
| Tree fraction | Dynamic World | 10 m | canopy cover |
| Water fraction | Dynamic World | 10 m | open water |
| Elevation | SRTM 1 arc-second | 30 m | terrain |
| Population | WorldPop | 100 m | exposure |
| Coordinates | pixel centroid | — | spatial validation |

LST is derived from the Landsat Collection 2 Level-2 surface temperature product, converted from digital numbers to degrees Celsius using the published scale and offset following the formulation of Ermida et al. [21], and filtered on the QA_PIXEL band to remove cloud, cloud shadow, snow and fill. Sentinel-2 [20] indices are computed from the harmonised surface-reflectance collection after cloud masking with the scene classification layer, and a median composite is taken over May–June to suppress residual cloud and haze. NDVI, NDBI and NDWI are normalised differences of the standard band pairs, and albedo is a Liang-style broadband estimate from five Sentinel-2 bands (blue, red, near-infrared and the two short-wave infrared bands). Dynamic World [17] provides per-pixel class probabilities at 10 m, from which we take the built, tree and water fractions. Elevation comes from SRTM [18] and population from WorldPop [19]; because WorldPop provides a single 2020 surface in the collection we use, the 2020 population layer is reused for all years and the population feature is therefore static in time. This is a documented limitation rather than an oversight, and it is discussed in Section 8.

Sentinel-2 and Dynamic World are supplied at 10 m and Landsat thermal at 30 m; sampling is performed at a 30 m scale so that each training row corresponds to one thermal pixel and its co-located predictors, using Earth Engine's default nearest-neighbour resampling.

## 3.3 Sampling design

Two distinct samples are exported, because two distinct questions are being asked.

The **full sample** requests 60,000 points per year uniformly at random across the buffered region, so all land-cover classes appear in proportion to their area. After masked pixels are dropped, 231,112 points remain across the six years (about 38,500 per year). This sample is the basis for the urban–rural surface heat island contrast reported in Section 5.

The **built-up sample** draws 60,000 points per year from pixels where the Dynamic World built fraction is at least 0.2. The sample is stratified on that binary mask, so the requested 60,000 points come from built-up land regardless of how much of the buffered region is rural. This is the sample used for all machine-learning experiments. The reason is that a naive uniform sample is dominated by bare and rural land, and a model fitted to it learns mostly the urban–rural contrast rather than the intra-urban variation that intervention design needs. Restricting to built-up pixels forces the model to explain why one part of the city is hotter than another.

Both samples are exported as CSV tables to Google Drive and downloaded locally. The built-up sample comprises 360,000 rows across six years with no missing values after quality control, and includes longitude and latitude so that spatial blocking and mapping are possible.

## 3.4 Quality control and cleaning

Cloud and shadow masking removes the majority of unusable observations before compositing. After download, the following cleaning steps are applied identically for all targets. They are implemented in the model script when it loads the exported tables, so the descriptive statistics in Table 2 are reported on the pre-clip values.

- Rows missing LST, longitude or latitude are dropped, and exact duplicates removed.
- Albedo is clipped to the physically plausible range 0.02–0.65. The raw maximum in the sample is 1.94, which is impossible for a natural surface and indicates residual cloud or a compositing artefact.
- Population is log-transformed as log(1 + POP) to compress a strongly right-skewed distribution.
- Every predictor is winsorised at its 0.1 and 99.9 percentiles to limit the influence of residual outliers.

No imputation is required, because the median compositing and masking steps leave a complete feature vector at every retained pixel.

# 4. Methodology

## 4.1 Prediction targets

The choice of target is the single most consequential modelling decision in this study, and it is usually left implicit. We therefore define three targets and fit the full pipeline to each.

The **local anomaly** is the LST of a pixel minus the mean LST of all sampled pixels in the same 2 km block in the same year:

```
LST_local = LST - mean(LST | block, year)
```

This target removes both the inter-annual weather signal and the broad spatial gradient, leaving only the deviation of a pixel from its immediate neighbourhood. It answers the question that matters for intervention design: within this part of the city, which pixels are hotter than their surroundings, and why? It is also the hardest target, because the residual variance is small by construction.

The **within-year anomaly** subtracts only the year mean:

```
LST_anom = LST - mean(LST | year)
```

This retains the city-wide spatial pattern, including the coastal–inland gradient, while removing the year-to-year differences in weather that would otherwise dominate.

The **raw LST** is the absolute surface temperature in degrees Celsius. For this target only, the calendar year is added as a predictor, because a model asked to reproduce absolute temperature across six years without knowing the year would be forced to average across very different meteorological conditions.

Reporting all three separates what a model can learn about a neighbourhood from what is simply geography and weather. A paper that reports only raw LST will appear more accurate and will be less useful.

## 4.2 Features

Nine predictors are used for the two anomaly targets: NDVI, NDBI, NDWI, albedo, built fraction, tree fraction, water fraction, elevation and log population. The raw-LST target adds the year, for the reason given above. Land-cover class is deliberately excluded as a categorical predictor: it is a discretisation of information already carried by the continuous fractions, and including it would make counterfactual perturbation incoherent, since a pixel cannot be smoothly moved from one class to another.

## 4.3 Models

Four model families are compared, chosen to span the bias–variance spectrum and to include both tree-based and non-tree approaches.

**Random Forest** [9] averages many decorrelated decision trees grown on bootstrap samples. It is robust to irrelevant features and requires little preprocessing, but it is memory-heavy when trees are deep.

**XGBoost** [10] is a regularised gradient-boosting implementation with second-order loss approximations. It typically achieves the lowest error on tabular data and provides native handling of missing values.

**LightGBM** [11] uses histogram-based splits and leaf-wise growth, which makes it substantially faster than XGBoost at comparable accuracy on large datasets.

**Multilayer perceptron** is a feed-forward network with standardised inputs, early stopping and L2 regularisation, whose hidden-layer configuration is chosen by tuning from a small set of one-, two- and three-layer options. It is included as a non-tree reference: if a neural network with the same inputs reaches the same error as the ensembles, the relationship is likely smooth and low-order rather than a collection of threshold effects.

## 4.4 Hyperparameter optimisation

Tuning uses Optuna [12] with the Tree-structured Parzen Estimator sampler and a fixed seed for reproducibility. The search space is defined per model: number of trees or leaves, depth, learning rate on a log scale, subsampling rates, minimum child weight or leaf size, and for the network the hidden-layer configuration, L2 penalty and initial learning rate.

Each trial is scored by three-fold GroupKFold cross-validation on the training tiles, where the group is the spatial block. This is important: tuning on a random split would select hyperparameters that overfit to spatial autocorrelation. To keep the search affordable, trials are evaluated on a random subsample of 60,000 training rows, and the winning configuration is then refitted on the full 303,387 training rows. Fifteen trials are run per model for the local target and twelve for the other two.

## 4.5 Spatial cross-validation

The city is partitioned into blocks of 0.02 degrees, approximately 2 km, giving 701 blocks in the built-up sample. Blocks, not rows, are the unit of splitting: 20 per cent of blocks are held out as a test set and the remaining 80 per cent form the training set. The resulting split is 303,387 training rows and 56,613 test rows.

This protocol follows the argument of Roberts et al. [14] that structured data require structured resampling. With a random split, a pixel's immediate neighbours would almost certainly appear in the training set, and a flexible model could interpolate them, producing an R² that reflects memorisation of location rather than learning of process. The cost of blocking is that the reported error is pessimistic relative to a random split; the benefit is that it is honest about performance in an unseen part of the city, which is the only performance that matters for extrapolation to new neighbourhoods.

## 4.6 Explainability

Attribution uses SHAP [13] with a tree explainer. Because explaining a deep 600-tree forest is prohibitively slow, attribution is computed on the LightGBM model rather than on the Random Forest; the two differ by 0.004 °C RMSE on the local target, so the attribution is equivalent for practical purposes, and this substitution is stated wherever SHAP results are reported. For each of 1,200 randomly sampled test points, SHAP decomposes the prediction into additive contributions from the nine features, in degrees Celsius. Global importance is the mean absolute contribution across the sample. Because the local-anomaly target is itself in degrees Celsius, the SHAP values are directly interpretable as "this feature makes this pixel this many degrees hotter or cooler than its neighbourhood mean", which is a more useful unit than a normalised importance score.

## 4.7 Counterfactual scenarios

The fitted model serves as the digital twin. Three interventions are simulated on the held-out test set by perturbing features and re-predicting, holding everything else fixed:

- **Greening**: NDVI increased by 0.10 and tree fraction by 0.05, corresponding to substantial additional canopy in a city where the median tree fraction is only 0.033.
- **Cool roof**: albedo increased by 0.15, corresponding to a white or highly reflective coating on a typical roof.
- **Combined**: both perturbations applied together.

Perturbed values are clipped to physically plausible ranges so that the model is never queried outside the region of feature space it was trained on. The reported statistic is the change in predicted LST relative to the unperturbed prediction, summarised as the mean, the median, and the percentage of pixels cooling by more than 0.5 °C and by more than 1.0 °C. Because the model is a surrogate fitted to observational data, these are associational counterfactuals: they describe what pixels with the perturbed properties tend to look like, not the result of a controlled experiment. Section 8 returns to this point.

# 5. Exploratory Analysis

Table 2 summarises the built-up sample.

**Table 2.** Descriptive statistics of the 360,000-point built-up sample, 2020–2025. Albedo is shown before the 0.02–0.65 clip applied in modelling; its mean and standard deviation are unaffected by the single extreme value.

| Variable | Mean | Std | Min | Median | Max |
|---|---|---|---|---|---|
| LST (°C) | 44.24 | 2.55 | 31.24 | 44.11 | 57.91 |
| NDVI | 0.182 | 0.120 | −0.124 | 0.157 | 0.822 |
| NDBI | 0.049 | 0.060 | −0.415 | 0.055 | 0.553 |
| NDWI | −0.244 | 0.110 | −0.735 | −0.231 | 0.264 |
| Albedo | 0.248 | 0.041 | 0.044 | 0.249 | 1.942 |
| Built fraction | 0.359 | 0.128 | 0.200 | 0.328 | 0.774 |
| Tree fraction | 0.040 | 0.022 | 0.018 | 0.033 | 0.438 |
| Water fraction | 0.043 | 0.018 | 0.017 | 0.040 | 0.455 |
| Elevation (m) | 34.5 | 25.1 | −20 | 32 | 296 |
| Population per cell | 122.6 | 152.9 | 0.08 | 36.0 | 686.1 |

The mean surface temperature of built-up pixels is 44.2 °C, with a standard deviation of 2.5 °C and a maximum of 57.9 °C. The tree fraction is strikingly low: the median built-up pixel has only 3.3 per cent tree cover, and the 75th percentile is 4.1 per cent. Karachi's built environment is therefore not merely dense but almost entirely unvegetated, which sets a low baseline for greening interventions and explains why even a small absolute increase in canopy produces a detectable modelled effect.

Inter-annual variation is large. Annual mean LST is 44.9 °C in 2020, falls to 42.9 °C in 2022 and peaks at 46.2 °C in 2024, a range of 3.3 °C. This is meteorological, not structural: the city's land cover does not change by enough in four years to move the mean by three degrees. It is precisely this variation that the anomaly targets are designed to remove, and it is the reason a raw-LST model without a year term would be badly specified.

Land-cover ordering follows physical expectation. Within the built-up sample, mean LST is 39.6 °C over water, 41.0 °C over trees, 43.1 °C over cropland, 44.2 °C over built land and 44.5 °C over bare ground. The small gap between built and bare land, and the fact that trees are more than three degrees cooler than built surfaces, is the empirical basis for the greening scenario. The gap is much larger in the full all-land-cover sample (35.9 °C over trees against 44.6 °C over built land), because that sample includes rural orchards and mangrove; the 3.2 °C figure is the urban-tree contrast, which is the relevant one for street-level greening.

The full sample also supports the classical urban–rural contrast, and here the result is a surprise. Defining urban as a built fraction of at least 0.2 and rural as below 0.1, the pooled six-year mean LST is 44.7 °C urban against 45.9 °C rural, a surface urban heat island of **−1.2 °C**. The sign is negative in every individual year, with annual values between −1.1 and −1.9 °C. This is not a data fault. Seventy-eight per cent of the rural sample is bare ground, largely inland desert at a mean elevation of 98 m, while the urban sample sits at 35 m near the coast; in an arid coastal city the surroundings really are hotter than the city. The classical urban-minus-rural index therefore measures geography rather than urban heat in Karachi, which is the empirical justification for the block-relative local anomaly used in all modelling below.

![Annual urban–rural surface heat island of Karachi from the full all-land-cover sample. The contrast is negative in every year; the pooled value is −1.2 °C.](figures/suhi_annual.png){w=0.62}

Correlations with LST are informative and, in two cases, counter-intuitive. NDBI is the strongest positive correlate at +0.45, followed by elevation at +0.39. Population correlates negatively at −0.33 in its raw form (−0.28 after the log transform used in the models), and tree fraction at −0.21. NDVI correlates only weakly at −0.12, which is expected in a sample restricted to built-up pixels where vegetation is sparse and the index saturates. Built fraction correlates at only −0.07, confirming that within the built-up stratum the *amount* of building matters less than its spectral and thermal character.

Two of these correlations require care. The positive elevation correlation does not mean that hills are intrinsically hot; it reflects the fact that low-lying land in Karachi is coastal, humid and ventilated by sea breeze, whereas the inland fringe is dry, dusty and less ventilated. Elevation is therefore acting as a proxy for a bundle of coastal-inland climate variables, and its importance in the models should be read that way. The negative population correlation is the same confound in a different guise: the densest districts are the old central and coastal ones, which are cooler than the sparsely populated inland fringe. A naive reading would conclude that population reduces temperature, which is plainly wrong; the correct reading is that population and elevation are collinear in this city, and the model is attributing coastal proximity to whichever of the two it finds convenient.

Finally, albedo correlates positively with LST at +0.08, which is the opposite of the cool-roof hypothesis. The explanation is again physical: in a dusty desert-coastal environment the brightest surfaces are dry bare ground and degraded land, which are also among the hottest. Albedo and moisture are negatively correlated here, and the moisture effect dominates. This is examined further in Section 7.

# 6. Results

## 6.1 Model comparison under the local-anomaly target

**Table 3.** Out-of-sample performance on 56,613 held-out pixels (local-anomaly target, sorted by RMSE).

<!-- results:models:local -->
| Model | RMSE (°C) | MAE (°C) | R² | MBE (°C) | n |
|---|---|---|---|---|---|
| RF | 1.219 | 0.928 | 0.219 | -0.002 | 56,613 |
| XGB | 1.221 | 0.931 | 0.216 | +0.002 | 56,613 |
| MLP | 1.222 | 0.932 | 0.215 | -0.008 | 56,613 |
| LGBM | 1.223 | 0.933 | 0.214 | +0.002 | 56,613 |
<!-- /results:models:local -->

Table 3 reports out-of-sample performance on 56,613 held-out pixels from 140 spatially disjoint blocks. The four models are effectively tied: the spread in RMSE from Random Forest (1.219 °C) to LightGBM (1.223 °C) is four thousandths of a degree, and the corresponding R² values span 0.214 to 0.219. Mean bias is negligible for all four, ranging from −0.008 to +0.002 °C, so no model systematically over- or under-predicts.

Hyperparameters were selected by three-fold GroupKFold *within the training tiles*, so no test block ever informed model selection, and the winning configuration was refitted on the full training set before the single evaluation reported here. The near-identical errors of four models with very different inductive biases also make it implausible that the result is an artefact of tuning.

The low absolute R² deserves direct comment rather than apology. The local-anomaly target has a standard deviation of about 1.4 °C by construction, because subtracting the block-year mean removes most of the variance that a conventional model would be credited with explaining. An R² of 0.22 on this target means the model explains roughly a fifth of the *residual* neighbourhood variation, which is a genuinely difficult quantity to predict from coarse spectral and terrain predictors. The same pipeline applied to the within-year anomaly reaches R² = 0.50 and to raw LST R² = 0.63, as reported in Section 6.2. The three numbers are not competing claims about model quality; they are three different questions.

That all four families converge is itself a finding. A tuned gradient-boosting ensemble, a bagged forest and a tuned perceptron arriving within 0.004 °C of one another implies that the learnable signal is smooth and largely additive at this scale, and that further architectural sophistication would not pay for itself. For a resource-constrained municipal application this is good news: the cheapest model is as good as the most expensive.

![Out-of-sample accuracy of the four models under the local-anomaly target. Differences are within noise.](figures/model_comparison_local.png){w=0.82}

![Predicted against observed local anomaly for the best model. The cloud is centred on the identity line with no systematic curvature.](figures/pred_vs_obs_local.png){w=0.62}

## 6.2 Sensitivity to the target definition

The same pipeline is refitted to the within-year anomaly and to raw LST. The pattern is consistent and instructive.

For the within-year anomaly, R² rises to **0.50** and RMSE to 1.75 °C. All four models again agree to within 0.006 °C, but here the ranking flips: the neural network is marginally best (1.754 °C) and LightGBM marginally worst (1.759 °C), which suggests that at this scale the remaining signal is largely smooth and additive rather than a set of sharp thresholds. The improvement over the local target comes from the fact that this target retains the coastal–inland gradient, which elevation and population describe well. The model is now partly predicting geography rather than neighbourhood process.

Mean bias, however, becomes systematically negative (−0.08 to −0.11 °C) under both the anomaly and raw targets. The cause is visible in the split itself: the held-out blocks are on average 0.32 °C warmer than the training blocks (mean within-year anomaly +0.24 °C against −0.04 °C), because a single random partition of 701 blocks does not guarantee a level-matched test set. A model trained on cooler blocks will under-predict on warmer ones. This is block-sampling variation rather than a defect in the learner, and it is another argument for the local-anomaly target, whose near-zero bias comes from removing block means before fitting.

For raw LST, with the year included as a predictor, accuracy is highest of the three: RMSE is 1.661 °C and R² = 0.626 for both LightGBM and Random Forest, with XGBoost at 1.663 °C and the MLP at 1.685 °C. The model can reproduce both the inter-annual meteorological offset and the spatial gradient. Attribution makes the difference explicit: under this target the calendar year alone contributes 0.88 °C of mean absolute SHAP value, followed by NDBI at 0.68 °C and elevation at 0.57 °C, whereas under the local target NDBI leads at 0.33 °C and year does not appear at all. This is the target a purely operational mapping product would use, and it is the one that would look best in a results table. It is also the least informative about where to plant a tree.

The practical implication is that target choice, not model choice, dominates the reported accuracy. Any urban-heat study that reports a single R² without stating its target definition is not comparable with any other study.

**Table 4.** Out-of-sample performance under the within-year anomaly target.

<!-- results:models:anomaly -->
| Model | RMSE (°C) | MAE (°C) | R² | MBE (°C) | n |
|---|---|---|---|---|---|
| MLP | 1.754 | 1.379 | 0.500 | -0.107 | 56,613 |
| RF | 1.755 | 1.379 | 0.500 | -0.083 | 56,613 |
| XGB | 1.757 | 1.380 | 0.498 | -0.107 | 56,613 |
| LGBM | 1.759 | 1.383 | 0.497 | -0.106 | 56,613 |
<!-- /results:models:anomaly -->

**Table 5.** Out-of-sample performance under the raw-LST target (year included as a predictor).

<!-- results:models:raw -->
| Model | RMSE (°C) | MAE (°C) | R² | MBE (°C) | n |
|---|---|---|---|---|---|
| LGBM | 1.661 | 1.299 | 0.626 | -0.109 | 56,613 |
| RF | 1.661 | 1.290 | 0.626 | -0.095 | 56,613 |
| XGB | 1.663 | 1.299 | 0.625 | -0.111 | 56,613 |
| MLP | 1.685 | 1.314 | 0.616 | -0.077 | 56,613 |
<!-- /results:models:raw -->

## 6.3 Feature attribution

**Table 6.** Global SHAP importance under the local-anomaly target (mean absolute contribution in °C).

<!-- results:shap:local -->
| Rank | Feature | Mean absolute SHAP (°C) |
|---|---|---|
| 1 | NDBI | 0.326 |
| 2 | NDWI | 0.156 |
| 3 | TREE_FRAC | 0.142 |
| 4 | BUILT_FRAC | 0.126 |
| 5 | POP_LOG | 0.088 |
| 6 | WATER_FRAC | 0.087 |
| 7 | ELEV | 0.087 |
| 8 | ALBEDO | 0.061 |
| 9 | NDVI | 0.038 |
<!-- /results:shap:local -->

Under the local-anomaly target the attribution is dominated by surface character rather than by terrain. Built-up intensity is the single most influential feature (mean absolute SHAP 0.326 °C), followed by the moisture index (0.156 °C) and the Dynamic World tree fraction (0.142 °C). The built fraction contributes 0.126 °C. Population and elevation fall to 0.088 and 0.087 °C respectively, which is exactly as intended: the block-mean subtraction has already absorbed the broad coastal–inland gradient those two variables proxy, so they no longer dominate.

One result is worth isolating. NDVI is the *least* influential feature of the nine, at 0.038 °C, while the Dynamic World tree fraction ranks third. This is not a contradiction but a consequence of the sample. Within built-up Karachi, NDVI is sparse, frequently near zero and easily saturated by a single shrub or a patch of grass, whereas the Dynamic World tree fraction is a supervised 10 m land-cover probability that distinguishes canopy from low vegetation. For a city with a median built-up tree fraction of 3.3 per cent, the fraction layer carries information that the index does not.

The signs are physically coherent. Higher built-up intensity and lower tree cover both push a pixel above its neighbourhood mean, and the magnitudes are a few tenths of a degree per standard deviation, consistent with the small residual variance of the target. The model has not discovered a hidden strong driver; it has recovered a modest, plausible one.

![SHAP summary for the LightGBM model, which is tied with the best model to within 0.005 °C (Section 4.6). Each point is one sampled pixel; horizontal position is that feature's contribution to the pixel's predicted anomaly.](figures/shap_summary_local.png){w=0.85}

## 6.4 Counterfactual cooling scenarios

**Table 7.** Simulated change in predicted LST for three intervention scenarios, for each target definition.

<!-- results:scenarios:all -->
| Target | Scenario | Mean ΔLST (°C) | Median ΔLST (°C) | Cells < −0.5 °C (%) | Cells < −1.0 °C (%) |
|---|---|---|---|---|---|
| Local | greening +10% NDVI | -0.60 | -0.58 | 60.8 | 10.1 |
| Local | cool roof +0.15 albedo | -0.04 | -0.01 | 3.9 | 1.0 |
| Local | combined | -0.60 | -0.59 | 61.6 | 10.9 |
| Within-year | greening +10% NDVI | -1.12 | -1.01 | 69.1 | 50.4 |
| Within-year | cool roof +0.15 albedo | -0.42 | -0.43 | 47.5 | 28.7 |
| Within-year | combined | -2.03 | -1.93 | 81.9 | 72.3 |
| Raw | greening +10% NDVI | -0.95 | -0.89 | 85.8 | 38.9 |
| Raw | cool roof +0.15 albedo | -0.09 | +0.00 | 9.5 | 3.4 |
| Raw | combined | -0.95 | -0.86 | 81.3 | 37.7 |
<!-- /results:scenarios:all -->

Greening produces a consistent and correctly signed effect under every target. On the local-anomaly target, raising NDVI by 0.10 and tree fraction by 0.05 lowers predicted LST by 0.60 °C on average (median 0.58 °C), with 61 per cent of held-out pixels cooling by more than 0.5 °C and 10 per cent by more than 1.0 °C. The magnitude is modest — about 0.44 of the target's standard deviation — but it is systematic, it is in the expected direction, and it is achieved in a city whose median built-up tree fraction is only 3.3 per cent. A five-percentage-point increase would roughly double canopy cover in the areas that most need it, so the scenario is ambitious but not implausible over a multi-year programme.

The cool-roof scenario, by contrast, is unstable across targets. On the local-anomaly target, increasing albedo by 0.15 produces a mean change of −0.04 °C, a median of −0.01 °C, and cooling above 0.5 °C in fewer than 4 per cent of pixels: the model is, to all practical purposes, indifferent. On the within-year anomaly target the same perturbation produces a mean cooling of 0.42 °C, and on raw LST 0.09 °C. The combined scenario follows suit, moving from −0.60 °C on the local target to −2.03 °C on the anomaly target.

That instability is itself the finding. Albedo's apparent effect is not a stable property of the surface; it is a function of how much of the coastal–inland gradient the target retains. A study reporting only the anomaly-target number would conclude that cool roofs work in Karachi. A study reporting only the local-target number would conclude that they do nothing. Neither single number is trustworthy on its own, which is the argument for reporting all three.

![Modelled change in LST under a +0.10 NDVI greening scenario, mapped over the held-out pixels.](figures/scenario_greening_map.png){w=0.72}

## 6.5 Spatial patterns

![Observed pre-monsoon LST across the built-up sample, 2024.](figures/heatmap_2024.png){w=0.95}

Mapping the sample reproduces the known geography of Karachi's heat. The inland north-eastern fringe, comprising Gadap and the adjoining dry peripheries, is consistently the hottest zone, with surface temperatures in the high forties and low fifties. The coastal strip, from Clifton through Saddar to Keamari, is cooler by several degrees, reflecting sea breeze, higher humidity and lower elevation. Intermediate densities in Gulshan and the central districts sit between the two.

The within-neighbourhood variation is large enough to justify the whole exercise. Inside a single 2 km block, the median observed range of LST across sampled pixels is 4.8 °C, the 75th percentile is 6.8 °C and the 90th percentile is 8.5 °C. A 1 km product such as MODIS would average away most of that spread, which is precisely the information an intervention planner needs.

The local-anomaly map is more useful for intervention targeting than the raw map, because it removes the coastal–inland gradient and exposes the pockets that are hot *relative to their own surroundings*. Those pockets are where a tree or a shaded corridor has the highest marginal return, since the surrounding area already demonstrates that cooler conditions are achievable at that location.

# 7. Discussion

## 7.1 Why a low R² is the right result

The most likely misreading of this paper is that R² = 0.22 indicates a weak model. It does not. It indicates a hard target, deliberately chosen. The local anomaly removes the between-block variance that any model can learn from elevation alone, and asks the model to explain within-block variation from predictors that are themselves spatially smooth. A model that achieved R² = 0.8 on this target would be suspicious, because it would imply that 30 m spectral indices encode neighbourhood-scale temperature differences far more precisely than the physics of advection and storage would allow.

The value of the target is that its errors are interpretable. An RMSE of 1.22 °C on a quantity whose own standard deviation is 1.4 °C means the model is useful for ranking — it can tell you which of two nearby pixels is likely hotter — without being precise enough to substitute for measurement. For prioritisation, ranking is what is needed.

## 7.2 The albedo paradox

The cool-roof result contradicts a large literature and deserves a careful explanation. Three mechanisms plausibly contribute.

First, confounding by moisture. In Karachi's dusty pre-monsoon environment, the highest-albedo surfaces are dry bare ground, degraded plots and construction sites. These are also the hottest surfaces, because they lack the evaporative cooling that vegetation and moist soil provide. A model fitted to observational data learns the joint distribution, and the albedo coefficient absorbs the moisture deficit. The result is a positive association that is real in the data and wrong as a causal claim.

Second, the surface-energy balance differs from the temperate cities where cool-roof evidence was established. A reflective roof reduces absorbed shortwave radiation, which lowers surface temperature. But in a coastal subtropical setting with high humidity and frequent advection, the roof surface temperature is a weaker determinant of the air temperature experienced at street level, and the benefit is diluted. The present analysis measures surface temperature, so it is measuring the quantity a cool roof most directly affects — and still finds no benefit, which points to the confounding explanation rather than the advection one.

Third, the perturbation is applied to a feature that the model uses as a weak, confounded proxy. Increasing albedo by 0.15 in feature space moves a pixel into a region of the training distribution occupied by bright dry land, and the model responds as it was taught to respond to bright dry land — that is, hardly at all, because it has learned that brightness and heat co-occur. The net simulated change is −0.04 °C, which is not a cooling effect but noise around zero.

The target-dependence of the result is the strongest evidence for this reading. The same perturbation cools 0.04 °C under the local target, 0.09 °C under raw LST and 0.42 °C under the within-year anomaly — a tenfold swing that tracks exactly how much of the coastal–inland gradient each target retains. A genuine physical effect of albedo on surface temperature would not change by an order of magnitude when the target is redefined. The instability means the albedo coefficient is absorbing whatever spatial gradient the target leaves in, which is the definition of confounding.

The correct conclusion is not that cool roofs do not work. It is that observational machine learning cannot answer the question, and that a credible cool-roof assessment for Karachi requires either a physical urban canopy model or a field experiment. Reporting the negative result, with the mechanism, is more valuable to a policymaker than a plausible-looking number derived from a confounded model.

## 7.3 Elevation and population as climate proxies

Once a target retains the coastal–inland gradient, elevation becomes the dominant predictor, and it is tempting to read this as a terrain effect. It is not. Karachi's elevation range is only about 300 m, which cannot produce a multi-degree surface temperature difference through lapse rate alone. The local-anomaly model, which removes that gradient, ranks elevation seventh of nine features with a mean absolute SHAP value of 0.087 °C, against 0.326 °C for NDBI — a direct demonstration of how much of elevation's apparent power is geography rather than terrain. Elevation is instead a proxy for distance from the coast, and hence for humidity, wind and aerosol loading. Population is collinear with the same gradient, which is why its coefficient is negative despite the obvious fact that dense settlement is not a cooling technology.

This has a direct methodological consequence: a driver analysis that reports "population reduces urban heat" from such a model would be publishing an artefact. The same gradient explains the negative urban–rural contrast in Section 5: the classical surface heat island of Karachi is negative because the rural reference lies inland and uphill, not because the city is cool. The defence used here is the local-anomaly target, which absorbs the gradient by construction, and an explicit statement of the confound in the discussion. Any future version of this work that adds a distance-to-coast feature should see elevation's apparent importance collapse, which would be a useful validation.

## 7.4 Implications for intervention prioritisation

Three practical conclusions follow.

Greening is the intervention with the strongest and most defensible modelled support. The effect is modest in absolute terms — about 0.6 °C of mean surface cooling — but it is an order of magnitude larger than the albedo scenario, correctly signed, and consistent with the land-cover evidence in Section 5, where trees are more than three degrees cooler than built surfaces. Because canopy cover in built-up Karachi is so low, marginal gains are available almost everywhere.

Targeting should use the local anomaly, not raw temperature. The hottest absolute pixels are largely determined by geography and cannot be changed; the hottest pixels *relative to their neighbours* are where an intervention has leverage.

Albedo-based interventions should not be prioritised on the basis of this analysis. They may well work, but this study provides no reliable estimate of how much: the simulated effect ranges from 0.04 °C on the local target through 0.09 °C on raw LST to 0.42 °C on the anomaly target, and the observational association between albedo and LST runs in the wrong direction. A decision-maker given only one of those numbers would be misled.

# 8. Limitations

Several limitations bound the claims that can be made.

Surface temperature is not air temperature. LST responds to insolation and material properties within minutes, whereas heat stress experienced by people depends on air temperature, humidity, wind and mean radiant temperature. The maps in this paper are a proxy for heat exposure, not a measurement of it, and the two can diverge substantially over hard, dry surfaces.

The analysis is associational. SHAP values describe how the fitted model uses each feature; they do not identify causal effects. The counterfactual scenarios are therefore best understood as "what pixels with these properties tend to look like" rather than "what would happen if this intervention were built". The albedo result in Section 7.2 is the clearest illustration of the difference.

The counterfactuals assume pixel independence. In reality, cooling one block affects its neighbours through advection, and a large-scale greening programme would alter the regional boundary layer in ways a pixel-wise surrogate cannot represent.

WorldPop provides only a 2020 population surface in the collection used, so the population feature is static across 2021–2025. This limits the model's ability to capture demographic change, although population is in any case a proxy rather than a mechanism.

The albedo layer contains physically impossible values (up to 1.94) before clipping, indicating residual cloud or compositing artefacts that survive the QA mask. Clipping to 0.02–0.65 removes the worst of it, but the layer remains the least reliable predictor in the stack.

The study covers one city, one season and six years. The fitted relationships should not be transferred to other cities, to other seasons, or to a future climate without refitting and revalidation. Monsoon-season dynamics in particular are not represented at all.

The rural reference used for the urban–rural contrast is defined only by built fraction, and in Karachi it selects mostly inland bare desert. A contrast against irrigated farmland or mangrove, which are the cooler land covers in the region, would produce a different and probably positive number. The negative surface heat island reported in Section 5 is therefore a statement about this particular reference choice as much as about the city, and it is reported as such rather than as a general claim about Karachi.

Finally, the evaluation, while spatially blocked, uses a single random partition of blocks. That partition leaves a small level difference between training and test blocks (Section 6.2), and a repeated blocked cross-validation with multiple partitions would give a tighter estimate of the uncertainty on the reported metrics. It is a straightforward extension.

# 9. Conclusion and Future Work

This paper set out to build an explainable machine-learning digital twin for urban heat in Karachi, and to be explicit about what such a twin can and cannot support.

A six-year, 30 m pre-monsoon LST record for Karachi Division was constructed entirely within Google Earth Engine's free tier, combining Landsat thermal data with Sentinel-2 indices, Dynamic World land-cover fractions, SRTM elevation and WorldPop population. From 360,000 built-up sample points, four model families were tuned with Optuna and evaluated on spatially disjoint 2 km blocks. Under a local-anomaly target they converge to RMSE ≈ 1.22 °C and R² ≈ 0.22 with negligible bias; under a within-year anomaly the same pipeline reaches RMSE 1.75 °C and R² = 0.50, and under raw LST RMSE 1.66 °C and R² = 0.63. The comparison demonstrates that target definition, not model architecture, dominates reported accuracy, and that spatial blocking is essential for an honest estimate.

Attribution and counterfactual simulation identify vegetation as the only intervention with a consistent modelled cooling effect of practical magnitude, and produce a negative result for albedo that is traced to confounding by surface moisture rather than dismissed. We regard the negative result as a contribution: it demonstrates that an explainable surrogate can protect a decision-maker from an unsupported conclusion that a less transparent model would have delivered with confidence.

Four extensions follow naturally. First, a causal design — an interrupted time series around parks and cool-roof projects that have actually been built — would replace the associational counterfactuals with something closer to evidence. Second, adding distance-to-coast and a boundary-layer height proxy would allow the elevation confound to be tested directly rather than merely acknowledged. Third, coupling the surrogate to a weather reanalysis would let it produce air-temperature and heat-index estimates rather than surface temperature, which is what public-health planning requires. Fourth, the same pipeline is city-agnostic: the Earth Engine export script takes a bounding box and a year range, so replication in Hyderabad, Lahore or Dhaka is a matter of configuration rather than redevelopment.

The broader point is that explainability and honesty about validation are not decoration on an urban-heat study; they are the difference between a model that supports a spending decision and one that merely produces a map.

# Data and Code Availability

All inputs are public: Landsat 8/9 Collection 2 (USGS), Sentinel-2 (ESA), Dynamic World (Google), SRTM (NASA) and WorldPop (University of Southampton).

**Code:** https://github.com/Gotam-Dulhani/coolkarachi — the six yearly export scripts, the exploratory analysis, the modelling pipeline and the paper and slide builders are included under `src/`.

**Data:** https://drive.google.com/drive/folders/1ZbUw7r6Sor7C47uXzFwsOGu8laFZ32Cg — the exported tables (`karachi_features_<year>.csv` and `karachi_ml_<year>.csv`, 2020–2025) and the cleaned 360,000-row sample.

The tables can also be regenerated in about an hour with an approved Earth Engine account using the commands in `README.md`. Random seeds, tuned hyperparameters and the full evaluation outputs are stored in `results/` so that every number in this paper can be reproduced.

# Appendix A. Tuned Hyperparameters

**Table 8.** Hyperparameters selected by Optuna for the local-anomaly target. Each configuration is the best of fifteen TPE trials scored by three-fold GroupKFold on the training tiles.

<!-- results:params:local -->
| Model | Selected hyperparameters |
|---|---|
| RF | n_estimators = 600, max_depth = 18, min_samples_leaf = 1, max_features = 0.3353 |
| XGB | n_estimators = 700, learning_rate = 0.01889, max_depth = 5, subsample = 0.6734, colsample_bytree = 0.7217, min_child_weight = 6 |
| LGBM | n_estimators = 600, learning_rate = 0.01667, num_leaves = 22, min_child_samples = 48, subsample = 0.9863 |
| MLP | hidden = 64, alpha = 5.481e-06, lr = 0.0001031 |
<!-- /results:params:local -->

# References

[1] T. R. Oke, "The energetic basis of the urban heat island," *Quarterly Journal of the Royal Meteorological Society*, vol. 108, no. 455, pp. 1–24, 1982.

[2] J. A. Voogt and T. R. Oke, "Thermal remote sensing of urban climates," *Remote Sensing of Environment*, vol. 86, no. 3, pp. 370–384, 2003.

[3] Q. Weng, "Thermal infrared remote sensing for urban climate and environmental studies: Methods, applications, and trends," *ISPRS Journal of Photogrammetry and Remote Sensing*, vol. 64, no. 4, pp. 335–344, 2009.

[4] M. L. Imhoff, P. Zhang, R. E. Wolfe, and L. Bounoua, "Remote sensing of the urban heat island effect across biomes in the continental USA," *Remote Sensing of Environment*, vol. 114, no. 3, pp. 504–513, 2010.

[5] L. Zhao, X. Lee, R. B. Smith, and K. Oleson, "Strong contributions of local background climate to urban heat islands," *Nature*, vol. 511, no. 7508, pp. 216–219, 2014.

[6] T. Chakraborty and X. Lee, "A simplified urban-extent algorithm to characterize surface urban heat islands on a global scale and examine vegetation control on their spatiotemporal variability," *International Journal of Applied Earth Observation and Geoinformation*, vol. 74, pp. 269–280, 2019.

[7] D. E. Bowler, L. Buyung-Ali, T. M. Knight, and A. S. Pullin, "Urban greening to cool towns and cities: A systematic review of the empirical evidence," *Landscape and Urban Planning*, vol. 97, no. 3, pp. 147–155, 2010.

[8] H. Akbari, S. Menon, and A. Rosenfeld, "Global cooling: Increasing world-wide urban albedos to offset CO2," *Climatic Change*, vol. 94, no. 3–4, pp. 275–286, 2009.

[9] L. Breiman, "Random forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001.

[10] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 2016, pp. 785–794.

[11] G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T.-Y. Liu, "LightGBM: A highly efficient gradient boosting decision tree," in *Advances in Neural Information Processing Systems 30*, 2017, pp. 3146–3154.

[12] T. Akiba, S. Sano, T. Yanase, T. Ohta, and M. Koyama, "Optuna: A next-generation hyperparameter optimization framework," in *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 2019, pp. 2623–2631.

[13] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *Advances in Neural Information Processing Systems 30*, 2017, pp. 4765–4774.

[14] D. R. Roberts, V. Bahn, S. Ciuti, M. S. Boyce, J. L. Elith, G. Guillera-Arroita, S. Hauenstein, J. J. Lahoz-Monfort, B. Schröder, W. Thuiller, D. I. Warton, B. A. Wintle, T. Hartig, and C. F. Dormann, "Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure," *Ecography*, vol. 40, no. 8, pp. 913–929, 2017.

[15] P. Bauer, B. Stevens, and W. Hazeleger, "A digital twin of Earth for the green transition," *Nature Climate Change*, vol. 11, pp. 80–83, 2021.

[16] N. Gorelick, M. Hancher, M. Dixon, S. Ilyushchenko, D. Thau, and R. Moore, "Google Earth Engine: Planetary-scale geospatial analysis for everyone," *Remote Sensing of Environment*, vol. 202, pp. 18–27, 2017.

[17] C. F. Brown, S. P. Brumby, B. Guzder-Williams, T. Birch, S. B. Hyde, J. Mazzariello, W. Czerwinski, V. J. Pasquarella, R. Haertel, S. Ilyushchenko, K. Schwehr, M. Weisse, F. Stolle, C. Hanson, O. Guinan, R. Moore, and A. M. Tait, "Dynamic World, near real-time global 10 m land use land cover mapping," *Scientific Data*, vol. 9, no. 1, article 251, 2022.

[18] T. G. Farr, P. A. Rosen, E. Caro, R. Crippen, R. Duren, S. Hensley, M. Kobrick, M. Paller, E. Rodriguez, L. Roth, D. Seal, S. Shaffer, J. Shimada, J. Umland, M. Werner, M. Oskin, D. Burbank, and D. Alsdorf, "The Shuttle Radar Topography Mission," *Reviews of Geophysics*, vol. 45, no. 2, RG2004, 2007.

[19] A. J. Tatem, "WorldPop, open data for spatial demography," *Scientific Data*, vol. 4, article 170004, 2017.

[20] M. Drusch, U. Del Bello, S. Carlier, O. Colin, V. Fernandez, F. Gascon, B. Hoersch, C. Isola, P. Laberinti, P. Martimort, A. Meygret, F. Spoto, O. Sy, F. Marchese, and P. Bargellini, "Sentinel-2: ESA's optical high-resolution mission for GMES operational services," *Remote Sensing of Environment*, vol. 120, pp. 25–36, 2012.

[21] S. L. Ermida, P. Soares, V. Mantas, F.-M. Göttsche, and I. F. Trigo, "Google Earth Engine open-source code for land surface temperature estimation from the Landsat series," *Remote Sensing*, vol. 12, no. 9, article 1471, 2020.

[22] IPCC, *Climate Change 2021: The Physical Science Basis. Contribution of Working Group I to the Sixth Assessment Report of the Intergovernmental Panel on Climate Change*. Cambridge, U.K.: Cambridge University Press, 2021.
