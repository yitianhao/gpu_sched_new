/*
 * Hardcoded latency profile for FasterRCNN_ResNet50_FPN
 */

#ifndef _LAT_PROF_H
#define _LAT_PROF_H

#include <cstddef>
#include <stdint.h>

static size_t total_kernel_calls = 929;
static float kernel_lats[] = {
45.52888822, 47.47377822, 54.96533278, 25.69244411, 49.809777  , 9.006222111,
394.5315551, 7.043555556, 6.275555556, 5.959111   , 4.807111111, 5.368888889,
241.5111102, 241.5431127, 241.8524422, 211.7084401, 7.576888889, 74.38577767,
6.055111111, 6.261333333, 5.756444444, 4.764444444, 5.013333333, 63.71555489,
63.96088878, 63.34577722, 7.854222222, 264.1244472, 6.439111111, 5.180444444,
4.952888889, 5.411555556, 5.233777778, 63.94310933, 63.83288867, 63.67644333,
7.143111111, 235.3457778, 5.966222222, 5.656888889, 5.624888889, 4.942222222,
5.148444444, 241.685335 , 240.899558 , 6.734222222, 232.451555 , 5.788444444,
5.329777778, 5.112888889, 4.778666667, 5.109333333, 242.1262206, 241.0240003,
351.0364482, 240.5297767, 6.808888889, 171.8720024, 5.582222222, 5.063111111,
5.532444444, 4.714666667, 5.105777778, 63.67644367, 63.83644389, 63.58755489,
7.114666667, 263.1822189, 5.934222222, 4.668444444, 5.070222222, 4.515555556,
4.842666667, 63.60177733, 63.62666667, 63.39199956, 6.421333333, 233.2977819,
5.525333333, 4.764444444, 5.287111111, 4.533333333, 4.711111111, 241.6604429,
241.0346661, 350.7235549, 240.1173333, 6.929777778, 170.5137822, 5.838222222,
4.899555556, 4.757333333, 5.639111111, 5.173333333, 63.55555467, 63.85777611,
63.20355522, 6.300444444, 263.1857774, 5.511111111, 5.376      , 4.977777778,
4.696888889, 4.707555556, 63.90755456, 63.53066678, 63.34577633, 6.144      ,
235.2106661, 5.667555556, 5.159111111, 4.824888889, 4.711111111, 4.668444444,
241.6106653, 240.8782211, 350.9795566, 240.0071088, 6.218666667, 329.8133376,
5.984      , 5.173333333, 4.568888889, 4.949333333, 4.561777778, 122.9724418,
122.6773317, 122.3644443, 6.506666667, 391.9751112, 5.482666667, 5.056      ,
5.422222222, 4.426666667, 4.579555556, 31.75466667, 34.03377733, 33.43999944,
6.044444444, 186.8693338, 5.856      , 5.240888889, 4.849777778, 4.611555556,
4.792888889, 123.5875567, 122.6133321, 6.631111111, 355.4737718, 5.482666667,
4.928      , 4.913777778, 5.326222222, 4.785777778, 123.1715544, 122.7555551,
177.5928871, 122.0017794, 6.225777778, 166.8302172, 6.005333333, 4.867555556,
4.949333333, 4.846222222, 4.860444444, 32.45511178, 34.453334  , 33.73155633,
7.953777778, 254.6773341, 5.393777778, 5.013333333, 5.148444444, 4.529777778,
4.597333333, 32.98133278, 34.22577767, 33.32266711, 5.802666667, 186.6915556,
5.560888889, 4.810666667, 4.992      , 4.558222222, 4.867555556, 125.1022228,
122.7519988, 177.6533322, 121.7742226, 6.183111111, 166.890664 , 5.422222222,
6.332444444, 4.839111111, 4.572444444, 4.579555556, 32.66844511, 34.30400011,
33.472     , 7.939555556, 254.4391089, 5.340444444, 4.760888889, 4.842666667,
4.821333333, 4.696888889, 32.79999956, 34.23999867, 33.83111156, 6.023111111,
186.4000022, 6.005333333, 5.162666667, 4.508444444, 4.732444444, 4.590222222,
123.7226664, 122.6239979, 177.9804418, 121.7848901, 6.485333333, 166.808882 ,
5.450666667, 4.892444444, 5.187555556, 4.558222222, 4.696888889, 32.63288978,
34.215111  , 33.46488967, 7.761777778, 254.3964486, 5.717333333, 5.031111111,
4.664888889, 4.707555556, 4.586666667, 32.86755589, 34.00177844, 33.03466689,
6.478222222, 186.4355569, 5.344      , 4.508444444, 5.162666667, 4.462222222,
4.561777778, 125.1768893, 122.7448874, 177.7564444, 121.7244466, 6.339555556,
326.4604491, 5.472      , 5.105777778, 4.821333333, 5.020444444, 4.707555556,
62.42844478, 63.53422167, 63.004443  , 5.738666667, 351.5377773, 5.216      ,
5.340444444, 4.558222222, 4.849777778, 4.764444444, 13.57511089, 18.72000033,
9.312000111, 5.866666667, 171.125334 , 5.319111111, 4.736      , 4.792888889,
5.006222222, 4.711111111, 64.03555533, 63.65866578, 6.595555556, 335.7475519,
5.564444444, 4.711111111, 5.28       , 4.572444444, 4.618666667, 63.19644289,
63.626666  , 91.24977811, 62.73777956, 6.016      , 164.2951117, 5.6        ,
5.048888889, 4.679111111, 4.686222222, 4.547555556, 13.21955544, 18.872889  ,
9.198222111, 16.51911089, 239.0079973, 5.294222222, 5.340444444, 4.615111111,
4.757333333, 4.792888889, 17.86666656, 18.99022222, 9.009777889, 5.969777778,
170.901335 , 5.187555556, 5.016888889, 4.593777778, 4.995555556, 4.48       ,
63.80444322, 63.87911089, 91.44533389, 62.552889  , 6.058666667, 164.3839992,
5.781333333, 5.006222222, 4.679111111, 4.689777778, 4.600888889, 13.30844467,
18.92622211, 9.063111111, 16.10311078, 238.4853362, 5.368888889, 5.027555556,
5.255111111, 4.732444444, 4.906666667, 16.34488878, 18.63111122, 9.105777667,
6.037333333, 170.3608891, 5.624888889, 4.497777778, 4.675555556, 4.419555556,
4.622222222, 64.09599978, 63.65155478, 91.13600011, 62.78044633, 5.799111111,
164.3840044, 5.44       , 5.304888889, 4.700444444, 4.423111111, 4.906666667,
13.36177778, 19.21777789, 9.095111111, 16.10666667, 238.4924503, 5.315555556,
4.846222222, 4.718222222, 4.796444444, 4.768      , 16.63644444, 19.31022222,
9.169777778, 5.987555556, 170.8408866, 5.219555556, 5.607111111, 4.597333333,
4.526222222, 4.469333333, 64.07466522, 63.555556  , 91.16800022, 62.34311089,
6.442666667, 164.4195557, 4.899555556, 4.476444444, 5.194666667, 4.565333333,
4.931555556, 13.50755544, 18.91911133, 9.208888667, 15.71555567, 238.8480003,
5.383111111, 4.650666667, 4.760888889, 4.526222222, 4.458666667, 16.44444433,
19.075556  , 9.073777667, 6.264888889, 170.3715548, 5.048888889, 4.775111111,
4.693333333, 4.718222222, 4.810666667, 64.060443  , 63.49155467, 91.60888922,
62.79822111, 5.909333333, 164.583113 , 5.194666667, 4.835555556, 4.853333333,
5.038222222, 4.647111111, 13.38666656, 18.97955522, 9.237333556, 16.03911133,
238.6844467, 5.521777778, 5.518222222, 4.835555556, 4.647111111, 4.764444444,
16.84622211, 18.94400044, 9.055999778, 5.880888889, 170.8195598, 5.063111111,
4.704      , 4.753777778, 4.896      , 4.551111111, 63.73333233, 63.80799944,
91.35288933, 62.41066744, 5.984      , 330.0444504, 5.596444444, 4.849777778,
4.846222222, 4.935111111, 4.590222222, 31.761778  , 34.33599978, 34.09422311,
6.122666667, 421.2657843, 5.336888889, 4.892444444, 5.134222222, 4.423111111,
4.668444444, 7.612444444, 7.420444444, 5.777777778, 5.731555556, 170.3146667,
6.812444444, 4.8        , 4.899555556, 4.494222222, 4.96       , 34.64177744,
34.30755622, 5.923555556, 338.9368897, 5.681777778, 4.835555556, 4.928      ,
4.842666667, 4.579555556, 32.10666656, 34.23999989, 47.92888878, 33.24800067,
5.923555556, 190.1937781, 5.475555556, 4.579555556, 4.586666667, 4.622222222,
4.579555556, 7.665777667, 6.926222222, 5.688888889, 59.23911244, 228.3377839,
5.155555556, 4.888888889, 4.999111111, 4.359111111, 4.778666667, 10.05511078,
7.000888889, 5.674666667, 5.987555556, 168.1777784, 5.774222222, 4.814222222,
4.568888889, 4.440888889, 4.483555556, 34.07644389, 34.357333  , 48.13511078,
32.88533456, 5.728      , 189.8808883, 5.429333333, 5.493333333, 4.490666667,
4.824888889, 4.455111111, 7.392      , 7.082666667, 5.898666667, 59.210667  ,
227.9573329, 5.191111111, 5.237333333, 4.497777778, 5.12       , 4.604444444,
10.23288878, 7.456      , 5.824      , 5.952      , 168.5013361, 5.436444444,
5.379555556, 4.636444444, 4.707555556, 4.654222222, 34.48177911, 34.30044522,
48.45866689, 33.02400044, 37.41155578, 5.020444444, 95.14311133, 6.595555556,
6.748444444, 13.66399989, 64.12088933, 6.368      , 6.389333333, 164.0711076,
11.10044444, 18.72355556, 22.432     , 15.10755578, 237.9199982, 15.57333333,
6.193777778, 325.1022169, 62.17600011, 52.62577722, 91.05778   , 15.36355578,
891.6373358, 61.280001  , 6.432      , 646.0053508, 240.9066651, 180.5475549,
351.2568969, 16.21688944, 3362.304009, 238.8373329, 7.864888889, 16.84266644,
3362.712918, 240.4586709, 240.832004 , 5.863111111, 129.699557 , 5.500444444,
6.417777778, 125.9555529, 8.558222222, 16.18488911, 888.8568723, 61.40444356,
63.374221  , 5.816888889, 59.38844456, 5.041777778, 5.973333333, 47.53066633,
5.763555556, 11.89333344, 239.470225 , 15.185778  , 6.865777778, 23.77600011,
5.059555556, 22.95111111, 5.635555556, 15.44888867, 64.57599944, 6.464      ,
5.457777778, 18.23288933, 5.152      , 17.85244478, 4.504888889, 14.70933333,
53.83466756, 5.372444444, 5.347555556, 18.58133333, 5.301333333, 18.282667  ,
4.714666667, 5.92       , 4.455111111, 4.522666667, 4.476444444, 4.462222222,
4.330666667, 4.632888889, 4.529777778, 4.309333333, 4.558222222, 6.161777778,
7.349333333, 4.757333333, 4.920888889, 6.922666778, 4.782222222, 10.15466667,
14.50311122, 4.853333333, 5.550222222, 4.647111111, 5.045333333, 5.205333333,
4.778666667, 6.826666667, 7.893333333, 4.615111111, 5.141333333, 4.384      ,
4.565333333, 4.888888889, 4.718222222, 6.090666667, 7.509333333, 4.817777778,
5.290666667, 4.604444444, 4.775111111, 5.194666667, 4.896      , 6.087111111,
7.484444444, 6.570666667, 4.721777778, 4.771555556, 4.920888889, 5.237333333,
4.871111111, 5.795555556, 7.701333444, 14.40355544, 9.045333222, 16.67200056,
5.621333333, 8.156444333, 5.304888889, 5.76       , 5.212444444, 4.988444444,
5.475555556, 4.949333333, 10.16533344, 20.17777778, 9.059555556, 7.182222222,
6.606222111, 12.92444456, 6.965333333, 13.04888889, 14.58133322, 7.025777778,
8.341333444, 7.331555556, 6.186666667, 4.782222222, 8.732444444, 8.551111   ,
9.511111222, 6.762666667, 7.640888889, 6.652444444, 6.819555556, 6.812444444,
7.249777778, 6.314666667, 7.075555556, 7.687111111, 6.492444444, 7.559111111,
43.19644389, 6.862222222, 4.398222222, 4.334222222, 4.906666667, 4.544      ,
8.664888778, 6.826666667, 8.728888778, 5.816888889, 16.15288911, 15.569778  ,
15.210667  , 14.574222  , 6.108444444, 5.902222222, 10.15466656, 7.192888889,
7.285333333, 7.296      , 83.54488822, 25.53600033, 4.871111111, 4.888888889,
7.608888889, 6.645333333, 6.279111111, 6.496      , 21.45777767, 25.14133333,
4.583111111, 29.51466667, 24.72888889, 4.611555556, 13.13777789, 22.86222222,
4.551111111, 9.027555556, 5.479111111, 6.282666667, 9.333333444, 4.817777778,
9.027555667, 4.913777778, 8.227555556, 6.247111111, 6.389333333, 4.945777778,
6.076444444, 5.436444444, 5.048888889, 5.770666667, 4.579555556, 5.621333333,
6.627555556, 6.094222111, 6.940444444, 7.100444444, 6.343111111, 6.716444444,
6.392889   , 5.198222222, 5.226666667, 5.756444444, 6.919111111, 6.453333333,
6.542222222, 10.24355556, 6.048      , 5.372444444, 4.778666667, 4.789333333,
5.397333333, 7.100444444, 5.287111111, 11.72622222, 10.82311111, 10.78755578,
10.30755544, 7.260444444, 87.69777933, 9.354666556, 8.138666667, 6.481777778,
6.378666667, 6.151111111, 5.194666667, 4.832      , 4.522666667, 5.667555556,
5.031111111, 6.122666778, 4.661333333, 4.593777778, 5.873777667, 4.835555556,
8.533333444, 5.265777778, 85.91999978, 5.763555556, 5.258666667, 6.087111111,
5.909333333, 7.925333333, 59.07555567, 678.1262072, 124.7395546, 5.006222222,
5.393777778, 5.777777778, 5.934222222, 7.768888778, 16.50133356, 159.7048881,
30.78044444, 4.672      , 5.361777778, 5.653333333, 5.92       , 7.416888778,
15.77955567, 121.0026661, 30.01600056, 4.544      , 5.223111111, 7.317333333,
5.614222222, 6.922666667, 6.631111111, 46.77688989, 11.56977778, 9.667555778,
2099.623115, 12.38044467, 11.63022244, 184.4017793, 13.50399989, 5.863111111,
40.53688889, 6.755555556, 80.03555544, 6.286222222, 5.297777778, 5.624888889,
4.796444444, 6.961777778, 5.283555556, 6.588444444, 5.976888889, 6.094222222,
5.674666667, 4.920888889, 5.024      , 5.539555556, 6.424888889, 6.08       ,
5.532444444, 5.880888889, 5.418666667, 4.551111111, 5.191111111, 5.831111111,
5.713777778, 4.910222222, 4.391111111, 4.558222222, 4.512      , 10.16533333,
8.316444444, 7.082666667, 5.827555556, 9.247999889, 5.358222222, 7.004444444,
5.728      , 5.415111111, 5.301333333, 6.378666667, 6.503111111, 5.344      ,
7.683555556, 7.328      , 6.336      , 6.709333333, 5.098666667, 4.508444444,
4.992      , 4.675555556, 5.020444444, 5.084444444, 5.105777778, 5.603555556,
7.168      , 6.254222222, 6.602666667, 7.943111111, 6.140444444, 5.176888889,
5.095111111, 4.611555556, 4.881777778, 19.10755567, 5.923555556, 19.829333  ,
6.734222222, 7.694222222, 4.963555556, 4.995555556, 7.128888889, 5.162666667,
5.376      , 4.583111111, 4.732444444, 4.547555556, 6.136888889, 
};

#endif