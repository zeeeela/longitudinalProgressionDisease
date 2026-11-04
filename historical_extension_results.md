# Historical HPC output evidence

Source: main_notebook_clean_ext_multigpu.ipynb. These are saved outputs, not local reruns.


## Original cell 35 (zero-based)

```text
pairs: 915 | feature columns: 77

```

```text
                     cur   prev  delta
feature                               
antibiotic_active  1.000  1.000  1.000
antibiotic_new24h  1.000  1.000  1.000
diuretic_active    1.000  1.000  1.000
diuretic_new24h    1.000  1.000  1.000
inpatient          1.000  1.000  1.000
icu                1.000  1.000  1.000
creatinine         0.795  0.786  0.690
bun                0.795  0.786  0.690
hemoglobin         0.793  0.785  0.690
wbc                0.792  0.784  0.687
sodium             0.789  0.773  0.675
hr                 0.369  0.360  0.289
rr                 0.367  0.357  0.286
spo2               0.367  0.357  0.286
net_fluid_24h      0.364  0.355  0.286
sbp                0.360  0.350  0.277
lactate            0.390  0.390  0.269
fio2               0.274  0.261  0.220
albumin            0.286  0.283  0.161
weight             0.234  0.237  0.133
troponin_t         0.114  0.102  0.059
temp_c             0.027  0.023  0.012
crp                0.013  0.010  0.005
ntprobnp           0.000  0.000  0.000
```


## Original cell 36 (zero-based)

```text
            disease split  n_pairs  accuracy  balanced_accuracy  macro_f1
0  Pleural effusion   val       28  0.428571           0.355026  0.349393
1      Cardiomegaly   val       54  0.888889           0.333333  0.313725
2             Edema   val       24  0.541667           0.530303  0.459341
3         Pneumonia   val       16  0.500000           0.358333  0.307359
4   Pulmonary edema   val       22  0.545455           0.513468  0.452991
5  Pleural effusion  test       55  0.581818           0.546032  0.548806
6      Cardiomegaly  test       57  0.824561           0.333333  0.301282
7             Edema  test       28  0.285714           0.238889  0.250000
8         Pneumonia  test       12  0.833333           0.857143  0.555556
9   Pulmonary edema  test       27  0.444444           0.401010  0.395421
```


## Original cell 41 (zero-based)

```text
Loading weights:   0%|          | 0/223 [00:00<?, ?it/s]
```

```text
                                      Pleural effusion  Cardiomegaly  Edema  \
experiment                                                                    
biomedclip                                       0.557         0.574  0.441   
densenet121_imagenet                             0.479         0.307  0.621   
hf:microsoft/rad-dino                            0.392         0.314  0.465   
timm:convnext_small.fb_in22k_ft_in1k             0.229         0.369  0.408   
xrv:densenet121-res224-chex                      0.424         0.303  0.518   
xrv:densenet121-res224-nih                       0.558         0.361  0.653   

                                      Pneumonia  Pulmonary edema  \
experiment                                                         
biomedclip                                0.306            0.544   
densenet121_imagenet                      0.343            0.776   
hf:microsoft/rad-dino                     0.338            0.563   
timm:convnext_small.fb_in22k_ft_in1k      0.307            0.414   
xrv:densenet121-res224-chex               0.635            0.523   
xrv:densenet121-res224-nih                0.561            0.742   

                                      mean_over_diseases  sd_over_seeds  
experiment                                                               
biomedclip                                         0.484            NaN  
densenet121_imagenet                               0.505            NaN  
hf:microsoft/rad-dino                              0.414            NaN  
timm:convnext_small.fb_in22k_ft_in1k               0.345            NaN  
xrv:densenet121-res224-chex                        0.480            NaN  
xrv:densenet121-res224-nih                         0.575            NaN  
```


## Original cell 43 (zero-based)

```text
VALIDATION (use for model choice)

```

```text
                        Pleural effusion  Cardiomegaly  Edema  Pneumonia  \
experiment                                                                 
A2_base_diag_img                   0.553         0.428  0.474      0.242   
A_base_img                         0.475         0.314  0.405      0.319   
B_base_img_tab                     0.499         0.314  0.376      0.306   
C_new_img                          0.455         0.314  0.476      0.307   
D_new_img_tab                      0.600         0.314  0.551      0.343   
E_new_img_tab_shuffled             0.521         0.314  0.454      0.203   
F_tab_nn                           0.193         0.314  0.364      0.222   

                        Pulmonary edema  mean_over_diseases  sd_over_seeds  
experiment                                                                  
A2_base_diag_img                  0.442               0.428            NaN  
A_base_img                        0.380               0.379            NaN  
B_base_img_tab                    0.412               0.381            NaN  
C_new_img                         0.500               0.410            NaN  
D_new_img_tab                     0.456               0.453            NaN  
E_new_img_tab_shuffled            0.532               0.405            NaN  
F_tab_nn                          0.469               0.312            NaN  
```

```text
TEST (report once)

```

```text
                        Pleural effusion  Cardiomegaly  Edema  Pneumonia  \
experiment                                                                 
A2_base_diag_img                   0.443         0.301  0.579      0.275   
A_base_img                         0.373         0.301  0.299      0.252   
B_base_img_tab                     0.284         0.301  0.429      0.275   
C_new_img                          0.391         0.301  0.443      0.456   
D_new_img_tab                      0.464         0.301  0.397      0.324   
E_new_img_tab_shuffled             0.434         0.301  0.507      0.405   
F_tab_nn                           0.345         0.301  0.301      0.196   

                        Pulmonary edema  mean_over_diseases  sd_over_seeds  
experiment                                                                  
A2_base_diag_img                  0.469               0.414            NaN  
A_base_img                        0.356               0.316            NaN  
B_base_img_tab                    0.601               0.378            NaN  
C_new_img                         0.454               0.409            NaN  
D_new_img_tab                     0.414               0.380            NaN  
E_new_img_tab_shuffled            0.528               0.435            NaN  
F_tab_nn                          0.306               0.290            NaN  
```

```text
A_base_img

```

```text
A2_base_diag_img

```

```text
E_new_img_tab_shuffled

```

```text
B_base_img_tab

```

```text
F_tab_nn

```

```text
C_new_img

```

```text
D_new_img_tab

```
