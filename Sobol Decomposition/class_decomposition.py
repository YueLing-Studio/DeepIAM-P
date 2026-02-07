from pathlib import Path
import numpy as np
from tqdm import tqdm
import xarray as xr
import yaml
from scipy.stats import qmc 

class decomposition(object):
    def __init__(self):

        self.current_dir = Path.cwd()
        with open(self.current_dir / 'input.yml') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        self.save = config['save']
        self.use_resampling = config['use_resampling']
        self.resampling = int(config['resampling'])
        self.sample_size_per_ms = int(config['sample_size_per_ms'])
        
        (self.current_dir / "Data/Output_files").mkdir(parents=True, exist_ok=True)
        
        print("- Loading data prepared for decomposition.")
        input_data_path = self.current_dir / "Data/Handling_files/XR_for_decomposition.nc"
        self.XRdata = xr.open_dataarray(input_data_path)
        self.analysis_dimensions = self.XRdata.Analysis_Dimension.values
        self.variables = self.XRdata.Variable.values

        print("- Creating robust lookup tables from pristine data source.")
        scenarios = self.XRdata.ModelScenario.values
        models = self.XRdata.Model.values
        pcms = self.XRdata.PC_m.values
        self.model_lookup = dict(zip(scenarios, models))
        self.pcm_lookup = dict(zip(scenarios, pcms))
        print("- Lookup tables created successfully.")


    def sampling_and_decomposing(self, printy='on'):
        if printy == 'on': print("- Starting variance decomposition process.")
        
        n_iterations = self.resampling if self.use_resampling == 'yes' else 1
        if printy == 'on':
            print(f"- Resampling is {'ON' if self.use_resampling == 'yes' else 'OFF'}. "
                  f"Running {n_iterations} iteration(s).")
        
        def generate_samples_for_var(self, data_for_var):
            sobol_eng = qmc.Sobol(d=1, scramble=True)

            values_nn = data_for_var.values
            valid_scenarios_keys = data_for_var.ModelScenario.values

            mods = np.array([self.model_lookup[s] for s in valid_scenarios_keys])
            pc_m_labels = np.array([self.pcm_lookup[s] for s in valid_scenarios_keys])

            std_dev = np.nanstd(values_nn)
            mean_val = np.nanmean(values_nn)
            values = (values_nn - mean_val) / std_dev if std_dev != 0 else values_nn - mean_val
            
            unimods = np.unique(mods)
            uni_pc_m = np.unique(pc_m_labels)
            
            ss = self.sample_size_per_ms
            analysis_dim_len = len(data_for_var.Analysis_Dimension)
            indices = np.zeros(shape=(7, n_iterations, analysis_dim_len))

            for n_i in range(n_iterations):
                sample_pairs = []
                for m in unimods:
                    for pc in uni_pc_m:
                        if np.any((mods == m) & (pc_m_labels == pc)):
                            sample_pairs.extend([(m, pc)] * ss)
                if not sample_pairs: 
                    continue
                
                sample1, sample2 = np.array(sample_pairs), np.array(sample_pairs)
                np.random.shuffle(sample1); np.random.shuffle(sample2)
                
                num_samples = len(sample1)
                M1, M1nn, M2, Nm, Nc, Nmc = (
                    np.zeros((num_samples, analysis_dim_len)) for _ in range(6)
                )

                for m in unimods:
                    for pc in uni_pc_m:
                        wh_data = np.where((mods == m) & (pc_m_labels == pc))[0]
                        if len(wh_data) == 0:
                            continue

                        wh1 = np.where((sample1[:,0]==m)&(sample1[:,1].astype(int)==pc))[0]
                        wh2 = np.where((sample2[:,0]==m)&(sample2[:,1].astype(int)==pc))[0]
                        wh_m = np.where((sample1[:,0]==m)&(sample2[:,1].astype(int)==pc))[0]
                        wh_c = np.where((sample2[:,0]==m)&(sample1[:,1].astype(int)==pc))[0]

                        if len(wh1) > 0:
                            u = sobol_eng.random(n=len(wh1)).flatten()
                            idx_in_wh = (u * len(wh_data)).astype(int)
                            random_indices_A = wh_data[idx_in_wh]
                            M1[wh1] = values[:, random_indices_A].T
                            M1nn[wh1] = values_nn[:, random_indices_A].T

                            u = sobol_eng.random(n=len(wh1)).flatten() 
                            idx_in_wh = (u * len(wh_data)).astype(int)
                            random_indices_C = wh_data[idx_in_wh]
                            Nmc[wh1] = values[:, random_indices_C].T

                        if len(wh2) > 0:
                            u = sobol_eng.random(n=len(wh2)).flatten()  
                            idx_in_wh = (u * len(wh_data)).astype(int)
                            random_indices_B = wh_data[idx_in_wh]
                            M2[wh2] = values[:, random_indices_B].T

                        if len(wh_m) > 0:
                            u = sobol_eng.random(n=len(wh_m)).flatten()  
                            idx_in_wh = (u * len(wh_data)).astype(int)
                            random_indices_mix_m = wh_data[idx_in_wh]
                            Nm[wh_m] = values[:, random_indices_mix_m].T

                        if len(wh_c) > 0:
                            u = sobol_eng.random(n=len(wh_c)).flatten()  
                            idx_in_wh = (u * len(wh_data)).astype(int)
                            random_indices_mix_c = wh_data[idx_in_wh]
                            Nc[wh_c] = values[:, random_indices_mix_c].T

                vtot_norm = np.nanvar(M1, axis=0); vtot_norm[vtot_norm==0] = 1
                s_m   = (np.nanmean(M1*Nm, axis=0)   - np.nanmean(M1,axis=0)*np.nanmean(M2,axis=0)) / vtot_norm
                s_c   = (np.nanmean(M1*Nc, axis=0)   - np.nanmean(M1,axis=0)*np.nanmean(M2,axis=0)) / vtot_norm
                comb  = (np.nanmean(M1*Nmc,axis=0)   - np.nanmean(M1,axis=0)*np.nanmean(M2,axis=0)) / vtot_norm
                s_mc  = comb - s_m - s_c
                s_z   = 1 - (s_m + s_c + s_mc)
                
                mean_M1nn = np.nanmean(M1nn, axis=0)
                std_M1nn  = np.nanstd(M1nn,  axis=0)
                vtot      = np.nanvar(M1nn, axis=0)
                cv        = np.divide(std_M1nn, mean_M1nn,
                                      out=np.zeros_like(mean_M1nn), where=mean_M1nn!=0)
                
                indices[:, n_i, :] = [vtot, s_m, s_c, s_mc, s_z, vtot_norm, cv]
            
            return np.nanmean(indices, axis=1)

        self.variances = np.zeros((len(self.variables), 7, len(self.analysis_dimensions)))
        
        iterator = tqdm(self.variables, desc="Decomposing Variables") if printy=='on' else self.variables
        for v_i, var in enumerate(iterator):
            data_for_var = self.XRdata.sel(Variable=var).dropna(dim='ModelScenario')
            if data_for_var.shape[1] == 0:
                self.variances[v_i] = np.zeros((7, len(self.analysis_dimensions)))
                continue
            
            vtot, s_m, s_c, s_mc, s_z, vtot_norm, cv = generate_samples_for_var(self, data_for_var)
            self.variances[v_i] = [vtot, s_m, s_c, s_z, s_mc, vtot_norm, cv]
        
        self.XRvar = xr.Dataset({
            "Var_total":      (("Variable","Analysis_Dimension"), self.variances[:,0]),
            "S_m":            (("Variable","Analysis_Dimension"), self.variances[:,1]),
            "S_c":            (("Variable","Analysis_Dimension"), self.variances[:,2]),
            "S_z":            (("Variable","Analysis_Dimension"), self.variances[:,3]),
            "S_mc":           (("Variable","Analysis_Dimension"), self.variances[:,4]),
            "Var_total_norm": (("Variable","Analysis_Dimension"), self.variances[:,5]),
            "CoefVar":        (("Variable","Analysis_Dimension"), self.variances[:,6])
        }, coords={
            "Variable": self.variables,
            "Analysis_Dimension": self.analysis_dimensions
        })

    def savings(self):
        if self.save == 'yes':
            output_path = self.current_dir / "Data/Output_files/Variances_final.nc"
            print(f"- Saving final variance decomposition results to {output_path}")
            self.XRvar.to_netcdf(output_path)
