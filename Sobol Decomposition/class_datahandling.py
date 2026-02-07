
from pathlib import Path
import pandas as pd
import numpy as np
import xarray as xr
import yaml

class data_handler(object):
    def __init__(self):
        self.current_dir = Path.cwd()
        with open(self.current_dir / 'input.yml') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        self.input_csv_path = Path(config['input_csv_path'])
        self.save = config['save']
        (self.current_dir / "Data/Handling_files").mkdir(parents=True, exist_ok=True)
        self.output_path = self.current_dir / "Data/Handling_files/XR_for_decomposition.nc"

    def load_and_prepare_data(self):
        print("- Loading pre-processed WIDE CSV:", self.input_csv_path)
        df = pd.read_csv(self.input_csv_path)

        year_cols = [col for col in df.columns if col.isnumeric()]
        
        print("- Calculating new analysis dimensions and melting to LONG format.")
        df[year_cols] = df[year_cols].apply(pd.to_numeric, errors='coerce')
        df['2020-2030rate'] = (df['2030'] - df['2020']) / 10.0
        df['2030-2040rate'] = (df['2040'] - df['2030']) / 10.0
        df['2020-2100cumulative'] = df[year_cols].sum(axis=1)
        id_vars = ['Model', 'Scenario', 'Variable', 'PC_m']
        value_vars = year_cols + ['2020-2030rate', '2030-2040rate', '2020-2100cumulative']
        df_long = df.melt(id_vars=id_vars, value_vars=value_vars, var_name='Analysis_Dimension', value_name='Value')
        df_long['Analysis_Dimension'] = df_long['Analysis_Dimension'].astype(str)
        df_long['ModelScenario'] = df_long['Model'] + '|' + df_long['Scenario']

        print("- Building final xarray object with corrected coordinate structure.")
        

        scenario_metadata = df_long[['ModelScenario', 'Model', 'PC_m']].drop_duplicates().set_index('ModelScenario')
        
        main_data_df = df_long.set_index(['Variable', 'Analysis_Dimension', 'ModelScenario'])['Value']
        
        final_xr = main_data_df.to_xarray()
        
        final_xr.coords['Model'] = scenario_metadata['Model'].to_xarray()
        final_xr.coords['PC_m'] = scenario_metadata['PC_m'].to_xarray()
        
        self.XR_final_for_decomposition = final_xr
        
        print("- Data structure is now definitively correct. Ready for decomposition.")

    def savings(self):
        if self.save == 'yes':
            print(f"- Saving final data structure to {self.output_path}")
            self.XR_final_for_decomposition.to_netcdf(self.output_path)
