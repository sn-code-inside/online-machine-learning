from spotPython.data.csvdataset import CSVDataset as spotPythonCSVDataset
from spotPython.data.pkldataset import PKLDataset as spotPythonPKLDataset
import tkinter as tk
import customtkinter
import pprint
import os
import pickle
import numpy as np
import copy
from torch.utils.data import DataLoader
import pandas as pd

from spotPython.utils.init import fun_control_init, design_control_init, surrogate_control_init, optimizer_control_init
from spotgui.ctk.CTk import CTkApp, SelectOptionMenuFrame
from spotRiver.hyperdict.river_hyper_dict import RiverHyperDict
from spotPython.hyperdict.light_hyper_dict import LightHyperDict
from spotPython.hyperdict.sklearn_hyper_dict import SklearnHyperDict
from spotgui.tuner.spotRun import (
    save_spot_python_experiment,
    run_spot_python_experiment,
    get_n_total,
    get_fun_evals,
    get_lambda_min_max,
    get_oml_grace_period,
    get_weights,
    get_kriging_noise,
    get_scenario_dict,
)
from spotRiver.data.selector import get_river_dataset_from_name
from spotPython.utils.convert import map_to_True_False, set_dataset_target_type, check_type
from spotRiver.utils.data_conversion import split_df
from spotPython.hyperparameters.values import (
    add_core_model_to_fun_control,
    get_core_model_from_name,
    get_river_core_model_from_name,
    get_metric_sklearn,
    update_fun_control_with_hyper_num_cat_dicts,
)
from spotRiver.fun.hyperriver import HyperRiver
from spotPython.fun.hyperlight import HyperLight
from spotPython.fun.hypersklearn import HyperSklearn
from spotPython.utils.metrics import get_metric_sign
from spotPython.utils.scaler import TorchStandardScaler


class spotPythonApp(CTkApp):
    def __init__(self):
        super().__init__()
        self.title("spotPython GUI")
        self.logo_text = "    SPOTPython"

        # self.scenario = "river"
        # self.hyperdict = RiverHyperDict
        self.scenario = "sklearn"
        self.hyperdict = SklearnHyperDict

        self.task_name = "regression_task"
        self.scenario_dict = get_scenario_dict(scenario=self.scenario)
        pprint.pprint(self.scenario_dict)
        # ---------------------------------------------------------------------- #
        # ---------------- 0 Sidebar Frame --------------------------------------- #
        # ---------------------------------------------------------------------- #
        # create sidebar frame with widgets in row 0 and column 0
        self.make_sidebar_frame()
        # ---------------------------------------------------------------------- #
        # ----------------- 1 Experiment_Main Frame ------------------------------ #
        # ---------------------------------------------------------------------- #
        # create experiment main frame with widgets in row 0 and column 1
        #
        self.make_experiment_frame()
        # ---------------------------------------------------------------------- #
        # ------------------ 2 Hyperparameter Main Frame ----------------------- #
        # ---------------------------------------------------------------------- #
        # create hyperparameter main frame with widgets in row 0 and column 2
        self.make_hyperparameter_frame()
        #
        # ---------------------------------------------------------------------- #
        # ----------------- 3 Execution_Main Frame ----------------------------- #
        # ---------------------------------------------------------------------- #
        # create execution_main frame with widgets in row 0 and column 4
        self.make_execution_frame()

        # ---------------------------------------------------------------------- #
        # ----------------- 4 Analysis_Main Frame ------------------------------ #
        # ---------------------------------------------------------------------- #
        # create analysis_main frame with widgets in row 0 and column 3
        self.make_analysis_frame()
        #

    def get_test_size(self):
        """Sets the test_size attribute to the value of the test_size_var attribute.
        Sets the following attribute:
        - test_size (int or float): test size for the data set.
        """
        test_size = self.test_size_var.get()
        # if test_size contains a point, it is a float, otherwise an integer:
        if "." in test_size:
            self.test_size = float(test_size)
        else:
            self.test_size = int(test_size)

    def get_target_type(self):
        """Sets the target_type attribute to the type of the target column in the data set.
        Sets the following attribute:
        - target_type (str): type of the target column in the data set.
        """
        if hasattr(self, "data_set"):
            val = copy.deepcopy(self.data_set.iloc[0, -1])
            self.target_type = check_type(val)
            print(f"Target type: {self.target_type}")
        else:
            print("No data set available. Setting target_type to None.")
            self.target_type = None

    def print_data(self):
        self.set_global_attributes()
        if self.scenario == "river":
            self.set_river_attributes()
            self.get_river_data()
            self.get_csv_data()
            self.get_pkl_data()
            self.prepare_data()
            self.print_cvs_pkl_data()
        elif self.scenario == "lightning":
            self.set_lightning_attributes()
            self.get_tkl_data()
            self.get_tkl_data_dimensions()
            self.print_lightning_data()
        elif self.scenario == "sklearn":
            self.set_sklearn_attributes()
            self.get_csv_data()
            self.get_pkl_data()
            self.prepare_data()
            self.print_cvs_pkl_data()

    def prepare_data(self):
        """Splits the data-set into training and test sets.
        Applies to sklearn and river scenarios.
        Sets the following attributes:
        - train: training data set
        - test: test data set
        - n_samples: number of samples in the data set
        """
        self.train, self.test, self.n_samples = split_df(
            dataset=self.data_set,
            test_size=self.test_size,
            target_type=self.target_type,
            seed=self.seed,
            shuffle=self.shuffle,
            stratify=None,
        )

    def set_lightning_attributes(self):
        """Set the attributes for the lightning scenario.
        These include:
        - db_dict_name
        - scaler
        - scaler_name
        - train
        - test
        - n_samples
        - target_type
        - weights
        - weights_entry
        - horizon
        - oml_grace_period
        - fun
        """
        self.db_dict_name = None  # experimental, do not use
        self.scaler = TorchStandardScaler()
        self.scaler_name = "TorchStandardScaler"
        self.train = None
        self.test = None
        self.n_samples = None
        self.target_type = None
        self.weights = 1.0
        self.weights_entry = None
        self.horizon = None
        self.oml_grace_period = None
        self.fun = HyperLight(log_level=self.log_level).fun

    def get_river_data(self):
        """Sets the data_set attribute to the selected data set.
        If the data set is a river data set, it is loaded as a river data set.
        Sets the following attributes:
        - data_set: river data set
        - n_samples: number of samples in the data set
        """
        self.data_set, self.n_samples = get_river_dataset_from_name(
            data_set_name=self.data_set_name,
            n_total=get_n_total(self.n_total_var.get()),
            river_datasets=self.scenario_dict[self.task_name]["datasets"],
        )
        if self.data_set is not None:
            print(f"Loading river data set: {self.data_set_name}")
            self.get_target_type()
            self.data_set = set_dataset_target_type(dataset=self.data_set, target="y")

    def get_csv_data(self):
        """Sets the data_set attribute to the selected data set.
        If the data set is a CSV file, it is loaded as a spotPythonCSVDataset.
        """
        if self.data_set_name.endswith(".csv"):
            print(f"Loading CSV data set: {self.data_set_name}")
            # Load the CSV file from the userData directory as a pandas DataFrame
            file_path = os.path.join("./userData/", self.data_set_name)
            self.data_set = pd.read_csv(file_path)
            self.n_samples = self.data_set.shape[0]
            self.get_target_type()
            self.data_set = set_dataset_target_type(dataset=self.data_set, target="y")
        else:
            print("No CSV data set loaded.")

    def get_pkl_data(self):
        """Sets the data_set attribute to the selected data set.
        If the data set is a PKL file, it is loaded as a spotPythonPKLDataset."""
        if self.data_set_name.endswith(".pkl"):
            self.data_set = spotPythonPKLDataset(filename=self.data_set_name, directory="./userData/")
            self.n_samples = self.data_set.shape[0]
            self.get_target_type()
            self.data_set = set_dataset_target_type(dataset=self.data_set, target="y")
        else:
            print("No PKL data set loaded.")

    def get_tkl_data(self):
        """Sets the data_set attribute to the selected data set.
        If the data set is a TKL (Tensor pKL) file,
        it is loaded as a pickle file."""
        if self.data_set_name.endswith(".tkl"):
            filename = os.path.join("./userData/", self.data_set_name)
            with open(filename, "rb") as f:
                self.data_set = pickle.load(f)
            self.n_samples = len(self.data_set)
            self.target_type = None
        else:
            print("No TKL data set loaded.")

    def get_tkl_data_dimensions(self):
        """Get dimensions of the data sets.
        Applicable to all data sets.
        Sets the following attributes:
        - n_samples: number of samples in the data set
        - n_cols: number of columns in the data set
        """
        if self.data_set is not None:
            self.n_samples = len(self.data_set)
            print(f"Number of samples: {self.n_samples}")
            if hasattr(self.data_set, "__ncols__"):
                self.n_cols = self.data_set.__ncols__()
            # check if data_set has the __getitem__ method
            elif hasattr(self.data_set, "__getitem__"):
                self.n_cols = self.data_set.__getitem__(0)[0].shape[0]
            else:
                self.n_cols = None
            print(f"Data set number of columns: {self.n_cols}")

    def print_lightning_data(self):
        # check if self.data_set is available
        if hasattr(self, "data_set"):
            # Set batch size for DataLoader
            batch_size = 5
            # Create DataLoader
            dataloader = DataLoader(self.data_set, batch_size=batch_size, shuffle=False)
            # Iterate over the data in the DataLoader
            for batch in dataloader:
                inputs, targets = batch
                print(f"Batch Size: {inputs.size(0)}")
                print(f"Inputs Shape: {inputs.shape}")
                print(f"Targets Shape: {targets.shape}")
                print("---------------")
                print(f"Inputs: {inputs}")
                print(f"Targets: {targets}")
                break
        else:
            print("No lightning data_set available. Please select a data_set.")

    def print_cvs_pkl_data(self):
        print("\nDataset in print_data():")
        print(f"n_samples: {self.n_samples}")
        print(f"target_type: {self.target_type}")
        print(f"test_size: {self.test_size}")
        print(f"shuffle: {self.shuffle}")
        print(f"{self.data_set.describe(include='all')}")
        print(f"Header of the dataset:\n {self.data_set.head()}")

    def set_global_attributes(self):
        """Set the global attributes for the spotPythonApp.
        These include:
        - log_level
        - verbosity
        - tolerance_x
        - ocba_delta
        - repeats
        - fun_repeats
        - target_column
        - n_theta
        - db_dict_name
        - eval
        - n_cols
        - seed
        - test_size
        - shuffle
        - n_total
        - max_time
        - fun_evals
        - init_size
        - noise
        - lbd_min
        - lbd_max
        - kriging_noise
        - max_surrogate_points
        - TENSORBOARD_CLEAN
        - tensorboard_start
        - tensorboard_stop
        - PREFIX
        - data_set_name
        - prep_model_name
        - prepmodel
        - scaler_name
        - scaler
        - metric_sklearn_name
        - metric_sklearn
        """
        # Entries NOT handled by the GUI:
        self.log_level = 50
        self.verbosity = 1
        self.tolerance_x = np.sqrt(np.spacing(1))
        self.ocba_delta = 0
        self.repeats = 1
        self.fun_repeats = 1
        self.target_column = "y"
        self.n_theta = 2
        self.db_dict_name = None  # experimental, do not use
        self.eval = None
        # Entries handled by the GUI:
        self.n_cols = None
        self.seed = int(self.seed_var.get())
        self.get_test_size()
        self.shuffle = map_to_True_False(self.shuffle_var.get())
        self.n_total = get_n_total(self.n_total_var.get())
        self.max_time = float(self.max_time_var.get())
        self.fun_evals = get_fun_evals(self.fun_evals_var.get())
        self.init_size = int(self.init_size_var.get())
        self.noise = map_to_True_False(self.noise_var.get())
        self.lbd_min, self.lbd_max = get_lambda_min_max(self.lambda_min_max_var.get())
        self.kriging_noise = get_kriging_noise(self.lbd_min, self.lbd_max)
        self.max_surrogate_points = int(self.max_sp_var.get())
        self.TENSORBOARD_CLEAN = map_to_True_False(self.tb_clean_var.get())
        self.tensorboard_start = map_to_True_False(self.tb_start_var.get())
        # if TENSOBOARD_START is True, set SUMMARY_WRITER to True, because
        # the tensorboard needs a summary writer
        self.SUMMARY_WRITER = map_to_True_False(self.tb_start_var.get())
        self.tensorboard_stop = map_to_True_False(self.tb_stop_var.get())
        self.PREFIX = self.experiment_name_entry.get()
        self.data_set_name = self.select_data_frame.get_selected_optionmenu_item()
        # if self has the attribute select_prep_model_frame, get the selected
        # optionmenu item
        if hasattr(self, "select_prep_model_frame"):
            self.prep_model_name = self.select_prep_model_frame.get_selected_optionmenu_item()
            self.prepmodel = self.check_user_prep_model(prep_model_name=self.prep_model_name)
        else:
            self.prep_model_name = None
            self.prepmodel = None

        # if self has the attribute select_scaler_frame, get the selected optionmenu item
        if hasattr(self, "select_scaler_frame"):
            self.scaler_name = self.select_scaler_frame.get_selected_optionmenu_item()
            self.scaler = self.check_user_prep_model(prep_model_name=self.scaler_name)
        else:
            self.scaler_name = None
            self.scaler = None
        if hasattr(self, "select_metric_sklearn_levels_frame"):
            self.metric_sklearn_name = self.select_metric_sklearn_levels_frame.get_selected_optionmenu_item()
            self.metric_sklearn = get_metric_sklearn(self.metric_sklearn_name)
        else:
            self.metric_sklearn_name = None
            self.metric_sklearn = None

    def set_sklearn_attributes(self):
        """Set the attributes for the sklearn scenario.
        These include:
        - eval (str): "evaluate_hold_out"
        """
        self.eval = "evaluate_hold_out"
        self.data_set = None
        self.db_dict_name = None  # experimental, do not use
        self.weights = get_metric_sign(self.metric_sklearn_name)
        self.weights_entry = None
        self.horizon = None
        self.oml_grace_period = None
        self.fun = HyperSklearn(log_level=self.log_level).fun_sklearn

    def set_river_attributes(self):
        """Set the attributes for the river scenario.
        These include:
        - data_set (None): data set
        - db_dict_name (None): dictionary name for the database
        - train (DataFrame): training data set
        - test (DataFrame): test data set
        - n_samples (int): number of samples in the data set
        - target_type (str): type of the target column
        - weights_entry (str): GUI entry for the weights, this is a string
        - weights (np.ndarray): weights for the metric
        - horizon (int): horizon for the OML
        - oml_grace_period (int): grace period for the OML
        - fun (function): function to be used for the experiment
        """
        self.data_set = None
        self.weights_entry = self.weights_var.get()
        self.weights = get_weights(
            self.select_metric_sklearn_levels_frame.get_selected_optionmenu_item(), self.weights_var.get()
        )
        self.horizon = int(self.horizon_var.get())
        self.oml_grace_period = get_oml_grace_period(self.oml_grace_period_var.get())
        self.fun = HyperRiver(log_level=self.log_level).fun_oml_horizon

    def prepare_experiment(self):
        self.set_global_attributes()
        task_name = self.task_frame.get_selected_optionmenu_item()
        core_model_name = self.select_core_model_frame.get_selected_optionmenu_item()

        # ----------------- Scenario specific ----------------- #
        if self.scenario == "river":
            self.set_river_attributes()
            self.get_river_data()
            self.get_csv_data()
            self.get_pkl_data()
            self.prepare_data()
        elif self.scenario == "lightning":
            self.set_lightning_attributes()
            self.get_tkl_data()
            self.get_tkl_data_dimensions()
            self.print_lightning_data()
        elif self.scenario == "sklearn":
            self.set_sklearn_attributes()
            self.get_csv_data()
            self.get_pkl_data()
            self.prepare_data()

        # ----------------- fun_control ----------------- #
        self.fun_control = fun_control_init(
            _L_in=self.n_cols,  # number of input features
            _L_out=1,
            _torchmetric=None,
            PREFIX=self.PREFIX,
            TENSORBOARD_CLEAN=self.TENSORBOARD_CLEAN,
            SUMMARY_WRITER=self.SUMMARY_WRITER,
            core_model_name=core_model_name,
            data_set_name=self.data_set_name,
            data_set=self.data_set,
            db_dict_name=self.db_dict_name,
            eval=self.eval,
            fun_evals=self.fun_evals,
            fun_repeats=self.fun_repeats,
            horizon=self.horizon,
            max_surrogate_points=self.max_surrogate_points,
            max_time=self.max_time,
            metric_sklearn=self.metric_sklearn,
            metric_sklearn_name=self.metric_sklearn_name,
            noise=self.noise,
            n_samples=self.n_samples,
            n_total=self.n_total,
            ocba_delta=self.ocba_delta,
            oml_grace_period=self.oml_grace_period,
            prep_model=self.prepmodel,
            prep_model_name=self.prep_model_name,
            progress_file=self.progress_file,
            scaler=self.scaler,
            scaler_name=self.scaler_name,
            scenario=self.scenario,
            seed=self.seed,
            shuffle=self.shuffle,
            task=task_name,
            target_column=self.target_column,
            target_type=self.target_type,
            tensorboard_start=self.tensorboard_start,
            tensorboard_stop=self.tensorboard_stop,
            test=self.test,
            test_size=self.test_size,
            train=self.train,
            tolerance_x=self.tolerance_x,
            verbosity=self.verbosity,
            weights=self.weights,
            weights_entry=self.weights_entry,
            log_level=self.log_level,
        )
        if self.scenario == "river":
            coremodel, core_model_instance = get_river_core_model_from_name(core_model_name)
        else:
            coremodel, core_model_instance = get_core_model_from_name(core_model_name)
        add_core_model_to_fun_control(
            core_model=core_model_instance,
            fun_control=self.fun_control,
            hyper_dict=self.hyperdict,
            filename=None,
        )
        dict = self.hyperdict().hyper_dict[coremodel]
        num_dict = self.num_hp_frame.get_num_item()
        cat_dict = self.cat_hp_frame.get_cat_item()
        update_fun_control_with_hyper_num_cat_dicts(self.fun_control, num_dict, cat_dict, dict)

        # ----------------- design_control ----------------- #
        self.design_control = design_control_init(
            init_size=self.init_size,
            repeats=self.repeats,
        )

        # ----------------- surrogate_control ----------------- #
        self.surrogate_control = surrogate_control_init(
            # If lambda is set to 0, no noise will be used in the surrogate
            # Otherwise use noise in the surrogate:
            noise=self.kriging_noise,
            n_theta=self.n_theta,
            min_Lambda=self.lbd_min,
            max_Lambda=self.lbd_max,
            log_level=self.log_level,
        )

        # ----------------- optimizer_control ----------------- #
        self.optimizer_control = optimizer_control_init()

    def save_experiment(self):
        self.prepare_experiment()
        save_spot_python_experiment(
            fun_control=self.fun_control,
            design_control=self.design_control,
            surrogate_control=self.surrogate_control,
            optimizer_control=self.optimizer_control,
            fun=self.fun,
        )
        print("\nExperiment saved.")
        pprint.pprint(self.fun_control)

    def run_experiment(self):
        self.prepare_experiment()
        run_spot_python_experiment(
            fun_control=self.fun_control,
            design_control=self.design_control,
            surrogate_control=self.surrogate_control,
            optimizer_control=self.optimizer_control,
            fun=self.fun,
            tensorboard_start=self.tensorboard_start,
            tensorboard_stop=self.tensorboard_stop,
        )
        print("\nExperiment finished.")


# TODO:
# Check the handling of l1/l2 in LogisticRegression. A note (from the River documentation):
# > For now, only one type of penalty can be used. The joint use of L1 and L2 is not explicitly supported.
# Therefore, we set l1 bounds to 0.0:
# modify_hyper_parameter_bounds(fun_control, "l1", bounds=[0.0, 0.0])
# set_control_hyperparameter_value(fun_control, "l1", [0.0, 0.0])
# modify_hyper_parameter_levels(fun_control, "optimizer", ["SGD"])

if __name__ == "__main__":
    customtkinter.set_appearance_mode("light")
    customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"
    app = spotPythonApp()
    app.mainloop()
