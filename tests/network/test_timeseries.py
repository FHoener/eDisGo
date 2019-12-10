import os
import pandas as pd
from pandas.util.testing import assert_series_equal, assert_frame_equal
from math import tan, acos
import pytest
import shutil
import numpy as np

from edisgo.network.topology import Topology
from edisgo.tools.config import Config
from edisgo.network.timeseries import TimeSeriesControl, TimeSeries, \
    import_load_timeseries
from edisgo.io import ding0_import


class TestTimeSeriesControl:

    @classmethod
    def setup_class(self):
        """Setup default values"""
        parent_dirname = os.path.dirname(os.path.dirname(__file__))
        test_network_directory = os.path.join(
            parent_dirname, 'ding0_test_network')
        self.topology = Topology()
        self.timeseries = TimeSeries()
        self.config = Config()
        ding0_import.import_ding0_grid(test_network_directory, self)

    def test_to_csv(self):
        cur_dir = os.getcwd()
        TimeSeriesControl(edisgo_obj=self, mode='worst-case')
        self.timeseries.to_csv(cur_dir)
        #create edisgo obj to compare
        parent_dirname = os.path.dirname(os.path.dirname(__file__))
        test_network_directory = os.path.join(
            parent_dirname, 'ding0_test_network')
        edisgo = pd.DataFrame()
        edisgo.topology = Topology()
        edisgo.timeseries = TimeSeries()
        edisgo.config = Config()
        ding0_import.import_ding0_grid(test_network_directory, edisgo)
        TimeSeriesControl(
            edisgo, mode='manual',
            timeindex=pd.read_csv(
                os.path.join(cur_dir, 'timeseries', 'loads_active_power.csv'),
                index_col=0).index,
            loads_active_power=pd.read_csv(
                os.path.join(cur_dir, 'timeseries', 'loads_active_power.csv'),
                index_col=0),
            loads_reactive_power=pd.read_csv(
                os.path.join(cur_dir, 'timeseries',
                             'loads_reactive_power.csv'), index_col=0),
            generators_active_power=pd.read_csv(
                os.path.join(cur_dir, 'timeseries',
                             'generators_active_power.csv'), index_col=0),
            generators_reactive_power=pd.read_csv(
                os.path.join(cur_dir, 'timeseries',
                             'generators_reactive_power.csv'), index_col=0),
            storage_units_active_power=pd.read_csv(
                os.path.join(cur_dir, 'timeseries',
                             'storage_units_active_power.csv'), index_col=0),
            storage_units_reactive_power=pd.read_csv(
                os.path.join(cur_dir, 'timeseries',
                             'storage_units_reactive_power.csv'), index_col=0)
        )
        # check if timeseries are the same
        assert np.isclose(self.timeseries.loads_active_power,
                          edisgo.timeseries.loads_active_power).all()
        assert np.isclose(self.timeseries.loads_reactive_power,
                          edisgo.timeseries.loads_reactive_power).all()
        assert np.isclose(self.timeseries.generators_active_power,
                          edisgo.timeseries.generators_active_power).all()
        assert np.isclose(self.timeseries.generators_reactive_power,
                          edisgo.timeseries.generators_reactive_power).all()
        assert np.isclose(self.timeseries.storage_units_active_power,
                          edisgo.timeseries.storage_units_active_power).all()
        assert np.isclose(self.timeseries.storage_units_reactive_power,
                          edisgo.timeseries.storage_units_reactive_power).all()
        # delete folder
        # Todo: check files before rmtree?
        shutil.rmtree(os.path.join(cur_dir, 'timeseries'), ignore_errors=True)
        self.timeseries = TimeSeries()

    def test_timeseries_imported(self):
        # test storage ts
        self.topology.add_storage_unit('Test_storage_1', 'Bus_MVStation_1',
                                       0.3)
        self.topology.add_storage_unit('Test_storage_2',
                                       'Bus_GeneratorFluctuating_2', 0.45)
        self.topology.add_storage_unit('Test_storage_3',
                                       'Bus_BranchTee_LVGrid_1_10', 0.05)

        timeindex = pd.date_range('1/1/2011', periods=8760, freq='H')
        ts_gen_dispatchable = pd.DataFrame({'Generator_1': [0.775]*8760},
                                           index=timeindex)
        # test error raising in case of missing ts for dispatchable gens
        msg = \
            'Your input for "timeseries_generation_dispatchable" is not valid.'
        with pytest.raises(ValueError, match=msg):
            TimeSeriesControl(edisgo_obj=self,
                              timeseries_generation_fluctuating='oedb')
        # test error raising in case of missing ts for loads
        msg = 'Your input for "timeseries_load" is not valid.'
        with pytest.raises(ValueError, match=msg):
            TimeSeriesControl(edisgo_obj=self,
                  timeseries_generation_fluctuating='oedb',
                  timeseries_generation_dispatchable=ts_gen_dispatchable)

        msg = "No timeseries for storage units provided."
        with pytest.raises(ValueError, match=msg):
            TimeSeriesControl(edisgo_obj=self,
                              timeseries_generation_fluctuating='oedb',
                              timeseries_generation_dispatchable=ts_gen_dispatchable,
                              timeseries_load='demandlib')

        msg = "Columns or indices of inserted storage timeseries do not match " \
              "topology and timeindex."
        with pytest.raises(ValueError, match=msg):
            TimeSeriesControl(edisgo_obj=self,
                              timeseries_generation_fluctuating='oedb',
                              timeseries_generation_dispatchable=ts_gen_dispatchable,
                              timeseries_load='demandlib',
                              timeseries_storage_units=pd.DataFrame())

        storage_ts = pd.concat([self.topology.storage_units_df.p_nom]*8760,
                               axis=1, keys=timeindex).T
        TimeSeriesControl(edisgo_obj=self,
                          timeseries_generation_fluctuating='oedb',
                          timeseries_generation_dispatchable=ts_gen_dispatchable,
                          timeseries_load='demandlib',
                          timeseries_storage_units=storage_ts)

        #Todo: test with inserted reactive generation and/or reactive load

        # remove storages
        self.topology.remove_storage('StorageUnit_MVGrid_1_Test_storage_1')
        self.topology.remove_storage('StorageUnit_MVGrid_1_Test_storage_2')
        self.topology.remove_storage('StorageUnit_LVGrid_1_Test_storage_3')

    def test_import_load_timeseries(self):
        with pytest.raises(NotImplementedError):
            import_load_timeseries(self.config, '')
        timeindex = pd.date_range('1/1/2018', periods=8760, freq='H')
        load = import_load_timeseries(self.config, 'demandlib',
                                      timeindex[0].year)
        assert (load.columns == ['retail', 'residential',
                                 'agricultural', 'industrial']).all()
        assert load.loc[timeindex[453], 'retail'] == 8.335076810751597e-05
        assert load.loc[timeindex[13], 'residential'] == 0.00017315167492271323
        assert load.loc[timeindex[6328], 'agricultural'] == \
               0.00010134645909959844
        assert load.loc[timeindex[4325], 'industrial'] == 9.91768322919766e-05

    def test_worst_case(self):
        """Test creation of worst case time series"""
        # test storage ts
        self.topology.add_storage_unit('Test_storage_1', 'Bus_MVStation_1',
                                       0.3)
        self.topology.add_storage_unit('Test_storage_2',
                                       'Bus_GeneratorFluctuating_2', 0.45)
        self.topology.add_storage_unit('Test_storage_3',
                                       'Bus_BranchTee_LVGrid_1_10', 0.05)

        ts_control = TimeSeriesControl(edisgo_obj=self, mode='worst-case')

        # check type
        assert isinstance(
            self.timeseries.generators_active_power, pd.DataFrame)
        assert isinstance(
            self.timeseries.generators_reactive_power, pd.DataFrame)
        assert isinstance(
            self.timeseries.loads_active_power, pd.DataFrame)
        assert isinstance(
            self.timeseries.loads_reactive_power, pd.DataFrame)
        assert isinstance(
            self.timeseries.storage_units_active_power, pd.DataFrame)
        assert isinstance(
            self.timeseries.storage_units_reactive_power, pd.DataFrame)

        # check shape
        number_of_timesteps = len(self.timeseries.timeindex)
        number_of_cols = len(self.topology.generators_df.index)
        assert self.timeseries.generators_active_power.shape == (
            number_of_timesteps, number_of_cols)
        assert self.timeseries.generators_reactive_power.shape == (
            number_of_timesteps, number_of_cols)
        number_of_cols = len(self.topology.loads_df.index)
        assert self.timeseries.loads_active_power.shape == (
            number_of_timesteps, number_of_cols)
        assert self.timeseries.loads_reactive_power.shape == (
            number_of_timesteps, number_of_cols)
        number_of_cols = len(self.topology.storage_units_df.index)
        assert self.timeseries.storage_units_active_power.shape == (
            number_of_timesteps, number_of_cols)
        assert self.timeseries.storage_units_reactive_power.shape == (
            number_of_timesteps, number_of_cols)

        # value
        gen = 'Generator_1'  # gas, mv
        exp = pd.Series(data=[1 * 0.775, 0 * 0.775], name=gen,
                        index=self.timeseries.timeindex)
        assert_series_equal(
            self.timeseries.generators_active_power.loc[:, gen], exp)
        pf = -tan(acos(0.9))
        assert_series_equal(
            self.timeseries.generators_reactive_power.loc[:, gen],
            exp * pf)

        gen = 'GeneratorFluctuating_2'  # wind, mv
        exp = pd.Series(data=[1 * 2.3, 0 * 2.3], name=gen,
                        index=self.timeseries.timeindex)
        assert_series_equal(
            self.timeseries.generators_active_power.loc[:, gen], exp)
        pf = -tan(acos(0.9))
        assert_series_equal(
            self.timeseries.generators_reactive_power.loc[:, gen],
            exp * pf)

        gen = 'GeneratorFluctuating_3'  # solar, mv
        exp = pd.Series(data=[0.85 * 2.67, 0 * 2.67], name=gen,
                        index=self.timeseries.timeindex)
        assert_series_equal(
            self.timeseries.generators_active_power.loc[:, gen], exp)
        pf = -tan(acos(0.9))
        assert_series_equal(
            self.timeseries.generators_reactive_power.loc[:, gen],
            exp * pf)

        gen = 'GeneratorFluctuating_20'  # solar, lv
        exp = pd.Series(data=[0.85 * 0.005, 0 * 0.005], name=gen,
                        index=self.timeseries.timeindex)
        assert_series_equal(
            self.timeseries.generators_active_power.loc[:, gen], exp)
        pf = -tan(acos(0.95))
        assert_series_equal(
            self.timeseries.generators_reactive_power.loc[:, gen],
            exp * pf)

        load = 'Load_retail_MVGrid_1_Load_aggregated_retail_' \
               'MVGrid_1_1'  # retail, mv
        exp = pd.Series(data=[0.15 * 0.31, 1.0 * 0.31],
                        name=load, index=self.timeseries.timeindex)
        assert_series_equal(
            self.timeseries.loads_active_power.loc[:, load], exp,
            check_exact=False, check_dtype=False)
        pf = tan(acos(0.9))
        assert_series_equal(
            self.timeseries.loads_reactive_power.loc[:, load],
            exp * pf, check_exact=False, check_dtype=False)

        load = 'Load_agricultural_LVGrid_1_2'  # agricultural, lv
        exp = pd.Series(data=[0.1 * 0.0523, 1.0 * 0.0523],
                        name=load, index=self.timeseries.timeindex)
        assert_series_equal(
            self.timeseries.loads_active_power.loc[:, load], exp,
            check_exact=False, check_dtype=False)
        pf = tan(acos(0.95))
        assert_series_equal(
            self.timeseries.loads_reactive_power.loc[:, load],
            exp * pf, check_exact=False, check_dtype=False)

        load = 'Load_residential_LVGrid_3_3'  # residential, lv
        exp = pd.Series(data=[0.1 * 0.001209, 1.0 * 0.001209],
                        name=load, index=self.timeseries.timeindex)
        assert_series_equal(
            self.timeseries.loads_active_power.loc[:, load], exp,
            check_exact=False, check_dtype=False)
        pf = tan(acos(0.95))
        assert_series_equal(
            self.timeseries.loads_reactive_power.loc[:, load],
            exp * pf, check_exact=False, check_dtype=False)

        storage = 'StorageUnit_MVGrid_1_Test_storage_1' # storage, mv
        exp = pd.Series(data=[1 * 0.3, -1 * 0.3],
                        name=storage, index=self.timeseries.timeindex)

        assert_series_equal(
            self.timeseries.storage_units_active_power.loc[:, storage], exp,
            check_exact=False, check_dtype=False)
        pf = -tan(acos(0.9))
        assert_series_equal(
            self.timeseries.storage_units_reactive_power.loc[:, storage],
            exp * pf, check_exact=False, check_dtype=False)

        storage = 'StorageUnit_LVGrid_1_Test_storage_3' # storage, lv
        exp = pd.Series(data=[1 * 0.05, -1 * 0.05],
                        name=storage, index=self.timeseries.timeindex)

        assert_series_equal(
            self.timeseries.storage_units_active_power.loc[:, storage], exp,
            check_exact=False, check_dtype=False)
        pf = -tan(acos(0.95))
        assert_series_equal(
            self.timeseries.storage_units_reactive_power.loc[:, storage],
            exp * pf, check_exact=False, check_dtype=False)

        # remove storages
        self.topology.remove_storage('StorageUnit_MVGrid_1_Test_storage_1')
        self.topology.remove_storage('StorageUnit_MVGrid_1_Test_storage_2')
        self.topology.remove_storage('StorageUnit_LVGrid_1_Test_storage_3')

        # test for only feed-in case
        TimeSeriesControl(edisgo_obj=self, mode='worst-case-feedin')

        # value
        gen = 'Generator_1'  # gas, mv
        exp = pd.Series(data=[1 * 0.775], name=gen,
                        index=self.timeseries.timeindex)
        assert_series_equal(
            self.timeseries.generators_active_power.loc[:, gen], exp)
        pf = -tan(acos(0.9))
        assert_series_equal(
            self.timeseries.generators_reactive_power.loc[:, gen],
            exp * pf)
        load = 'Load_retail_LVGrid_9_14'  # industrial, lv
        exp = pd.Series(data=[0.1 * 0.001222],
                        name=load, index=self.timeseries.timeindex)
        assert_series_equal(
            self.timeseries.loads_active_power.loc[:, load], exp,
            check_exact=False, check_dtype=False)
        pf = tan(acos(0.95))
        assert_series_equal(
            self.timeseries.loads_reactive_power.loc[:, load],
            exp * pf, check_exact=False, check_dtype=False)

        # test for only load case
        TimeSeriesControl(edisgo_obj=self, mode='worst-case-load')

        gen = 'Generator_1'  # gas, mv
        exp = pd.Series(data=[0 * 0.775], name=gen,
                        index=self.timeseries.timeindex)
        assert_series_equal(
            self.timeseries.generators_active_power.loc[:, gen], exp)
        pf = -tan(acos(0.9))
        assert_series_equal(
            self.timeseries.generators_reactive_power.loc[:, gen],
            exp * pf)
        load = 'Load_retail_LVGrid_9_14'  # industrial, lv
        exp = pd.Series(data=[1.0 * 0.001222],
                        name=load, index=self.timeseries.timeindex)
        assert_series_equal(
            self.timeseries.loads_active_power.loc[:, load], exp,
            check_exact=False, check_dtype=False)
        pf = tan(acos(0.95))
        assert_series_equal(
            self.timeseries.loads_reactive_power.loc[:, load],
            exp * pf, check_exact=False, check_dtype=False)

        # test error raising in case of missing load/generator parameter

        gen = 'GeneratorFluctuating_14'
        val_pre = self.topology._generators_df.at[gen, 'bus']
        self.topology._generators_df.at[gen, 'bus'] = None
        with pytest.raises(AttributeError, match=gen):
            ts_control._worst_case_generation(modes=None)
        self.topology._generators_df.at[gen, 'bus'] = val_pre
        gen = 'GeneratorFluctuating_24'
        val_pre = self.topology._generators_df.at[gen, 'p_nom']
        self.topology._generators_df.at[gen, 'p_nom'] = None
        with pytest.raises(AttributeError, match=gen):
            ts_control._worst_case_generation(modes=None)
        self.topology._generators_df.at[gen, 'p_nom'] = val_pre
        load = 'Load_agricultural_LVGrid_1_1'
        val_pre = self.topology._loads_df.at[load, 'peak_load']
        self.topology._loads_df.at[load, 'peak_load'] = None
        with pytest.raises(AttributeError, match=load):
            ts_control._worst_case_load(modes=None)
        self.topology._loads_df.at[load, 'peak_load'] = val_pre

        # test no other generators

    def test_add_loads_timeseries(self):
        """Test method add_loads_timeseries"""
        peak_load = 2.3
        annual_consumption = 3.4
        num_loads = len(self.topology.loads_df)
        # add single load for which timeseries is added
        # test worst-case
        tsc = TimeSeriesControl(edisgo_obj=self, mode='worst-case')
        load_name = self.topology.add_load(load_id=4, bus='Bus_MVStation_1',
                               peak_load=peak_load,
                               annual_consumption=annual_consumption,
                               sector='retail')
        tsc.add_loads_timeseries(load_name)
        active_power_new_load = \
            self.timeseries.loads_active_power.loc[:,
                ['Load_retail_MVGrid_1_4']]
        timeindex = pd.date_range('1/1/1970', periods=2, freq='H')
        assert (self.timeseries.loads_active_power.shape == (2, num_loads+1))
        assert (self.timeseries.loads_reactive_power.shape ==
                (2, num_loads+1))
        assert (active_power_new_load.index == timeindex).all()
        assert np.isclose(
            active_power_new_load.loc[timeindex[0], load_name],
            (0.15*peak_load))
        assert np.isclose(
            active_power_new_load.loc[timeindex[1], load_name],
            peak_load)
        self.topology.remove_load(load_name)
        # test manual
        timeindex = pd.date_range('1/1/2018', periods=24, freq='H')
        generators_active_power, generators_reactive_power, \
            loads_active_power, loads_reactive_power, \
            storage_units_active_power, storage_units_reactive_power = \
            self.create_random_timeseries_for_topology(timeindex)

        tsc = TimeSeriesControl(
            edisgo_obj=self, mode='manual', timeindex=timeindex,
            loads_active_power=loads_active_power,
            loads_reactive_power=loads_reactive_power,
            generators_active_power=generators_active_power,
            generators_reactive_power=generators_reactive_power,
            storage_units_active_power=storage_units_active_power,
            storage_units_reactive_power=storage_units_reactive_power)

        load_name = self.topology.add_load(load_id=4, bus='Bus_MVStation_1',
                               peak_load=peak_load,
                               annual_consumption=annual_consumption,
                               sector='retail')
        new_load_active_power = pd.DataFrame(
            index=timeindex, columns=[load_name],
            data=([peak_load] * len(timeindex)))
        new_load_reactive_power = pd.DataFrame(
            index=timeindex, columns=[load_name],
            data=([peak_load*0.5] * len(timeindex)))
        tsc.add_loads_timeseries(load_name,
                                 loads_active_power=new_load_active_power,
                                 loads_reactive_power=new_load_reactive_power)
        active_power = \
            self.timeseries.loads_active_power[load_name]
        reactive_power = \
            self.timeseries.loads_reactive_power[load_name]
        assert (active_power.values == peak_load).all()
        assert (reactive_power.values == peak_load * 0.5).all()
        assert (self.timeseries.loads_active_power.shape == (24, num_loads+1))
        assert (self.timeseries.loads_reactive_power.shape ==
                (24, num_loads + 1))

        self.topology.remove_load(load_name)
        # test import timeseries from dbs
        timeindex = pd.date_range('1/1/2011', periods=24, freq='H')
        ts_gen_dispatchable = pd.DataFrame({'Generator_1': [0.775] * 24},
                                           index=timeindex)
        tsc = TimeSeriesControl(timeindex=timeindex,
            edisgo_obj=self, timeseries_generation_fluctuating='oedb',
            timeseries_generation_dispatchable=ts_gen_dispatchable,
            timeseries_load='demandlib',
            timeseries_storage_units=storage_units_active_power)

        load_name = self.topology.add_load(load_id=4, bus='Bus_MVStation_1',
                               peak_load=peak_load,
                               annual_consumption=annual_consumption,
                               sector='retail')
        tsc.add_loads_timeseries(load_name)
        active_power = \
            self.timeseries.loads_active_power[load_name]
        reactive_power = \
            self.timeseries.loads_reactive_power[load_name]
        assert np.isclose(active_power.iloc[4],
                          (4.150392788534633e-05*annual_consumption))
        assert np.isclose(reactive_power.iloc[13],
                          (7.937985538711569e-05 * annual_consumption *
                           tan(acos(0.9))))

        assert (self.timeseries.loads_active_power.shape == (24, num_loads+1))
        assert (self.timeseries.loads_reactive_power.shape ==
                (24, num_loads+1))
        self.topology.remove_load(load_name)
        # Todo: add more than one load

    def test_add_generators_timeseries(self):
        """Method add_generators_timeseries"""
        # TEST WORST-CASE
        tsc = TimeSeriesControl(edisgo_obj=self, mode='worst-case')
        num_gens = len(self.topology.generators_df)
        timeindex = pd.date_range('1/1/1970', periods=2, freq='H')
        # add single generator
        p_nom = 1.7
        gen_name = self.topology.add_generator(generator_id=5, p_nom=p_nom,
                                    bus="Bus_BranchTee_LVGrid_1_7",
                                    generator_type='solar')
        tsc.add_generators_timeseries(gen_name)
        assert self.timeseries.generators_active_power.shape == (2, num_gens+1)
        assert self.timeseries.generators_reactive_power.shape == \
            (2, num_gens+1)
        assert \
            (self.timeseries.generators_active_power.index == timeindex).all()
        assert (self.timeseries.generators_active_power.loc[
            timeindex, gen_name].values == [0.85*p_nom, 0]).all()
        assert np.isclose(self.timeseries.generators_reactive_power.loc[
            timeindex, gen_name], [-tan(acos(0.95))*0.85*p_nom, 0]).all()
        # add multiple generators and check
        p_nom2 = 1.3
        gen_name2 = self.topology.add_generator(generator_id=2, p_nom=p_nom2,
                                                bus="Bus_Generator_1",
                                                generator_type='gas')
        p_nom3 = 2.4
        gen_name3 = self.topology.add_generator(generator_id=6, p_nom=p_nom3,
                                                bus="Bus_BranchTee_LVGrid_1_14",
                                                generator_type='hydro')
        tsc.add_generators_timeseries([gen_name2, gen_name3])
        # check expected values
        assert self.timeseries.generators_active_power.shape == (2, num_gens+3)
        assert self.timeseries.generators_reactive_power.shape == (
            2, num_gens + 3)
        assert np.isclose(
            self.timeseries.generators_active_power.loc[
                timeindex, [gen_name2, gen_name3]].values,
            [[p_nom2, p_nom3], [0, 0]]).all()
        assert np.isclose(
            self.timeseries.generators_reactive_power.loc[
                timeindex, [gen_name2, gen_name3]].values,
            [[-p_nom2*tan(acos(0.9)), -p_nom3*tan(acos(0.95))], [0, 0]]).all()
        # remove added generators
        self.topology.remove_generator(gen_name)
        self.topology.remove_generator(gen_name2)
        self.topology.remove_generator(gen_name3)
        # TEST MANUAL
        timeindex = pd.date_range('1/1/2018', periods=24, freq='H')
        generators_active_power, generators_reactive_power, \
            loads_active_power, loads_reactive_power, \
            storage_units_active_power, storage_units_reactive_power = \
            self.create_random_timeseries_for_topology(timeindex)

        tsc = TimeSeriesControl(
            edisgo_obj=self, mode='manual', timeindex=timeindex,
            loads_active_power=loads_active_power,
            loads_reactive_power=loads_reactive_power,
            generators_active_power=generators_active_power,
            generators_reactive_power=generators_reactive_power,
            storage_units_active_power=storage_units_active_power,
            storage_units_reactive_power=storage_units_reactive_power)
        # add single mv solar generator
        gen_name = self.topology.add_generator(generator_id=5, p_nom=p_nom,
                                               bus="Bus_BranchTee_LVGrid_1_7",
                                               generator_type='solar')
        new_gen_active_power = pd.DataFrame(
            index=timeindex, columns=[gen_name],
            data=([p_nom * 0.97] * len(timeindex)))
        new_gen_reactive_power = pd.DataFrame(
            index=timeindex, columns=[gen_name],
            data=([p_nom * 0.5] * len(timeindex)))
        tsc.add_generators_timeseries(
            gen_name, generators_active_power=new_gen_active_power,
            generators_reactive_power=new_gen_reactive_power)
        # check expected values
        assert self.timeseries.generators_active_power.shape == (
            24, num_gens + 1)
        assert self.timeseries.generators_reactive_power.shape == \
            (24, num_gens + 1)
        assert \
            (self.timeseries.generators_active_power.index == timeindex).all()
        assert (self.timeseries.generators_active_power.loc[
                    timeindex, gen_name].values == 0.97 * p_nom).all()
        assert np.isclose(self.timeseries.generators_reactive_power.loc[
                              timeindex, gen_name], p_nom*0.5).all()
        # add multiple generators and check
        p_nom2 = 1.3
        gen_name2 = self.topology.add_generator(generator_id=2, p_nom=p_nom2,
                                                bus="Bus_Generator_1",
                                                generator_type='gas')
        p_nom3 = 2.4
        gen_name3 = self.topology.add_generator(generator_id=6, p_nom=p_nom3,
                                                bus="Bus_BranchTee_LVGrid_1_14",
                                                generator_type='hydro')
        new_gens_active_power = pd.DataFrame(
            index=timeindex, columns=[gen_name2, gen_name3],
            data=(np.array([[p_nom2 * 0.97], [p_nom3 * 0.98]])
                  .repeat(len(timeindex), axis=1).T))
        new_gens_reactive_power = pd.DataFrame(
            index=timeindex, columns=[gen_name2, gen_name3],
            data=(np.array([[p_nom2 * 0.5], [p_nom3 * 0.4]])
                  .repeat(len(timeindex), axis=1).T))
        tsc.add_generators_timeseries(
            [gen_name2, gen_name3],
            generators_active_power=new_gens_active_power,
            generators_reactive_power=new_gens_reactive_power)
        # check expected values
        assert self.timeseries.generators_active_power.shape == (
            24, num_gens + 3)
        assert self.timeseries.generators_reactive_power.shape == (
            24, num_gens + 3)
        assert np.isclose(
            self.timeseries.generators_active_power.loc[
                timeindex, [gen_name2, gen_name3]].values,
            [p_nom2*0.97, p_nom3*0.98]).all()
        assert np.isclose(
            self.timeseries.generators_reactive_power.loc[
                timeindex, [gen_name2, gen_name3]].values,
            [p_nom2*0.5, p_nom3*0.4]).all()
        # remove added generators
        self.topology.remove_generator(gen_name)
        self.topology.remove_generator(gen_name2)
        self.topology.remove_generator(gen_name3)
        # TEST TIMESERIES IMPORT
        # test import timeseries from dbs
        timeindex = pd.date_range('1/1/2011', periods=24, freq='H')
        ts_gen_dispatchable = pd.DataFrame({'Generator_1': [0.775] * 24},
                                           index=timeindex)
        tsc = TimeSeriesControl(timeindex=timeindex,
                                edisgo_obj=self,
                                timeseries_generation_fluctuating='oedb',
                                timeseries_generation_dispatchable=ts_gen_dispatchable,
                                timeseries_load='demandlib',
                                timeseries_storage_units=storage_units_active_power)

        # add single mv solar generator
        gen_name = self.topology.add_generator(generator_id=5, p_nom=p_nom,
                                               bus="Bus_BranchTee_LVGrid_1_7",
                                               generator_type='solar',
                                               weather_cell_id=1122075)
        tsc.add_generators_timeseries(gen_name)
        assert (self.timeseries.generators_active_power.shape == (
                24, num_gens + 1))
        assert (self.timeseries.generators_reactive_power.shape ==
                (24, num_gens + 1))
        #Todo: check values

        # add multiple generators and check
        p_nom2 = 1.3
        gen_name2 = self.topology.add_generator(generator_id=2, p_nom=p_nom2,
                                                bus="Bus_Generator_1",
                                                generator_type='gas')
        p_nom3 = 2.4
        gen_name3 = self.topology.add_generator(generator_id=6, p_nom=p_nom3,
                                                bus="Bus_BranchTee_LVGrid_1_14",
                                                generator_type='hydro')
        new_gens_active_power = pd.DataFrame(
            index=timeindex, columns=[gen_name2, gen_name3],
            data=(np.array([[p_nom2 * 0.97], [p_nom3 * 0.98]])
                  .repeat(len(timeindex), axis=1).T))
        tsc.add_generators_timeseries(
            [gen_name2, gen_name3],
            timeseries_generation_dispatchable=new_gens_active_power)
        assert (self.timeseries.generators_active_power.shape == (
            24, num_gens + 3))
        assert (self.timeseries.generators_reactive_power.shape ==
                (24, num_gens + 3))
        assert np.isclose(
            self.timeseries.generators_active_power.loc[
                timeindex, [gen_name2, gen_name3]].values,
            [p_nom2*0.97, p_nom3*0.98]).all()
        assert np.isclose(
            self.timeseries.generators_reactive_power.loc[
                timeindex, [gen_name2, gen_name3]].values,
            [-tan(acos(0.9))*p_nom2*0.97, -tan(acos(0.95))*p_nom3*0.98]).all()
        # check values when reactive power is inserted as timeseries
        new_gens_reactive_power = pd.DataFrame(
            index=timeindex, columns=[gen_name2, gen_name3],
            data=(np.array([[p_nom2 * 0.54], [p_nom3 * 0.45]])
                  .repeat(len(timeindex), axis=1).T))
        tsc.add_generators_timeseries([gen_name2, gen_name3],
            timeseries_generation_dispatchable=new_gens_active_power,
            generation_reactive_power=new_gens_reactive_power)
        assert (self.timeseries.generators_active_power.shape == (
            24, num_gens + 3))
        assert (self.timeseries.generators_reactive_power.shape ==
                (24, num_gens + 3))
        assert np.isclose(
            self.timeseries.generators_active_power.loc[
                timeindex, [gen_name2, gen_name3]].values,
            [p_nom2 * 0.97, p_nom3 * 0.98]).all()
        assert np.isclose(
            self.timeseries.generators_reactive_power.loc[
                timeindex, [gen_name2, gen_name3]].values,
            [p_nom2 * 0.54, p_nom3 * 0.45]).all()

    def test_check_timeseries_for_index_and_cols(self):
        """Test check_timeseries_for_index_and_cols method"""
        timeindex = pd.date_range('1/1/2017', periods=13, freq='H')
        tsc = TimeSeriesControl(
            edisgo_obj=self, mode='manual', timeindex=timeindex)
        added_comps = ['Comp_1', 'Comp_2']
        timeseries_with_wrong_timeindex = pd.DataFrame(
            index=timeindex[0:12], columns=added_comps,
            data=np.random.rand(12, len(added_comps)))
        #Todo: check what happens with assertion. Why are strings not the same?
        msg = "Inserted timeseries for the following components have the a " \
              "wrong time index:"
        with pytest.raises(ValueError, match=msg):
            tsc.check_timeseries_for_index_and_cols(
                timeseries_with_wrong_timeindex, added_comps)
        timeseries_with_wrong_comp_names = pd.DataFrame(
            index=timeindex, columns=['Comp_1'],
            data=np.random.rand(13, 1))
        msg = "Columns of inserted timeseries are not the same " \
              "as names of components to be added. Timeseries " \
              "for the following components were tried to be " \
              "added:"
        with pytest.raises(ValueError, match=msg):
            tsc.check_timeseries_for_index_and_cols(
                timeseries_with_wrong_comp_names, added_comps)

    def create_random_timeseries_for_topology(self, timeindex):
        # create random timeseries
        load_names = self.topology.loads_df.index
        loads_active_power = \
            pd.DataFrame(index=timeindex, columns=load_names,
                         data=np.multiply(np.random.rand(len(timeindex),
                                                         len(load_names)),
                                      ([self.topology.loads_df.peak_load] *
                                       len(timeindex))))
        loads_reactive_power = \
            pd.DataFrame(index=timeindex, columns=load_names,
                         data=np.multiply(np.random.rand(len(timeindex),
                                                         len(load_names)),
                                      ([self.topology.loads_df.peak_load] *
                                       len(timeindex))))
        generator_names = self.topology.generators_df.index
        generators_active_power = \
            pd.DataFrame(index=timeindex, columns=generator_names,
                         data=np.multiply(
                             np.random.rand(len(timeindex),
                                            len(generator_names)),
                             ([self.topology.generators_df.p_nom] *
                              len(timeindex))))
        generators_reactive_power = \
            pd.DataFrame(index=timeindex, columns=generator_names,
                         data=np.multiply(
                             np.random.rand(len(timeindex),
                                            len(generator_names)),
                             ([self.topology.generators_df.p_nom] *
                              len(timeindex))))
        storage_names = self.topology.storage_units_df.index
        storage_units_active_power = \
            pd.DataFrame(index=timeindex, columns=storage_names,
                         data=np.multiply(
                             np.random.rand(len(timeindex),
                                            len(storage_names)),
                             ([self.topology.storage_units_df.p_nom] *
                              len(timeindex))))
        storage_units_reactive_power = \
            pd.DataFrame(index=timeindex, columns=storage_names,
                         data=np.multiply(
                             np.random.rand(len(timeindex),
                                            len(storage_names)),
                             ([self.topology.storage_units_df.p_nom] *
                              len(timeindex))))
        return generators_active_power, generators_reactive_power, \
               loads_active_power, loads_reactive_power, \
               storage_units_active_power, storage_units_reactive_power

    #Todo: implement test for methods drop_existing_load_timeseries and
    # drop_existing_generator_timeseries
